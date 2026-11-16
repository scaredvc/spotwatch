# Design note: mask-based occupancy

SpotWatch has two ways to turn vehicle detections into spot readings, chosen with
`occupancy_mode` in the config:

- `boxes` (the built-in default): each vehicle is a bounding box.
- `masks` (used in `config.example.yaml`): each vehicle also has a segmentation
  outline from a `-seg` model.

This note explains why masks were added, the safety rules around them, and the
measurements behind each choice. All numbers come from one Blink XT2 camera behind
a second-floor window, on small samples from October 2026, checked by hand. They
are development evidence, not an accuracy benchmark.

## The problem with boxes

From an upstairs window, a parked car's bounding box reaches up over the curb
behind it. A box that covers enough of a spot (`occlusion_overlap`, 0.4 by
default) makes that spot unknown, because the spot might be hidden. That is the
safe answer, but it means a far spot behind a near car can read unknown even when
it is visibly empty. The car's silhouette covers much less of that far spot than
its box does.

## Decisions

### 1. Occupancy is decided by the ground point

A vehicle occupies the spot where its **ground point** lands. The ground point is
the lowest point of its segmentation outline (usually the tire nearest the
camera), or the middle of the box's bottom edge when there is no usable outline.
Each vehicle goes to at most one spot: the one containing its ground point, or the
nearest one within `ground_tolerance_px`.

Compared on the 89 usable photos from October 15-17, using their saved detections:

| Ground point | Reading changes between photos | Readings differing from lowest point |
|---|---|---|
| Lowest point of the outline (used) | 14 | - |
| Lowest point if inside a spot, else box bottom middle | 14 | 0 |
| Majority of the outline's bottom-edge points | 25 | - |
| Middle of the outline under its center, on the outline's bottom | 45 | 29 (1 to free) |
| Center of the outline | 72 | 191 (86 to free) |

The outline's center sits on the car's body, which the camera angle projects above
and behind the car's own spot, so parked cars read free.

### 2. Masks decide obstruction only for strong, complete vehicles

In mask mode, a spot is unknown when a vehicle's silhouette, rather than its box,
covers `occlusion_overlap` of it. Masks are used only where they are safe:

- **Weak detections** (between `weak_confidence` and `min_confidence`) keep the
  box check. A weak box on a spot makes it unknown, never free.
- **Missing, degenerate or non-finite masks** fall back to the box.

### 3. A car cut off by the frame edge can't decide a spot

A box within two pixels of any image edge counts as clipped. Its lowest visible
point is not where the car touches the road, so in mask mode:

- it can't make a spot occupied;
- its box still blocks a free reading when it covers enough of a spot;
- if its ground point is inside or within tolerance of a spot, that spot reads
  unknown even when the visible box covers very little of it.

Replayed on 27 usable photos from October 15-16: a naive mask experiment without
these rules produced 13 false "free" candidates for the spot holding an
edge-clipped SUV, including one visually confirmed parked SUV. With the rules, that
spot had no free readings. Another spot, which box mode left unknown after its car
left, read free in all seven later photos.

### 4. YOLO11s-seg instead of YOLO11n-seg

On October 18 the live monitor sent 11 "free" alerts for two far-curb spots while a
black car was parked there. Low sun on the window washed out the far curb (contrast
in that strip fell from ~48 to ~25), and YOLO11n-seg's confidence for the same
parked car dropped from 0.80 to about 0.10, below `weak_confidence: 0.15`. A hood
and a visor over the camera did not reduce the haze.

Replayed on 17 hand-checked readings (12 "car there", 5 "free"):

| Change | False "free" | Free spots read wrong |
|---|---|---|
| YOLO11n-seg, `weak_confidence: 0.15` (before) | 7 | 0 |
| YOLO11n-seg, `weak_confidence: 0.05` | 2 | 1 |
| YOLO11s-seg, `weak_confidence: 0.15` (used) | 1 | 0 |

Across all 397 good photos from October 18-19, the two models disagreed mostly
where YOLO11s-seg found a car that YOLO11n-seg called free or unknown (88 readings).
Inference took ~0.11 s per photo instead of ~0.07 s on the development machine,
which doesn't matter at one photo a minute.

### 5. Camera movement is measured on the street strip

`max_shift_px: 10` first rejected 25 of 62 frames from October 17 as "camera moved
~10-22px". That came from fitting the whole frame and reading the worst corner;
features near the top of the frame tilted the fit. Measured with ORB features and
RANSAC on the street strip around the spots (100 px above the highest spot down to
the lowest), the same frames moved 1.4-6.5 px, median 3.7 px. The check now uses
that strip and reports movement at the spot corners, so all 62 frames pass.

`max_align_shift_px: 25` also moves the spot outlines by the measured transform
(1-6 px here). That day it changed 14 readings from unknown to occupied and none to
free. They were borderline cases: one van's ground point went from 11.3 px outside
a spot (limit `ground_tolerance_px: 10`) to 9.1 px after alignment.

## How it's tested

[`tests/test_segmentation.py`](../tests/test_segmentation.py) covers the rules above
with synthetic geometry, plus frozen model outputs from development frames in
[`tests/fixtures/segmentation_cases.json`](../tests/fixtures/segmentation_cases.json)
(detections and outlines only, no images). Movement and alignment are covered in
[`tests/test_quality.py`](../tests/test_quality.py).

To compare modes on your own saved photos, run [`replay`](../spotwatch/replay.py)
once per config and compare the JSON reports it writes under `runs/replays/`.
Replay applies the same quality gate and spot rules as the live monitor, but never
connects to a camera, writes the database or sends alerts. It reports readings,
not simulated alerts, because saved photos may be minutes apart.

```
uv run python -m spotwatch.replay --config config.yaml --camera my-camera --images runs/my-camera
```

`--pattern '20261016*.jpg'` (repeatable) picks photos by name and `--out` names the
report. [`config.segmentation.yaml`](../config.segmentation.yaml) is the
configuration used for these trials (console alerts, separate database).

## Open problems

- A car split in two by a telephone pole can still read free. The model comparison
  found one such case.
- No detector can certify a spot hidden behind another vehicle. Those spots read
  unknown; a second camera with a different view is the real fix.
- Clipping is a conservative heuristic, not a full answer to partial visibility.
- The samples above come from one camera over a few days. More lighting, vehicle
  types and real obstructions are needed before quoting accuracy.
