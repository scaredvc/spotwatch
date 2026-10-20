# Trying the segmentation candidate

`notify_taken: true` now enables a Discord/console message when a previously
free spot becomes confirmed occupied. It uses `occupied_frames` (two by default),
with a red-outlined photo. Repeated occupied readings and startup occupied spots
do not send taken alerts. Unknown readings pause confirmation, as before.
In-memory alert history resets when the monitor restarts.

The usual `config.yaml` still selects box detection and Discord alerts. The
separate `config.segmentation.yaml` selects YOLO11s-seg, mask occlusion, console
notifications and `spotwatch-segmentation.db`. Nothing switches automatically.

Keep collecting photos with the usual command:

```powershell
uv run python -m spotwatch --save-every 60
```

Test the candidate on saved photos while the monitor keeps running:

```powershell
uv run python -m spotwatch.replay --config config.segmentation.yaml --camera front-camera --images runs/front-camera --pattern '20261016*.jpg'
```

Repeat `--pattern` to include other days. Omit it to process all `.jpg` files in
that directory. Reports go to timestamped JSON files under `runs/replays`; use
`--out runs/replays/my-review.json` to choose a report filename. Replay never
creates a camera connection, writes the monitor database, or sends notifications.
It applies the same quality gate and occupancy rules as the live runner. It
reports readings, not simulated alerts: saved images may be minutes apart.
Exclude annotated check images when choosing patterns for raw camera evaluation.

Only when ready to try a live console-only candidate, stop the old monitor first:

```powershell
uv run python -m spotwatch --config config.segmentation.yaml --save-every 60 --save-dir runs/segmentation
```

Do not run two Blink pollers concurrently against the same session/camera.
Restart with the usual command to return to the default behavior. Both configs
use the same camera session and 07:00–19:00 active hours. YOLO downloads the
segmentation weights on first use if `yolo11s-seg.pt` is absent.

## What changed

- Strong, fully visible vehicles use silhouette masks to judge obstruction.
  Their ground point still determines which single spot they occupy.
- Boxes touching within two pixels of any image edge are considered clipped.
  In mask mode they cannot establish occupied from a truncated ground point;
  their box still blocks unsafe free readings when it covers enough of a spot.
  A clipped ground point in or within assignment tolerance of a spot also keeps
  that spot unknown, even when the visible box covers very little of it.
- Weak detections retain the existing conservative box checks. Missing,
  degenerate and non-finite masks fall back to boxes.
- Candidate spot 2 has two lower-boundary vertices moved down 20 pixels to
  include the observed near-curb ground points. Original calibration is intact.

## Model size (October 19)

On the afternoon of October 18 the live monitor sent 11 "free" alerts for spots 3
and 4, the far-curb spots on the left. All were wrong: a black car was parked there
each time. Low sun on the window washed out the far curb (contrast in that strip fell
from ~48 to ~25), and YOLO11n-seg's confidence for the same parked car dropped from
0.80 at 14:06 to about 0.10, below `weak_confidence: 0.15`, so the spot read free.
A cardboard hood and a visor over the camera did not reduce the haze.

Replayed on 17 hand-checked readings (12 "car there", 5 "free"):

| Change | False "free" | Free spots read wrong |
|---|---|---|
| YOLO11n-seg, `weak_confidence: 0.15` (before) | 7 | 0 |
| YOLO11n-seg, `weak_confidence: 0.05` | 2 | 1 |
| YOLO11s-seg, `weak_confidence: 0.15` (used) | 1 | 0 |

Across all 397 good October 18-19 photos the two models disagreed mostly where
YOLO11s-seg found a car that YOLO11n-seg called free or unknown (88 readings on spots
3 and 4). YOLO11s-seg called a spot free where YOLO11n-seg did not three times; two
were correct and one was a car split in half by the telephone pole in spot 6, which
remains an open problem. Inference takes ~0.11s per photo instead of ~0.07s on this
machine, which is negligible at one photo a minute.

## Camera movement check (October 17)

`max_shift_px: 10` rejected 25 of 62 October 17 frames as "camera moved ~10-22px".
That number came from fitting the whole frame and reading the worst corner; features
near the top made the fit tilt. Measured on the street strip around the spots
(100px above the highest spot down to the lowest), the same frames moved 1.4-6.5px,
median 3.7px. The check now uses that strip and reports movement at the spot
corners, so all 62 frames pass.

`max_align_shift_px: 25` additionally moves the spot outlines by the measured
transform (1-6px here). On October 17 this changed 14 readings from unknown to
occupied and none to free. They are borderline cases: e.g. at 14:50 the van's
ground point was 11.3px from spot 1 (limit `ground_tolerance_px: 10`) and 9.1px
after alignment. Spot 1's outline probably sits a little high for vehicles that
park there; consider redrawing it.

## Choosing the ground point (October 17)

A car's ground point is the lowest point of its segmentation outline (the tire
nearest the camera); without an outline, the middle of the box's bottom edge.
Compared on the 89 usable October 15-17 photos, using their saved detections:

| Ground point | Reading changes between photos | Readings differing from lowest point |
|---|---|---|
| Lowest point of the outline (used) | 14 | - |
| Lowest point if inside a spot, else box bottom middle | 14 | 0 |
| Majority of the outline's bottom-edge points | 25 | - |
| Middle of the outline under its center, on the outline's bottom | 45 | 29 (1 to free) |
| Center of the outline | 72 | 191 (86 to free) |

The outline's center sits on the car's body, which the camera angle projects
above and behind the car's own spot, so parked cars read free.

## October 15–16 replay

29 retained photos: 27 usable, 2 quality-rejected. With the candidate:

- Spot 6 is occupied before the October 16 departure and free in all seven
  subsequent saved photos; baseline marks those seven unknown.
- Spot 2 has no free readings in the 27 usable photos. The 19 October 15 photos
  remain unknown because the SUV is clipped; October 16 yields seven occupied
  and one unknown. The naive mask experiment had 13 false-free candidates on
  October 15, including a visually confirmed parked-SUV counterexample.
- Both unusable photos produce unknown for every spot without running inference.

These are development regression cases, not an independent accuracy benchmark.
We still need empty spot 2 examples, additional vehicles and lighting, and real
obstructions. No detector can certify a spot hidden behind another vehicle.
Clipping is a conservative heuristic, not a complete solution to partial visibility.

Tests include synthetic geometry, frozen model outputs, model adapter/config
validation, live runner mode selection and offline quality gating. Run:

```powershell
uv run python -m pytest -q
```
