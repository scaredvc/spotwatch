# SpotWatch

[![CI](https://github.com/scaredvc/spotwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/scaredvc/spotwatch/actions/workflows/ci.yml)
![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green)](LICENSE)

Finding street parking is frustrating. SpotWatch monitors designated parking spaces through a fixed camera
and notifies you when one becomes available.

About once a minute it takes a picture with a Blink camera, finds vehicles with a YOLO11 segmentation
model, and checks each curb spot you've outlined. When a spot turns free, it sends an alert with a photo
of that spot highlighted.

> **Not affiliated with Amazon or Blink.** SpotWatch talks to Blink through
> [blinkpy](https://github.com/fronzbot/blinkpy) (MIT), an unofficial library. Blink can change their
> service at any time and break it.

## The hard parts

Finding cars in a picture is what YOLO already does. Most of SpotWatch is about when a picture can be
trusted, because a wrong "spot is free" sends you outside for nothing. When in doubt, a spot reads
**unknown**, and unknown never sends an alert.

| Problem | Approach | Code |
|---|---|---|
| From an upstairs window, a parked car's box spills over the spots behind it. | A car occupies the spot where its **ground point** lands: the lowest point of its segmentation mask, usually the tire nearest the camera. Box overlap is not used to decide occupancy. | [`occupancy.py`](spotwatch/occupancy.py) |
| A spot can be hidden, or a car only half visible. | A spot covered by another vehicle, a car cut off by the frame edge, or a low-confidence detection sitting on the spot (a parked car's score can dip at dusk) reads unknown, not free. | [`occupancy.py`](spotwatch/occupancy.py) |
| Night, dusk, glare and window haze. | Frames that are too dark, blurry or glared are rejected before detection. | [`quality.py`](spotwatch/quality.py) |
| Battery cameras wobble a few pixels between snapshots, and sometimes get bumped. | ORB features matched with RANSAC on the street strip measure the shift against the calibration frame. Small shifts move the spot outlines to match; large ones reject the frame. | [`quality.py`](spotwatch/quality.py) |
| Blink sometimes hands back the previous photo as a new one. | A snapshot counts only if its thumbnail URL changed **and** its bytes differ from the last photo. Otherwise retry, then back off. | [`blink_source.py`](spotwatch/blink_source.py) |
| Two cameras can see the same spot. | Stale and blocked views are ignored, any occupied view wins, and a spot is free only if some camera actually sees it. | [`fusion.py`](spotwatch/fusion.py) |
| One car leaving should be one message. | A per-spot state machine alerts once when a spot turns free and re-arms only after the spot is confirmed taken again. | [`state_machine.py`](spotwatch/state_machine.py) |

## Architecture

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 50, "rankSpacing": 36, "curve": "basis"}}}%%
flowchart TB
    CAL[/"<b>Calibration</b><br/><small>calibrate.py</small><br/>spot outlines<br/>+ reference frame"/]
    SRC["<b>1 · Get a frame</b><br/><small>sources.py</small><br/>Blink, image folder, video<br/><i>stale snapshots rejected</i>"]
    Q(["<b>2 · Quality gate</b><br/><small>quality.py</small><br/>dark? blurry? glare? moved?<br/><i>wobble: outlines re-aligned</i>"])
    DET["<b>3 · Detect vehicles</b><br/><small>detector.py</small><br/>YOLO11 segmentation"]
    OCC["<b>4 · Read each spot</b><br/><small>occupancy.py</small><br/>ground point in outline<br/>→ occupied<br/><i>occluded / weak / clipped</i><br/><i>→ unknown</i>"]
    FUSE["<b>5 · Fuse cameras</b><br/><small>fusion.py</small><br/>any occupied wins<br/><i>stale / blocked ignored</i>"]
    SM["<b>6 · State machine</b><br/><small>state_machine.py</small><br/>alert once when free<br/>re-arm when taken<br/><i>unknown never alerts</i>"]
    NOTIFY["<b>Alert</b><br/><small>notifier.py</small><br/>console / Discord<br/>+ photo"]
    DB[("<b>SQLite log</b><br/><small>storage.py</small>")]

    SRC --> Q
    CAL -. "outlines" .-> Q
    Q -- "usable" --> DET --> OCC --> FUSE --> SM
    Q -. "unusable:<br/>spots unknown" .-> FUSE
    SM -- "turned free" --> NOTIFY
    SM -.-> DB

    classDef gate stroke:#d29922,stroke-width:2px
    classDef out stroke:#3fb950,stroke-width:2px
    classDef side stroke-dasharray:4 3
    class Q gate
    class NOTIFY out
    class CAL,DB side
```

[`runner.py`](spotwatch/runner.py) polls each camera on its own interval and updates a spot only after every
camera due that tick has reported, so one camera can't decide a shared spot on its own. Every snapshot,
reading, state change and alert is logged to SQLite for checking a run afterwards.

Around the live loop are offline tools that use the same quality gate and spot rules:
[`calibrate`](spotwatch/calibrate.py) (click the spot outlines), [`check`](spotwatch/check.py) (read one
frame and save an annotated picture), [`replay`](spotwatch/replay.py) (run saved photos through a config
with no camera, database writes or alerts) and [`record`](spotwatch/record.py) (save frames to build a test set).

## Setup

You need Python 3.11+, a Blink camera that can see the street, and a computer that stays on.

```
git clone https://github.com/scaredvc/spotwatch.git
cd spotwatch
uv sync                     # or: pip install -r requirements.txt
```

**1. Log in to Blink once.** This asks for your email, password and the 2FA PIN Blink sends, then saves
session tokens (never your password) to `blink_session.json`. Keep that file private.

```
uv run python -m spotwatch.blink_setup
```

**2. Make your config.**

```
copy config.example.yaml config.yaml        # cp on macOS/Linux
```

Set `blink_camera` to your camera's name as it appears in the Blink app and list your spots.

**3. Outline the spots.** A window opens on a fresh picture; click the corners of each spot.

```
uv run python -m spotwatch.calibrate --camera my-camera
```

Outline the patch of *road* where a parked car's tires sit, not the car body. Saving writes the outlines
into `config.yaml` and keeps the picture in `calibration/` so SpotWatch can tell later if the camera has
moved. [docs/calibration.md](docs/calibration.md) has the keys and tips for drawing good outlines.

**4. Check it.** Grab one picture and see what each spot reads:

```
uv run python -m spotwatch.check --camera my-camera --detect
```

This saves an annotated picture: green = free, red = occupied, gray = unknown.

**5. Run it.**

```
uv run python -m spotwatch
```

The first run downloads the YOLO model (about 20 MB). To get alerts on your phone, create a Discord
webhook, put it in `.env` as `DISCORD_WEBHOOK_URL=...` (see `.env.example`) and set `notifier: discord`
in `config.yaml`. Test it with `uv run python -m spotwatch.notifier --test`.

## Tuning with your own photos

Add `--save-every 60` to keep a picture a minute (and every alert picture) in `runs/`. Then replay a
folder of saved pictures through a config without touching the camera:

```
uv run python -m spotwatch.replay --config config.yaml --camera my-camera --images runs/my-camera
```

Check alerts against the pictures. When SpotWatch is wrong, the picture usually shows why: a car hidden
behind another, a dark car in glare, a spot outline drawn too high. `docs/segmentation.md` has worked
examples.

## Limits

- **It can only judge what it can see.** A spot hidden behind a parked van, a pole or a passing truck
  reads unknown, or worse, gets missed. Pick a camera spot with a clear, high view of the curb.
- **Lighting.** Low sun on the window, dusk, rain and night all reduce accuracy. Pictures that are too
  dark or blurry are rejected. Set `active_hours` to the hours your view is usable.
- **Far, dark cars** in haze are the hardest case. `yolo11s-seg.pt` (used in `config.example.yaml`)
  handles them much better than the smaller `yolo11n-seg.pt`.
- **Blink limits.** Each snapshot wakes the camera and uses battery; once a minute is about the practical
  limit. Blink may refuse snapshots while the camera is recording a clip or someone has live view open.

## Troubleshooting

| Message | What it means |
|---|---|
| `new snapshot not posted yet` / `same photo as last time` | Blink hasn't produced the new picture yet. SpotWatch retries and backs off; nothing to do. |
| `stale snapshot: thumbnail did not change` | Blink refused a new picture, often because the camera is busy recording or in live view. |
| `camera shifted ~Npx; spot outlines aligned` | Normal wobble, corrected automatically (up to `max_align_shift_px`). |
| `camera moved` / every frame rejected | The camera was bumped. Rerun `calibrate` ([guide](docs/calibration.md#when-the-camera-moves)). |
| `too blurry or hazy` | Dusk, fog or reflection on the window. Readings are unknown until it clears. |

## Privacy

Pictures of your street stay on your computer (`runs/`, `calibration/`, `spotwatch.db`), all gitignored.
**If you use Discord alerts, each alert photo is uploaded to Discord.** If you share pictures for a bug
report, blur license plates, faces and house numbers first.

## License

[AGPL-3.0](LICENSE), matching the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO library
it uses. See [CONTRIBUTING.md](CONTRIBUTING.md) to help out.
