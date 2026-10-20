"""Draw spot polygons on a camera frame and save them to config.yaml.

    uv run python -m spotwatch.calibrate --camera side-camera --image recordings/side-camera/20261014-101500.jpg
    uv run python -m spotwatch.calibrate --camera side-camera          # grabs a fresh frame from the camera

In the window:
    left click      add a corner
    right click     undo the last corner
    Enter / Space   finish this spot (needs 3+ corners); you'll pick the spot id in the terminal
    d               delete a spot's polygon on this camera (asks which, in the terminal)
    s               save to config.yaml
    q / Esc         quit (asks if you have unsaved changes)

What to outline: the patch of ROAD where a parked car's tires would sit, not
the space the car body takes up in the picture. Spotwatch marks a spot
occupied when a car's bottom edge (where it touches the road) lands inside the
polygon. For a spot at the edge of the frame, outline the part you can see.

Your config.yaml comments and layout are kept; only the polygons (and any new
spots you create) change. Saving also stores the frame you drew on as
calibration/<camera>.jpg. Spotwatch compares every new frame against it and
stops trusting the camera if the view has shifted (camera bumped or moved).
"""

import argparse
import io
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from spotwatch.config import parse_config

WINDOW = "spotwatch calibrate"
COLORS = [(0, 200, 255), (255, 160, 0), (80, 220, 80), (255, 80, 200), (0, 120, 255), (200, 200, 0), (160, 80, 255)]


# ---------- config.yaml editing (keeps comments) ----------


def make_yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)  # matches how config.yaml is written
    y.width = 4096  # keep polygon lists on one line
    return y


def load_doc(path):
    with open(path, encoding="utf-8") as f:
        return make_yaml().load(f)


def dump_doc(doc) -> str:
    buf = io.StringIO()
    make_yaml().dump(doc, buf)
    return buf.getvalue()


def find_camera(doc, camera_name):
    for cam in doc.get("cameras") or []:
        if cam.get("name") == camera_name:
            return cam
    raise KeyError(f"no camera named {camera_name!r} in config")


def spot_ids(doc) -> list[str]:
    return [str(s["id"]) for s in doc.get("spots") or []]


def flow_list(points) -> CommentedSeq:
    seq = CommentedSeq()
    for x, y in points:
        pair = CommentedSeq([int(x), int(y)])
        pair.fa.set_flow_style()
        seq.append(pair)
    seq.fa.set_flow_style()
    return seq


def set_polygon(doc, camera_name, spot_id, points, new_spot_name=None) -> None:
    """Put a polygon for spot_id on this camera. Creates the spot if needed."""
    if len(points) < 3:
        raise ValueError("a polygon needs at least 3 points")
    spot_id = str(spot_id)
    if spot_id not in spot_ids(doc):
        spot = CommentedMap()
        spot["id"] = spot_id
        spot["name"] = new_spot_name or spot_id
        if doc.get("spots") is None:
            doc["spots"] = CommentedSeq()
        doc["spots"].append(spot)

    cam = find_camera(doc, camera_name)
    polygons = cam.get("polygons")
    if not polygons:  # missing, None, or the empty {} placeholder
        polygons = CommentedMap()
        cam["polygons"] = polygons
    polygons[spot_id] = flow_list(points)


def remove_polygon(doc, camera_name, spot_id) -> bool:
    cam = find_camera(doc, camera_name)
    polygons = cam.get("polygons") or {}
    if str(spot_id) in polygons:
        del polygons[str(spot_id)]
        return True
    return False


def camera_polygons(doc, camera_name) -> dict[str, list[tuple[int, int]]]:
    polygons = find_camera(doc, camera_name).get("polygons") or {}
    return {str(k): [(int(x), int(y)) for x, y in v] for k, v in polygons.items()}


def save_doc(doc, path) -> None:
    """Check the result still loads as a valid config, then write it."""
    text = dump_doc(doc)
    parse_config(make_yaml().load(text))  # raises ConfigError if we broke something
    path = Path(path)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_reference(doc, camera_name, image) -> Path:
    """Keep the frame the polygons were drawn on, for the camera-moved check."""
    from spotwatch.quality import reference_path

    settings = doc.get("settings") or {}
    path = reference_path(settings.get("reference_dir", "calibration"), camera_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


# ---------- drawing ----------


def draw(image, polygons: dict, current: list, frame_note: str = "") -> np.ndarray:
    out = image.copy()
    overlay = image.copy()
    for i, (spot_id, pts) in enumerate(sorted(polygons.items())):
        color = COLORS[i % len(COLORS)]
        arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [arr], color)
        cv2.polylines(out, [arr], True, color, 2)
        cx, cy = arr.mean(axis=0).astype(int)
        cv2.putText(out, spot_id, (int(cx) - 10, int(cy) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, spot_id, (int(cx) - 10, int(cy) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)

    if current:
        arr = np.array(current, dtype=np.int32)
        cv2.polylines(out, [arr], False, (0, 0, 255), 2)
        for x, y in current:
            cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)

    help_text = "click: corner | right click: undo | Enter: finish spot | d: delete | s: save | q: quit"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, help_text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if frame_note:
        cv2.putText(out, frame_note, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ---------- the interactive part ----------


class Session:
    """Holds the editing state. Kept separate from the OpenCV window so it can be tested."""

    def __init__(self, doc, camera_name):
        self.doc = doc
        self.camera_name = camera_name
        self.current: list[tuple[int, int]] = []
        self.unsaved = False

    def click(self, x, y):
        self.current.append((int(x), int(y)))

    def undo(self):
        if self.current:
            self.current.pop()

    def finish(self, spot_id, new_spot_name=None):
        set_polygon(self.doc, self.camera_name, spot_id, self.current, new_spot_name)
        self.current = []
        self.unsaved = True

    def delete(self, spot_id) -> bool:
        removed = remove_polygon(self.doc, self.camera_name, spot_id)
        self.unsaved = self.unsaved or removed
        return removed

    def polygons(self):
        return camera_polygons(self.doc, self.camera_name)


def ask_spot_id(session: Session):
    """Terminal prompt. Returns (spot_id, new_name or None), or (None, None) to cancel."""
    existing = spot_ids(session.doc)
    drawn = session.polygons()
    print("\nSpots:")
    for sid in existing:
        mark = " (already drawn on this camera, will be replaced)" if sid in drawn else ""
        name = next(s.get("name", sid) for s in session.doc["spots"] if str(s["id"]) == sid)
        print(f"  {sid}: {name}{mark}")
    answer = input("Spot id for this polygon (existing id, a new id, or blank to cancel): ").strip()
    if not answer:
        return None, None
    if answer in existing:
        return answer, None
    name = input(f"New spot {answer!r}. Give it a name (blank = same as id): ").strip()
    return answer, name or None


def grab_frame(config, camera_name):
    from spotwatch.sources import make_source

    camera = next((c for c in config.cameras if c.name == camera_name), None)
    if camera is None:
        raise SystemExit(f"no camera named {camera_name!r} in config")
    source = make_source(camera, config)
    print(f"Grabbing a frame from {camera_name} (Blink can take 10-30 seconds)...")
    frame = source.read(datetime.now())
    source.close()
    if not frame.good:
        raise SystemExit(f"couldn't get a frame: {frame.reason}")
    return frame.image


def run(config_path: Path, camera_name: str, image_path: Path | None):
    doc = load_doc(config_path)
    find_camera(doc, camera_name)  # fail early on a typo

    if image_path:
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"couldn't read image {image_path}")
        note = f"{camera_name} | {image_path.name}"
    else:
        config = parse_config(make_yaml().load(dump_doc(doc)))
        image = grab_frame(config, camera_name)
        note = f"{camera_name} | live frame {datetime.now():%H:%M:%S}"

    session = Session(doc, camera_name)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            session.click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            session.undo()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, image.shape[1], image.shape[0])
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)

    while True:
        cv2.imshow(WINDOW, draw(image, session.polygons(), session.current, note))
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            key = ord("q")  # window closed with the X

        if key in (13, 10, 32):  # Enter / Space
            if len(session.current) < 3:
                print("Need at least 3 corners before finishing a spot.")
                continue
            spot_id, name = ask_spot_id(session)
            if spot_id is None:
                print("Cancelled. Corners kept; keep clicking or right click to undo.")
                continue
            session.finish(spot_id, name)
            print(f"Spot {spot_id} drawn. Press s to save.")
        elif key == ord("d"):
            spot_id = input("Delete which spot's polygon on this camera? ").strip()
            print("Deleted." if spot_id and session.delete(spot_id) else "Nothing deleted.")
        elif key == ord("s"):
            save_doc(session.doc, config_path)
            ref = save_reference(session.doc, camera_name, image)
            session.unsaved = False
            print(f"Saved to {config_path}. Reference frame for the camera-moved check: {ref}")
        elif key in (ord("q"), 27):
            if session.unsaved:
                answer = input("You have unsaved spots. Save before quitting? [Y/n] ").strip().lower()
                if answer != "n":
                    save_doc(session.doc, config_path)
                    ref = save_reference(session.doc, camera_name, image)
                    print(f"Saved to {config_path}. Reference frame: {ref}")
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Draw parking spot polygons on a camera frame.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera", required=True, help="camera name from config.yaml")
    parser.add_argument("--image", help="use this saved frame instead of grabbing a live one")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    run(Path(args.config), args.camera, Path(args.image) if args.image else None)


if __name__ == "__main__":
    main()
