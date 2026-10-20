"""Vehicle detection. Turns an image into a list of Detections."""

import numpy as np

from spotwatch.occupancy import VEHICLE_CLASSES, Detection


class Detector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class YoloDetector(Detector):
    """Ultralytics YOLO, pretrained on COCO, which already knows cars,
    trucks, motorcycles and buses. No training needed to start."""

    def __init__(self, model_path: str = "yolo11n.pt", min_confidence: float = 0.35,
                 use_masks: bool = False):
        from ultralytics import YOLO  # imported here so tests don't need it

        self.model = YOLO(model_path)
        self.use_masks = use_masks
        if use_masks and self.model.task != "segment":
            raise ValueError("Mask occupancy requires a segmentation model (e.g. yolo11n-seg.pt)")
        self.min_confidence = min_confidence
        names = self.model.names  # {class_id: "car", ...}
        self.class_ids = [i for i, name in names.items() if name in VEHICLE_CLASSES]

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            image,
            classes=self.class_ids,
            conf=self.min_confidence,
            verbose=False,
        )
        detections = []
        height, width = image.shape[:2]
        for result in results:
            masks = result.masks.xy if self.use_masks and result.masks is not None else []
            for index, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = result.names[int(box.cls[0])]
                mask = [tuple(map(float, p)) for p in masks[index]] if index < len(masks) else None
                clipped = x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2
                detections.append(Detection(label, float(box.conf[0]), x1, y1, x2, y2,
                                            mask_polygon=mask, edge_clipped=clipped))
        return detections


class ScriptedDetector(Detector):
    """Returns pre-written detections in order. For tests and dry runs."""

    def __init__(self, script: list[list[Detection]]):
        self.script = list(script)
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return result
