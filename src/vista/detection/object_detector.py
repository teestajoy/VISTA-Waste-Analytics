import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vista.detection.image_input import ImageInputError, validate_image


@dataclass(frozen=True)
class BoundingBox:
    """Pixel coordinates relative to the original image."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class Detection:
    """One detected object and its model confidence."""

    box: BoundingBox
    class_id: int
    class_label: str
    confidence: float


class DetectionOutputError(ValueError):
    """Raised when the model returns inconsistent or invalid detections."""


class YoloObjectDetector:
    """Run a YOLO model on validated image files."""

    def __init__(
        self,
        model_path: str | Path = "yolo26n.pt",
        confidence_threshold: float = 0.25,
        *,
        model: Any | None = None,
    ) -> None:
        if isinstance(confidence_threshold, bool) or not isinstance(
            confidence_threshold, (int, float)
        ):
            raise TypeError("confidence_threshold must be a number between 0 and 1")
        if (
            not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.confidence_threshold = float(confidence_threshold)
        self.model_path = Path(model_path)

        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise ImportError(
                    "YOLO detection requires the 'ultralytics' package."
                ) from error
            model = YOLO(str(self.model_path))

        # Keep the model instance so weights are loaded only once per detector.
        self._model = model

    def detect(self, image_path: str | Path) -> list[Detection]:
        """Validate an image, run inference, and return plain Python results."""
        validate_image(image_path)
        result = self._predict(image_path)
        return self._convert_result(result)

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in one in-memory OpenCV BGR frame (uint8 HWC array)."""
        if not isinstance(frame, np.ndarray):
            raise ImageInputError("Frame must be a NumPy array in BGR format.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ImageInputError("Frame must be a three-channel BGR image.")
        if frame.shape[0] == 0 or frame.shape[1] == 0 or frame.size == 0:
            raise ImageInputError("Frame must not be empty.")
        if frame.dtype != np.uint8:
            raise ImageInputError("Frame dtype must be uint8 for YOLO inference.")

        result = self._predict_frame(frame)
        return self._convert_result(result)

    def _predict(self, image_path: str | Path) -> Any:
        """Small inference boundary, kept separate for straightforward mocking."""
        return self._predict_source(str(image_path))

    def _predict_frame(self, frame: np.ndarray) -> Any:
        """Run inference directly on a validated in-memory image frame."""
        return self._predict_source(frame)

    def _predict_source(self, source: Any) -> Any:
        """Run model inference and select its first result, if one exists."""
        results = self._model.predict(
            source=source,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if results is None:
            return None
        return results[0] if len(results) else None

    @staticmethod
    def _convert_result(result: Any | None) -> list[Detection]:
        if result is None or getattr(result, "boxes", None) is None:
            return []

        boxes = result.boxes
        coordinates = _to_list(boxes.xyxy)
        class_ids = _to_list(boxes.cls)
        confidences = _to_list(boxes.conf)
        if not len(coordinates) == len(class_ids) == len(confidences):
            raise DetectionOutputError(
                "Model output has different counts for boxes, class IDs, and confidences."
            )
        names = getattr(result, "names", None)

        detections = []
        for index, (coordinates_for_box, raw_class_id, raw_confidence) in enumerate(
            zip(coordinates, class_ids, confidences)
        ):
            if len(coordinates_for_box) != 4:
                raise DetectionOutputError(
                    f"Model output box at index {index} must contain four coordinates."
                )
            try:
                x_min, y_min, x_max, y_max = (
                    float(value) for value in coordinates_for_box
                )
                class_id = int(raw_class_id)
                confidence = float(raw_confidence)
            except (TypeError, ValueError, OverflowError) as error:
                raise DetectionOutputError(
                    f"Model output at index {index} contains invalid values."
                ) from error

            if not all(
                math.isfinite(value) for value in (x_min, y_min, x_max, y_max)
            ):
                raise DetectionOutputError(
                    f"Model output box at index {index} contains non-finite coordinates."
                )
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise DetectionOutputError(
                    f"Model output confidence at index {index} must be between 0 and 1."
                )

            detections.append(
                Detection(
                    box=BoundingBox(x_min, y_min, x_max, y_max),
                    class_id=class_id,
                    class_label=str(_class_name(names, class_id)),
                    confidence=confidence,
                )
            )
        return detections


def _to_list(values: Any) -> list[Any]:
    """Convert tensors and array-like values to ordinary Python lists."""
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def _class_name(names: Any, class_id: int) -> Any:
    if names is None:
        return str(class_id)
    if isinstance(names, dict):
        return names.get(class_id, names.get(str(class_id), str(class_id)))
    try:
        return names[class_id]
    except (IndexError, KeyError, TypeError):
        return str(class_id)
