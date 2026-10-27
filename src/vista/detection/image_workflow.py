from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from vista.detection.image_input import validate_image
from vista.detection.object_cropper import crop_detection
from vista.detection.object_detector import Detection


@dataclass(frozen=True)
class DetectedObjectCrop:
    """A detection paired with the image region it identifies."""

    detection: Detection
    crop: Image.Image


class Detector(Protocol):
    """Interface required by the detect-and-crop workflow."""

    def detect(self, path: str | Path) -> list[Detection]:
        """Return detections found in the image at path."""


def detect_and_crop(
    image_path: str | Path,
    detector: Detector,
) -> list[DetectedObjectCrop]:
    """Validate an image, detect objects, and return each detection with its crop."""
    validate_image(image_path)

    with Image.open(image_path) as image:
        image.load()
        detections = detector.detect(image_path)
        return [
            DetectedObjectCrop(
                detection=detection,
                crop=crop_detection(image, detection),
            )
            for detection in detections
        ]
