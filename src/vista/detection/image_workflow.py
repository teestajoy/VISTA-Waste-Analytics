from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from vista.detection.image_input import validate_image
from vista.detection.object_cropper import crop_detection
from vista.detection.object_detector import Detection
from vista.classification.paper_cardboard_classifier import PaperCardboardPrediction


@dataclass(frozen=True)
class DetectedObjectCrop:
    """A detection paired with the image region it identifies."""

    detection: Detection
    crop: Image.Image


@dataclass(frozen=True)
class ClassifiedDetectedObjectCrop:
    """A generic detection paired with its crop and optional pilot prediction."""

    detection: Detection
    crop: Image.Image | None
    prediction: PaperCardboardPrediction | None
    error: str | None = None


class Detector(Protocol):
    """Interface required by the detect-and-crop workflow."""

    def detect(self, path: str | Path) -> list[Detection]:
        """Return detections found in the image at path."""


class CropClassifier(Protocol):
    """Classifier boundary for an already-cropped Pillow image."""

    def predict(self, image: Image.Image) -> PaperCardboardPrediction:
        """Return a paper/cardboard pilot prediction for one crop."""


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


def classify_detections(
    image: Image.Image,
    detections: list[Detection],
    classifier: CropClassifier,
) -> list[ClassifiedDetectedObjectCrop]:
    """Crop and classify each YOLO detection without changing the detection record."""
    rgb_image = image.convert("RGB")
    results: list[ClassifiedDetectedObjectCrop] = []
    for detection in detections:
        try:
            crop = crop_detection(rgb_image, detection)
        except ValueError as error:
            results.append(
                ClassifiedDetectedObjectCrop(detection, None, None, str(error))
            )
            continue
        try:
            prediction = classifier.predict(crop)
        except Exception as error:
            results.append(
                ClassifiedDetectedObjectCrop(detection, crop, None, str(error))
            )
            continue
        results.append(ClassifiedDetectedObjectCrop(detection, crop, prediction))
    return results
