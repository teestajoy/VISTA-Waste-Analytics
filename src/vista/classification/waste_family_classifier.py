"""Fail-closed interface for the primary MSW versus E-waste classifier.

No verified family checkpoint or reviewed E-waste training set exists yet. The
unavailable implementation therefore returns Uncertain instead of deriving a
family from YOLO labels or the paper/cardboard pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from vista.classification.taxonomy import WasteFamily


CLASS_NAMES = ("MSW", "E_WASTE")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONFIDENCE_THRESHOLD = 0.70
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "waste_family_resnet18_pilot"
    / "best_model.pth"
)
EXPECTED_PREPROCESSING = {
    "color_mode": "RGB",
    "resize": [IMAGE_SIZE, IMAGE_SIZE],
    "tensor": "ToTensor",
    "normalize_mean": list(IMAGENET_MEAN),
    "normalize_std": list(IMAGENET_STD),
}


class FamilyClassifierLoadError(RuntimeError):
    """Raised when a local family checkpoint is missing or incompatible."""


@dataclass(frozen=True)
class WasteFamilyPrediction:
    """A family result with an optional uncalibrated model score."""

    family: WasteFamily | None
    score: float | None
    is_uncertain: bool
    status: str

    def __post_init__(self) -> None:
        if self.family is not None and not isinstance(self.family, WasteFamily):
            raise ValueError("family must be an MSW or E_WASTE WasteFamily value.")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("score must be None or a number between 0 and 1.")
        if not isinstance(self.is_uncertain, bool):
            raise ValueError("is_uncertain must be a boolean.")
        if not self.status.strip():
            raise ValueError("status must explain the prediction state.")
        if self.is_uncertain and self.family is not None:
            raise ValueError("An uncertain prediction cannot contain a family label.")
        if not self.is_uncertain and (self.family is None or self.score is None):
            raise ValueError("A certain prediction requires a family and score.")


class FamilyPredictor(Protocol):
    """Interface implemented by a future reviewed and evaluated family model."""

    def predict(self, image: Image.Image) -> WasteFamilyPrediction:
        """Classify a complete RGB image as MSW, E-waste, or Uncertain."""


class UnavailableWasteFamilyClassifier:
    """Safe placeholder while VISTA lacks verified family training evidence."""

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason or (
            "No verified MSW/E-waste family model is available; reviewed E-waste "
            "training examples are still required."
        )

    def predict(self, image: Image.Image) -> WasteFamilyPrediction:
        if not isinstance(image, Image.Image):
            raise TypeError("Family classifier input must be a Pillow image.")
        return WasteFamilyPrediction(
            family=None,
            score=None,
            is_uncertain=True,
            status=self.reason,
        )


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_factory=None,
) -> nn.Module:
    """Load only a checkpoint with the expected family order and preprocessing."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Waste-family checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise FamilyClassifierLoadError(f"Could not read family checkpoint: {error}") from error
    if not isinstance(checkpoint, dict):
        raise FamilyClassifierLoadError("Family checkpoint must contain a metadata dictionary.")
    metadata_path = checkpoint_path.with_name("training_metadata.json")
    metrics_path = checkpoint_path.with_name("final_test_metrics.json")
    if not metadata_path.is_file() or not metrics_path.is_file():
        raise FamilyClassifierLoadError(
            "Family model artifacts are incomplete; training metadata and final held-out "
            "test metrics are required before inference."
        )
    try:
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FamilyClassifierLoadError(f"Family model reports are unreadable: {error}") from error
    per_class = metrics.get("per_class", {}) if isinstance(metrics, dict) else {}
    if (
        metadata.get("class_names") != list(CLASS_NAMES)
        or metadata.get("architecture") != "torchvision ResNet18"
        or metadata.get("preprocessing") != EXPECTED_PREPROCESSING
        or not metadata.get("test_set_used_only_after_validation_model_selection")
        or set(per_class) != set(CLASS_NAMES)
        or any(int(per_class[name].get("support", 0)) < 1 for name in CLASS_NAMES)
    ):
        raise FamilyClassifierLoadError(
            "Family model reports do not document the expected architecture, preprocessing, "
            "class order, and both held-out classes."
        )
    class_names = tuple(checkpoint.get("class_names", ()))
    if class_names != CLASS_NAMES:
        raise FamilyClassifierLoadError(
            f"Checkpoint class order must be {CLASS_NAMES}; found {class_names}."
        )
    if checkpoint.get("preprocessing") != EXPECTED_PREPROCESSING:
        raise FamilyClassifierLoadError(
            "Checkpoint preprocessing does not match the documented RGB 224x224 "
            "ImageNet normalization."
        )
    state_dict = checkpoint.get("state_dict")
    head_weight = state_dict.get("fc.weight") if isinstance(state_dict, dict) else None
    head_bias = state_dict.get("fc.bias") if isinstance(state_dict, dict) else None
    if (
        not isinstance(head_weight, torch.Tensor)
        or tuple(head_weight.shape) != (len(CLASS_NAMES), 512)
        or not isinstance(head_bias, torch.Tensor)
        or tuple(head_bias.shape) != (len(CLASS_NAMES),)
    ):
        raise FamilyClassifierLoadError(
            "Checkpoint final layer is not a two-output ResNet18 family classifier."
        )
    factory = model_factory or (lambda: models.resnet18(weights=None))
    model = factory()
    try:
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as error:
        raise FamilyClassifierLoadError(
            f"Checkpoint is not compatible with torchvision ResNet18: {error}"
        ) from error
    return model


class WasteFamilyResNet18Classifier:
    """Experimental MSW/E-waste inference from a verified local checkpoint.

    The 0.70 threshold is a provisional demonstration cutoff, not a calibrated
    confidence boundary. Low scores return Uncertain for manual review.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        *,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: str | torch.device | None = None,
        model=None,
    ) -> None:
        if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
            raise TypeError("confidence_threshold must be a number between 0 and 1")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = float(confidence_threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = build_model_from_checkpoint(checkpoint_path) if model is None else model
        output_layer = getattr(self._model, "fc", None)
        if getattr(output_layer, "out_features", None) != len(CLASS_NAMES):
            raise FamilyClassifierLoadError("Family classifier must have exactly two outputs.")
        self._model.to(self.device)
        self._model.eval()
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def predict(self, image: Image.Image) -> WasteFamilyPrediction:
        if not isinstance(image, Image.Image):
            raise TypeError("Family classifier input must be a Pillow image.")
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        score_tensor, index_tensor = probabilities.max(dim=0)
        score = float(score_tensor.item())
        if score < self.confidence_threshold:
            return WasteFamilyPrediction(
                family=None,
                score=score,
                is_uncertain=True,
                status=(
                    f"Uncertain below provisional {self.confidence_threshold:.0%} threshold; "
                    "manual review recommended. Threshold is not calibrated."
                ),
            )
        family = WasteFamily(CLASS_NAMES[int(index_tensor.item())])
        return WasteFamilyPrediction(
            family=family,
            score=score,
            is_uncertain=False,
            status="Experimental family-model output; raw score is not calibrated.",
        )
