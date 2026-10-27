"""Inference for VISTA's experimental paper/cardboard pilot classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


CLASS_NAMES = ("paper", "cardboard")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "paper_cardboard_resnet18_pilot"
    / "best_model.pth"
)


class ClassifierLoadError(RuntimeError):
    """Raised when the local pilot checkpoint is missing or incompatible."""


@dataclass(frozen=True)
class PaperCardboardPrediction:
    """A pilot class result and its uncalibrated softmax confidence."""

    label: str | None
    confidence: float
    threshold: float

    @property
    def is_uncertain(self) -> bool:
        return self.label is None


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_factory: Any | None = None,
) -> nn.Module:
    """Load the saved ResNet18 state and verify class order and output shape."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Paper/cardboard pilot checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ClassifierLoadError(f"Could not read pilot checkpoint: {error}") from error

    if not isinstance(checkpoint, dict):
        raise ClassifierLoadError("Pilot checkpoint must contain a metadata dictionary.")
    class_names = tuple(checkpoint.get("class_names", ()))
    if class_names != CLASS_NAMES:
        raise ClassifierLoadError(
            f"Checkpoint class order must be {CLASS_NAMES}; found {class_names}."
        )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ClassifierLoadError("Pilot checkpoint is missing its model state_dict.")
    classifier_weight = state_dict.get("fc.weight")
    classifier_bias = state_dict.get("fc.bias")
    if (
        not isinstance(classifier_weight, torch.Tensor)
        or tuple(classifier_weight.shape) != (2, 512)
        or not isinstance(classifier_bias, torch.Tensor)
        or tuple(classifier_bias.shape) != (2,)
    ):
        raise ClassifierLoadError(
            "Checkpoint final layer is incompatible; expected a 2-class ResNet18 head."
        )

    factory = model_factory or (lambda: models.resnet18(weights=None))
    model = factory()
    try:
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as error:
        raise ClassifierLoadError(
            f"Checkpoint is not compatible with torchvision ResNet18: {error}"
        ) from error
    return model


class PaperCardboardClassifier:
    """Predict paper/cardboard on one RGB crop using the local pilot checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        *,
        confidence_threshold: float = 0.70,
        device: str | torch.device | None = None,
        model: Any | None = None,
    ) -> None:
        if isinstance(confidence_threshold, bool) or not isinstance(
            confidence_threshold, (int, float)
        ):
            raise TypeError("confidence_threshold must be a number between 0 and 1")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = float(confidence_threshold)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = (
            build_model_from_checkpoint(checkpoint_path) if model is None else model
        )
        output_layer = getattr(self._model, "fc", None)
        if getattr(output_layer, "out_features", None) != len(CLASS_NAMES):
            raise ClassifierLoadError("Pilot classifier must have exactly two output classes.")
        self._model.to(self.device)
        self._model.eval()
        self._transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def predict(self, image: Image.Image) -> PaperCardboardPrediction:
        """Return a class only above the provisional, uncalibrated threshold."""
        if not isinstance(image, Image.Image):
            raise TypeError("Classifier input must be a Pillow image.")
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        confidence_tensor, class_index_tensor = probabilities.max(dim=0)
        confidence = float(confidence_tensor.item())
        class_index = int(class_index_tensor.item())
        label = CLASS_NAMES[class_index] if confidence >= self.confidence_threshold else None
        return PaperCardboardPrediction(label, confidence, self.confidence_threshold)
