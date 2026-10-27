"""Inference adapter for the separate folder-derived provisional family model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from vista.classification.taxonomy import WasteFamily
from vista.classification.waste_family_classifier import (
    CONFIDENCE_THRESHOLD,
    EXPECTED_PREPROCESSING,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


CLASS_NAMES = ("MSW", "E_WASTE")
LABEL_PROVENANCE = "folder_derived_provisional"
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "models" / "waste_family_provisional" / "best_model.pth"
)


class ProvisionalFamilyModelError(RuntimeError):
    """Raised when the experimental checkpoint lacks its provenance evidence."""


class ProvisionalWasteFamilyClassifier:
    """ResNet18 family prediction marked provisional and never verified.

    The score threshold is provisional. A low score yields Uncertain; high scores
    remain explicitly labeled as folder-derived experimental outputs.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        *,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: str | torch.device | None = None,
        model=None,
    ) -> None:
        if not isinstance(confidence_threshold, (int, float)) or isinstance(confidence_threshold, bool):
            raise TypeError("confidence_threshold must be a number between 0 and 1")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.threshold = float(confidence_threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path = Path(checkpoint_path)
        if model is None:
            model = self._load_model(checkpoint_path)
        self.model = model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @staticmethod
    def _load_model(checkpoint_path: Path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Provisional family checkpoint not found: {checkpoint_path}")
        metadata_path = checkpoint_path.with_name("training_metadata.json")
        metrics_path = checkpoint_path.with_name("final_test_metrics.json")
        mapping_path = checkpoint_path.with_name("class_mapping.json")
        if not all(path.is_file() for path in (metadata_path, metrics_path, mapping_path)):
            raise ProvisionalFamilyModelError("Provisional family artifacts are incomplete.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except (OSError, ValueError, RuntimeError) as error:
            raise ProvisionalFamilyModelError(f"Could not read provisional family artifacts: {error}") from error
        if (
            not isinstance(checkpoint, dict)
            or metadata.get("model_status") != "PROVISIONAL"
            or metadata.get("label_provenance") != LABEL_PROVENANCE
            or metadata.get("verified_ground_truth") is not False
            or metadata.get("class_names") != list(CLASS_NAMES)
            or metadata.get("architecture") != "torchvision ResNet18"
            or metadata.get("preprocessing") != EXPECTED_PREPROCESSING
            or mapping.get("class_names") != list(CLASS_NAMES)
            or mapping.get("class_to_index") != {name: index for index, name in enumerate(CLASS_NAMES)}
            or mapping.get("label_provenance") != LABEL_PROVENANCE
            or checkpoint.get("provisional") is not True
            or checkpoint.get("label_provenance") != LABEL_PROVENANCE
            or tuple(checkpoint.get("class_names", ())) != CLASS_NAMES
            or checkpoint.get("preprocessing") != EXPECTED_PREPROCESSING
            or metrics.get("evaluation_status") != "PROVISIONAL folder-derived labels"
            or set(metrics.get("per_class", {})) != set(CLASS_NAMES)
            or any(int(metrics["per_class"][name].get("support", 0)) < 1 for name in CLASS_NAMES)
            or not metadata.get("test_set_used_only_after_validation_model_selection")
        ):
            raise ProvisionalFamilyModelError(
                "Checkpoint and reports do not identify a compatible folder-derived provisional model."
            )
        state = checkpoint.get("state_dict")
        weight = state.get("fc.weight") if isinstance(state, dict) else None
        bias = state.get("fc.bias") if isinstance(state, dict) else None
        if (
            not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (2, 512)
            or not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (2,)
        ):
            raise ProvisionalFamilyModelError("Checkpoint does not have a two-family ResNet18 output layer.")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError) as error:
            raise ProvisionalFamilyModelError(f"Checkpoint is incompatible with ResNet18: {error}") from error
        return model

    def predict(self, image: Image.Image) -> dict[str, object]:
        if not isinstance(image, Image.Image):
            raise TypeError("Provisional family classifier input must be a Pillow image.")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        score, index = probabilities.max(dim=0)
        confidence = float(score.item())
        if confidence < self.threshold:
            return {
                "family": None,
                "score": confidence,
                "is_uncertain": True,
                "status": f"PROVISIONAL score below {self.threshold:.0%}; manual review recommended.",
            }
        return {
            "family": WasteFamily(CLASS_NAMES[int(index.item())]),
            "score": confidence,
            "is_uncertain": False,
            "status": "PROVISIONAL folder-derived output; not verified or calibrated.",
        }
