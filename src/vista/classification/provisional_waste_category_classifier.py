"""Inference adapter for the folder-derived provisional category model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from vista.classification.taxonomy import WasteCategory, WasteFamily


CLASS_NAMES = ("paper", "cardboard", "plastic", "glass", "organic", "battery")
CATEGORY_TO_FAMILY = {
    "paper": WasteFamily.MSW,
    "cardboard": WasteFamily.MSW,
    "plastic": WasteFamily.MSW,
    "glass": WasteFamily.MSW,
    "organic": WasteFamily.MSW,
    "battery": WasteFamily.E_WASTE,
}
LABEL_PROVENANCE = "folder_derived_provisional"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_PREPROCESSING = {
    "color_mode": "RGB",
    "resize": [IMAGE_SIZE, IMAGE_SIZE],
    "tensor": "ToTensor",
    "normalize_mean": list(IMAGENET_MEAN),
    "normalize_std": list(IMAGENET_STD),
}
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "models" / "waste_category_provisional" / "best_model.pth"
)


class ProvisionalCategoryModelError(RuntimeError):
    """Raised when the provisional category artifacts are missing or incompatible."""


@dataclass(frozen=True)
class ProvisionalCategoryPrediction:
    category: WasteCategory | None
    candidate_category: WasteCategory
    family_mapping: WasteFamily
    score: float
    is_uncertain: bool
    status: str
    top_candidates: tuple[tuple[WasteCategory, float], ...] = ()


class ProvisionalWasteCategoryClassifier:
    """Six-way image classifier for full frames or crops; outputs are provisional."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        *, confidence_threshold: float = 0.70,
        device: str | torch.device | None = None,
        model=None,
    ) -> None:
        if isinstance(confidence_threshold, bool) or not isinstance(confidence_threshold, (int, float)):
            raise TypeError("confidence_threshold must be a number between 0 and 1")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.threshold = float(confidence_threshold)
        self.class_names = CLASS_NAMES
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model if model is not None else self._load_model(Path(checkpoint_path))
        head = getattr(self.model, "fc", None)
        if getattr(head, "out_features", None) != len(CLASS_NAMES):
            raise ProvisionalCategoryModelError("Category classifier must have six outputs.")
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @staticmethod
    def _load_model(checkpoint_path: Path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Provisional category checkpoint not found: {checkpoint_path}")
        metadata_path = checkpoint_path.with_name("training_metadata.json")
        metrics_path = checkpoint_path.with_name("final_test_metrics.json")
        mapping_path = checkpoint_path.with_name("class_mapping.json")
        if not all(path.is_file() for path in (metadata_path, metrics_path, mapping_path)):
            raise ProvisionalCategoryModelError("Provisional category artifacts are incomplete.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except (OSError, ValueError, RuntimeError) as error:
            raise ProvisionalCategoryModelError(f"Could not read category artifacts: {error}") from error
        expected_mapping = {name: family.value for name, family in CATEGORY_TO_FAMILY.items()}
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
            or mapping.get("category_to_family") != expected_mapping
            or mapping.get("label_provenance") != LABEL_PROVENANCE
            or checkpoint.get("provisional") is not True
            or checkpoint.get("label_provenance") != LABEL_PROVENANCE
            or tuple(checkpoint.get("class_names", ())) != CLASS_NAMES
            or checkpoint.get("preprocessing") != EXPECTED_PREPROCESSING
            or checkpoint.get("category_to_family") != expected_mapping
            or metrics.get("evaluation_status") != "PROVISIONAL folder-derived category labels"
            or set(metrics.get("per_class", {})) != set(CLASS_NAMES)
            or any(int(metrics["per_class"][name].get("support", 0)) < 1 for name in CLASS_NAMES)
            or not metadata.get("test_set_used_only_after_validation_model_selection")
        ):
            raise ProvisionalCategoryModelError(
                "Checkpoint and reports do not identify a compatible folder-derived provisional category model."
            )
        state = checkpoint.get("state_dict")
        weight = state.get("fc.weight") if isinstance(state, dict) else None
        bias = state.get("fc.bias") if isinstance(state, dict) else None
        if (
            not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (len(CLASS_NAMES), 512)
            or not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (len(CLASS_NAMES),)
        ):
            raise ProvisionalCategoryModelError("Checkpoint does not have a six-output ResNet18 category layer.")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError) as error:
            raise ProvisionalCategoryModelError(f"Checkpoint is incompatible with ResNet18: {error}") from error
        return model

    def predict(self, image: Image.Image) -> ProvisionalCategoryPrediction:
        if not isinstance(image, Image.Image):
            raise TypeError("Provisional category classifier input must be a Pillow image.")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        score, index = probabilities.max(dim=0)
        confidence = float(score.item())
        candidate = WasteCategory(CLASS_NAMES[int(index.item())])
        top_scores, top_indices = probabilities.topk(min(3, len(CLASS_NAMES)))
        top_candidates = tuple(
            (WasteCategory(CLASS_NAMES[int(candidate_index.item())]), float(candidate_score.item()))
            for candidate_score, candidate_index in zip(top_scores, top_indices)
        )
        family = CATEGORY_TO_FAMILY[candidate.value]
        uncertain = confidence < self.threshold
        return ProvisionalCategoryPrediction(
            category=None if uncertain else candidate,
            candidate_category=candidate,
            family_mapping=family,
            score=confidence,
            is_uncertain=uncertain,
            status=(
                f"PROVISIONAL score below {self.threshold:.0%}; manual review recommended."
                if uncertain
                else "PROVISIONAL folder-derived category; score is not calibrated or verified."
            ),
            top_candidates=top_candidates,
        )
