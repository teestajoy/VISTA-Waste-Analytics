"""Load a trained ResNet18 checkpoint and classify one crop.

Accepted checkpoint formats (torch.save): a state_dict, or a dict with
'state_dict' / 'model_state_dict' and optional 'classes'. When no class list is
stored, 6 / 7 outputs map to the MSW / e-waste heads, and 9 outputs use the
alphabetical folder order of the MWCD dataset.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from vista.classification.taxonomy import CATEGORY_FAMILY, WasteCategory

MSW_HEAD = ["plastic", "paper", "cardboard", "glass", "organic", "non-recyclable"]
EWASTE_HEAD = ["metal", "battery", "PCB", "mobile", "laptop", "cable", "component"]
MWCD_FOLDERS = ["battery", "cardboard", "glass", "metal", "organic", "paper", "plastic", "textile", "trash"]
FOLDER_TO_CATEGORY = {"trash": "non-recyclable"}
E_WASTE_NAMES = {"metal", "battery", "PCB", "mobile", "laptop", "cable", "component"}

_PREP = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@dataclass
class Prediction:
    label: str
    category: str | None  # taxonomy category, None if unsupported (e.g. textile)
    family: str  # "MSW" | "E_WASTE" | "UNSUPPORTED"
    confidence: float
    top3: list[tuple[str, float]]


class WasteClassifier:
    def __init__(self, model: nn.Module, classes: list[str]) -> None:
        self.model, self.classes = model.eval(), classes
        self.target_layer = model.layer4

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "WasteClassifier":
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        classes = None
        if isinstance(raw, nn.Module):
            return cls(raw, classes or list(range(raw.fc.out_features)))
        state = raw
        if isinstance(raw, dict):
            classes = raw.get("classes")
            state = raw.get("state_dict") or raw.get("model_state_dict") or raw
        n = state["fc.weight"].shape[0]
        classes = classes or {6: MSW_HEAD, 7: EWASTE_HEAD, 9: MWCD_FOLDERS}.get(n) or [str(i) for i in range(n)]
        model = models.resnet18(weights=None, num_classes=n)
        model.load_state_dict(state)
        return cls(model, list(classes))

    @classmethod
    def from_env(cls) -> "WasteClassifier | None":
        path = os.environ.get("VISTA_CHECKPOINT")
        return cls.from_checkpoint(path) if path and Path(path).is_file() else None

    def prepare(self, crop: Image.Image) -> torch.Tensor:
        return _PREP(crop.convert("RGB")).unsqueeze(0)

    def predict(self, crop: Image.Image) -> Prediction:
        with torch.no_grad():
            probs = torch.softmax(self.model(self.prepare(crop)), 1)[0]
        top = probs.topk(min(3, len(self.classes)))
        label = self.classes[int(top.indices[0])]
        category = FOLDER_TO_CATEGORY.get(label, label)
        try:
            family = CATEGORY_FAMILY[WasteCategory(category)].value
        except ValueError:
            category, family = None, "UNSUPPORTED"
        return Prediction(label, category, family, float(top.values[0]),
                          [(self.classes[int(i)], float(v)) for v, i in zip(top.values, top.indices)])
