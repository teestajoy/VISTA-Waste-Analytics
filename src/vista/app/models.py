"""Model manager: loads every model once, verifies it, and reports honest readiness."""
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tvm, transforms

from vista.app import config
from vista.classification.loader import EWASTE_HEAD, MSW_HEAD, MWCD_FOLDERS

log = logging.getLogger("vista.models")
PREP = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
STREAMS = ["MSW", "E_WASTE"]


@dataclass
class Loaded:
    name: str
    path: str
    model: nn.Module
    classes: list[str]
    temperature: float = 1.0
    meta: dict = field(default_factory=dict)


def load_resnet(path: Path, name: str) -> Loaded:
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    meta: dict = {}
    if isinstance(raw, nn.Module):
        n = raw.fc.out_features
        return Loaded(name, str(path), raw.eval(), [str(i) for i in range(n)], 1.0, {"format": "full module"})
    state, classes = raw, None
    if isinstance(raw, dict):
        classes = raw.get("classes")
        meta = {k: raw[k] for k in ("version", "trained_on", "epochs", "val_acc", "seed", "split", "few_shot", "support_images") if k in raw}
        temperature = float(raw.get("temperature", 1.0))
        state = raw.get("state_dict") or raw.get("model_state_dict") or raw
    else:
        temperature = 1.0
    n = state["fc.weight"].shape[0]
    default = {2: STREAMS, 6: MSW_HEAD, 7: EWASTE_HEAD, 9: MWCD_FOLDERS}.get(n)
    classes = list(classes or default or [str(i) for i in range(n)])
    if len(classes) != n:
        raise ValueError(f"{path.name}: {len(classes)} class names but fc has {n} outputs")
    m = tvm.resnet18(weights=None, num_classes=n)
    m.load_state_dict(state)
    meta["calibrated"] = temperature != 1.0
    return Loaded(name, str(path), m.eval(), classes, temperature, meta)


class ModelManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()  # Grad-CAM hooks + shared models: serialise inference
        self.device = "cpu"           # classifiers are small; YOLO picks CUDA itself if present
        self.yolo = None
        self.yolo_error: str | None = None
        self.fine: Loaded | None = None
        self.stream: Loaded | None = None
        self.msw: Loaded | None = None
        self.ewaste: Loaded | None = None
        self.errors: dict[str, str] = {}

    def load_all(self) -> None:
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(config.YOLO_WEIGHTS)
        except Exception as e:  # reported, not hidden
            self.yolo_error = f"{type(e).__name__}: {e}"
            log.warning("YOLO unavailable: %s", self.yolo_error)
        for attr, path in (("fine", config.FINE_CHECKPOINT), ("stream", config.STREAM_CHECKPOINT),
                           ("msw", config.MSW_HEAD_CHECKPOINT), ("ewaste", config.EWASTE_HEAD_CHECKPOINT)):
            if Path(path).is_file():
                try:
                    setattr(self, attr, load_resnet(Path(path), attr))
                except Exception as e:
                    self.errors[attr] = f"{type(e).__name__}: {e}"
                    log.error("failed to load %s: %s", path, e)

    # ---- readiness -------------------------------------------------------------------------
    def status(self) -> dict:
        generic = self.yolo is not None and Path(config.YOLO_WEIGHTS).name.lower().startswith(("yolov8", "yolo11", "yolo26")) \
            and "waste" not in str(config.YOLO_WEIGHTS).lower() and "vista" not in str(config.YOLO_WEIGHTS).lower()
        def item(l: Loaded | None, path, hint):
            return {"ready": l is not None, "path": str(path), "classes": l.classes if l else None,
                    "temperature": l.temperature if l else None, "meta": l.meta if l else None,
                    "error": self.errors.get(hint), "how_to_fix": None if l else f"Train it with scripts/train_classifier.py (see README) and save to {path}"}
        fine_ready = self.fine is not None or (self.msw is not None and self.ewaste is not None)
        return {
            "device": self.device,
            "detector": {"ready": self.yolo is not None, "weights": config.YOLO_WEIGHTS, "error": self.yolo_error,
                         "kind": "generic COCO-pretrained (NOT a validated waste detector)" if generic else "custom weights",
                         "how_to_fix": None if self.yolo else "pip install ultralytics; set VISTA_YOLO to a .pt file"},
            "stream_classifier": {**item(self.stream, config.STREAM_CHECKPOINT, "stream"),
                                  "mode": "dedicated" if self.stream else ("provisional (aggregated from fine-grained probabilities)" if fine_ready else "unavailable")},
            "fine_classifier": {**item(self.fine, config.FINE_CHECKPOINT, "fine"), "ready": fine_ready,
                                "mode": ("FEW-SHOT PROTOTYPES (not trained)" if self.fine and (self.fine.meta or {}).get("few_shot") else "combined") if self.fine else ("two heads" if fine_ready else "unavailable")},
            "msw_head": item(self.msw, config.MSW_HEAD_CHECKPOINT, "msw"),
            "ewaste_head": item(self.ewaste, config.EWASTE_HEAD_CHECKPOINT, "ewaste"),
            "explainability": {"ready": fine_ready, "method": "Grad-CAM on ResNet18 layer4"},
        }


manager = ModelManager()
