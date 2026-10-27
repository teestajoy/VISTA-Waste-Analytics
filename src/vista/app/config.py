"""Central configuration. Everything is overridable through environment variables (see .env.example)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _p(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else ROOT / p


DATA_DIR = _p("VISTA_DATA_DIR", "data")
UPLOAD_DIR = DATA_DIR / "uploads"
RUN_DIR = DATA_DIR / "runs"
DB_PATH = DATA_DIR / "vista.sqlite"
EVAL_DIR = _p("VISTA_EVAL_DIR", "models/eval")

# Model artefacts. Missing files are reported in /api/models/status, never silently replaced.
YOLO_WEIGHTS = os.environ.get("VISTA_YOLO", "yolov8n.pt")          # waste-tuned weights go here after training
FINE_CHECKPOINT = _p("VISTA_CHECKPOINT", "models/resnet18_fine.pt")   # combined fine-grained ResNet18
STREAM_CHECKPOINT = _p("VISTA_STREAM_CHECKPOINT", "models/resnet18_stream.pt")  # optional 2-class MSW/E-waste
MSW_HEAD_CHECKPOINT = _p("VISTA_MSW_CHECKPOINT", "models/resnet18_msw.pt")
EWASTE_HEAD_CHECKPOINT = _p("VISTA_EWASTE_CHECKPOINT", "models/resnet18_ewaste.pt")

DETECT_CONF = float(os.environ.get("VISTA_DETECT_CONF", "0.25"))
DETECT_IOU = float(os.environ.get("VISTA_DETECT_IOU", "0.45"))
DETECT_IMGSZ = int(os.environ.get("VISTA_IMGSZ", "640"))
FINE_TAU = float(os.environ.get("VISTA_FINE_TAU", "0.70"))     # confidence rule from the report
STREAM_TAU = float(os.environ.get("VISTA_STREAM_TAU", "0.60"))
CROP_PAD = float(os.environ.get("VISTA_CROP_PAD", "0.08"))     # fractional padding around a box
MAX_UPLOAD_MB = int(os.environ.get("VISTA_MAX_UPLOAD_MB", "10"))
KEEP_IMAGES = os.environ.get("VISTA_KEEP_IMAGES", "1") == "1"   # keep uploaded originals for reports
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
GATE_MODE = os.environ.get("VISTA_GATE", "auto")  # auto | rule | learned  (rule first, learned gate later - see hier.py)
