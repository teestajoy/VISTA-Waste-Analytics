"""Fine-tune YOLOv8 on a waste detection dataset in YOLO format (e.g. TACO exported to YOLO) and save eval metrics.
  python scripts/train_yolo.py --data path/to/data.yaml --epochs 50
Then set VISTA_YOLO=runs/detect/vista/weights/best.pt before starting the app."""
import argparse, json
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--base", default="yolov8n.pt")
ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
m = YOLO(a.base)
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, seed=a.seed, project="runs/detect", name="vista", exist_ok=True, augment=True, patience=15)
best = YOLO("runs/detect/vista/weights/best.pt"); r = best.val(data=a.data, split="test", imgsz=a.imgsz)
out = {"map50": float(r.box.map50), "map50_95": float(r.box.map), "precision": float(r.box.mp), "recall": float(r.box.mr), "classes": list(best.names.values()),
       "weights": "runs/detect/vista/weights/best.pt", "model_size_mb": Path("runs/detect/vista/weights/best.pt").stat().st_size / 1e6}
d = Path("models/eval/detector"); d.mkdir(parents=True, exist_ok=True); (d / "metrics.json").write_text(json.dumps(out, indent=2)); print(out)
