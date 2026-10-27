# VISTA web application (FastAPI + browser UI)

Run: install requirements (PyTorch first, see README.md), then `./run_app.sh` (PowerShell: `$env:PYTHONPATH="src"; python -m uvicorn vista.app.api:app --port 8000`)
and open http://127.0.0.1:8000 . API docs: /docs . Tests: `PYTHONPATH=src pytest tests/test_app.py`.
The UI is plain HTML/JS served by FastAPI (no Node build; Plotly vendored, works offline). Camera needs localhost or https.

## Mapping to the architecture figure
| Stage | Where |
|---|---|
| 1 Camera / upload, 2 resize + ImageNet normalise | `static/app.js`, `service.read_image` (EXIF fix, type/size checks), `models.PREP` |
| 3 YOLOv8n (COCO-pretrained; conf, NMS IoU, imgsz configurable) | `service.detect`: boxes, labels, confidence shown immediately |
| 4 Per-item crop (outward rounding, padding, clipping) | `service._crop` |
| 5 Stream gate (rule first, learned gate later) | `hier.classify`: detector-label rule, then dedicated ResNet18 stream model, else provisional aggregate |
| 6/7 ResNet18 heads (512-d to FC) | combined model with masked softmax per routed stream, or separate `resnet18_msw.pt` / `resnet18_ewaste.pt` |
| 8 Contamination: HSV saturation variance + Laplacian energy, 0..1 | `measure.contamination` |
| 9 Condition: contour solidity = area / convex-hull area, 4 levels | `measure.condition` (contour + hull overlay) |
| 10 CO2e = f x m, value = p x m | `measure.impact` (placeholder factors; mass user-entered or flagged default) |
| 11 Grad-CAM, layer4: alpha_k = mean gradient, ReLU(sum alpha_k A^k) | `explain.py` (hooks removed; failure raises, never a fake map) |
| 12 JSON report, CSV, Excel | `service.analyze`, `exports.py` |
| 13a FastAPI /predict, /health | `api.py` (plus the full /api/... set) |
| 13b dashboard | browser UI; original Streamlit app remains in `dashboard/` |

## Models: honest status
Nothing is pre-trained in this repo. The Model page and `/api/models/status` show each component as ready or missing.
```bash
python scripts/download_trashnet.py --out data/trashnet        # optional (TrashNet is already inside MWCD)
python scripts/prepare_datasets.py --mwcd D:/MWCD --ewaste D:/BD_EWaste --out data/merged   # then read MERGE_REPORT.json
python scripts/train_classifier.py --data data/merged --mode fine      # -> models/resnet18_fine.pt, models/eval/fine/*
python scripts/train_classifier.py --data data/merged --mode stream    # -> models/resnet18_stream.pt (learned gate)
python scripts/evaluate_hierarchical.py                                # hierarchical vs flat, same untouched test split
python scripts/train_yolo.py --data path/to/data.yaml                  # detector fine-tune from COCO yolov8n, if YOLO labels exist
```
Split: seeded, duplicate-group-aware 70/15/15 (`models/eval/split.json`). Training: ImageNet-pretrained ResNet18, class-balanced sampling, label smoothing,
cosine LR, early stopping, temperature calibration fitted on validation only, test evaluated once. Few-shot fallback (no training): put a few images per
class in `models/support/<class>/` and run `python scripts/build_fewshot.py`; the UI labels it "FEW-SHOT PROTOTYPES (not trained)".
Results below tau (0.70 category, 0.60 stream) show "review required".

## Limitations
Rule/provisional stream gate until a stream model is trained; contamination and condition are heuristics; carbon/value factors in `src/vista/impact/carbon.py`
are illustrative placeholders; mass is not measurable from one image; Grad-CAM is a coarse cue. `tests/test_dataset_duplicates.py` has one pre-existing
Windows-path assertion that fails on Linux.
