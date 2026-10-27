"""Detection and analysis orchestration (no HTTP here, so it is unit-testable)."""
import io
import json
import time
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from vista.app import APP_VERSION, REPORT_SCHEMA_VERSION, config, explain, hier, measure, store
from vista.app.models import manager

GUIDANCE = {
    "plastic": ("Rinse, remove caps/labels where possible and place in dry-recyclables (plastic).", "Check the resin code; films and multilayer packs are often not accepted."),
    "paper": ("Keep dry and clean; place in the paper stream.", "Greasy or wet paper is usually not recyclable."),
    "cardboard": ("Flatten and keep dry; place in the cardboard/paper stream.", "Remove tape and food residue."),
    "glass": ("Place in the glass stream; do not mix with ceramics.", "Handle broken glass carefully."),
    "organic": ("Send to compost/biowaste collection.", "Keep separate from dry recyclables to avoid contamination."),
    "non-recyclable": ("Dispose in residual (general) waste.", "Check local rules for items that look recyclable."),
    "textile": ("Donate if reusable, otherwise use textile collection.", ""),
    "metal": ("Take to a metal / e-waste collection point.", "Sharp edges; do not burn or crush."),
    "battery": ("Take to a battery or e-waste collection point; never put in general waste.", "Fire/chemical hazard: do not open, crush or puncture; tape terminals."),
    "PCB": ("Hand over to authorised e-waste recyclers.", "Contains hazardous substances; do not burn or disassemble."),
    "mobile": ("Use an e-waste take-back or authorised collection point; wipe personal data first.", "Contains a battery: do not open or puncture."),
    "laptop": ("Use an e-waste take-back or authorised collector; wipe personal data first.", "Contains a lithium battery."),
    "cable": ("Take to an e-waste collection point (copper is recoverable).", ""),
    "component": ("Take to an authorised e-waste collector.", "May contain hazardous materials."),
    "keyboard": ("Take to an e-waste collection point.", ""), "mouse": ("Take to an e-waste collection point.", ""),
    "light_bulb": ("Use a lamp/e-waste collection point; do not break (CFL tubes contain mercury).", "Broken fluorescent bulbs: ventilate and avoid skin contact."),
    "medical_waste": ("Do not mix with household waste: hand over to a licensed medical-waste handler.", "Potentially infectious or sharp: do not handle with bare hands."),
}


class UserError(ValueError):
    pass


def read_image(data: bytes, filename: str) -> Image.Image:
    if Path(filename or "").suffix.lower() not in config.ALLOWED_EXT:
        raise UserError("Unsupported file type. Use JPEG, PNG or WEBP.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise UserError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise UserError("The file is not a readable image.") from e
    img = ImageOps.exif_transpose(img).convert("RGB")   # correct phone orientation
    if max(img.size) > 3000:
        img.thumbnail((3000, 3000))
    return img


def _meta_path(image_id: str) -> Path:
    return config.UPLOAD_DIR / f"{image_id}.json"


def detect(img: Image.Image, filename: str, conf: float, iou: float, persist: bool = True) -> dict:
    if manager.yolo is None:
        raise UserError(f"Detector not available: {manager.yolo_error or 'weights missing'}")
    t = time.perf_counter()
    with manager.lock:
        res = manager.yolo.predict(img, conf=conf, iou=iou, imgsz=config.DETECT_IMGSZ, verbose=False)[0]
    dets = []
    for i, b in enumerate(res.boxes):
        x0, y0, x1, y1 = [round(float(v), 1) for v in b.xyxy[0].tolist()]
        dets.append({"id": i, "box_xyxy": [x0, y0, x1, y1], "class_id": int(b.cls), "label": str(res.names[int(b.cls)]),
                     "confidence": round(float(b.conf), 4)})
    image_id = uuid.uuid4().hex[:12] if persist else None
    if persist:
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        img.save(config.UPLOAD_DIR / f"{image_id}.jpg", quality=92)
    meta = {"image_id": image_id, "filename": filename, "width": img.width, "height": img.height, "detections": dets,
            "params": {"conf": conf, "iou": iou, "imgsz": config.DETECT_IMGSZ}}
    if persist:
        _meta_path(image_id).write_text(json.dumps(meta))
    return {**meta, "detector": manager.status()["detector"], "latency_ms": round((time.perf_counter() - t) * 1000),
            "message": None if dets else "No objects detected at this threshold. You can lower the threshold or analyse the full frame."}


def annotate(img: Image.Image, dets: list[dict], selected: int | None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, img.width // 300)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(12, img.width // 50))
    except OSError:
        font = ImageFont.load_default()
    for det in dets:
        sel = det["id"] == selected
        col = (255, 140, 0) if sel else (40, 120, 220)
        x0, y0, x1, y1 = det["box_xyxy"]
        d.rectangle([x0, y0, x1, y1], outline=col, width=lw + (2 if sel else 0))
        d.text((x0 + 3, max(0, y0 - 18)), f"#{det['id']} {det['label']} {det['confidence']:.2f}", fill=col, font=font)
    return out


def _crop(img: Image.Image, box: list[float]) -> tuple[Image.Image, list[int]]:
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise UserError("Invalid bounding box.")
    pw, ph = (x1 - x0) * config.CROP_PAD, (y1 - y0) * config.CROP_PAD
    l, t = max(0, int(x0 - pw)), max(0, int(y0 - ph))
    r, b = min(img.width, int(x1 + pw + 0.999)), min(img.height, int(y1 + ph + 0.999))
    if r - l < 8 or b - t < 8:
        raise UserError("The selected region is too small to analyse.")
    return img.crop((l, t, r, b)), [l, t, r, b]


def _save_png(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def analyze(image_id: str, detection_id: int | None, box: list[float] | None, mass_kg: float | None,
            tau_fine: float | None = None, tau_stream: float | None = None) -> dict:
    mp, ip = _meta_path(image_id), config.UPLOAD_DIR / f"{image_id}.jpg"
    if not mp.exists() or not ip.exists():
        raise UserError("Unknown image. Upload or capture the image again.")
    meta = json.loads(mp.read_text())
    img = Image.open(ip).convert("RGB")
    t0 = time.perf_counter()
    if detection_id is not None:
        det = next((d for d in meta["detections"] if d["id"] == detection_id), None)
        if det is None:
            raise UserError("Unknown detection id.")
    elif box:
        det = {"id": -1, "box_xyxy": [float(v) for v in box], "class_id": None, "label": "user-selected region", "confidence": None}
    else:  # explicit full-frame analysis chosen by the user (no box is invented)
        det = {"id": -1, "box_xyxy": [0.0, 0.0, float(img.width), float(img.height)], "class_id": None, "label": "full frame (user-selected)", "confidence": None}
    crop, crop_xyxy = _crop(img, det["box_xyxy"])
    tau_f, tau_s = tau_fine or config.FINE_TAU, tau_stream or config.STREAM_TAU
    t1 = time.perf_counter()
    with manager.lock:
        try:
            cls = hier.classify(crop, manager, tau_f, tau_s, det_label=det["label"] if det["id"] >= 0 else None)
        except hier.ClassificationUnavailable as e:
            raise UserError(str(e)) from e
        ctx = cls.pop("_ctx")
        warnings = []
        gc, gc_status, gc_err, heat_img, over_img = None, "ok", None, None, None
        try:
            gc = explain.gradcam(ctx["head"].model, ctx["batch"], ctx["class_idx"])
            heat_img, over_img = explain.render(crop, gc["cam"], 0.5)
        except Exception as e:
            gc_status, gc_err = "failed", f"{type(e).__name__}: {e}"
            warnings.append(f"Grad-CAM failed: {gc_err}")
    t2 = time.perf_counter()
    cont = measure.contamination(crop)
    cond, cond_ov = measure.condition(crop)
    cat = cls["fine"]["prediction"]
    imp = measure.impact(cat, mass_kg)
    aid = uuid.uuid4().hex[:12]
    run = config.RUN_DIR / aid
    _save_png(crop, run / "crop.png")
    _save_png(annotate(img, meta["detections"], det["id"] if det["id"] >= 0 else None), run / "annotated.png")
    if heat_img:
        _save_png(heat_img, run / "heatmap.png")
        _save_png(over_img, run / "gradcam_overlay.png")
    if cond_ov:
        _save_png(cond_ov, run / "contour.png")
    if cls["uncertain"]:
        warnings.append("Prediction is below the confidence rule - treat as 'review required'. " + "; ".join(cls["uncertainty_reasons"]))
    if cls["stream"]["source"].startswith("provisional"):
        warnings.append("Stream decision is provisional (no dedicated stream model loaded).")
    if manager.status()["detector"]["kind"].startswith("generic"):
        warnings.append("Detector is a generic COCO model, not a validated waste detector.")
    rec, hazard = GUIDANCE.get(cat, ("Check local guidance.", ""))
    if cls["uncertain"]:
        rec = "Uncertain result: have a person verify the item before segregating. (Model guess: " + rec + ")"
    st = manager.status()
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION, "report_id": aid, "timestamp_utc": store.now(), "application_version": APP_VERSION,
        "models": {"detector": {"weights": config.YOLO_WEIGHTS, "kind": st["detector"]["kind"]},
                   "stream": {"mode": st["stream_classifier"]["mode"], "path": st["stream_classifier"]["path"]},
                   "fine": {"mode": st["fine_classifier"]["mode"], "path": st["fine_classifier"]["path"], "classes": st["fine_classifier"]["classes"]},
                   "preprocessing": "Resize 224x224, ImageNet mean/std"},
        "input": {"filename": meta["filename"], "image_id": image_id, "width": meta["width"], "height": meta["height"]},
        "detection": {"detection_id": det["id"], "box_xyxy": det["box_xyxy"], "label": det["label"], "confidence": det["confidence"],
                      "crop_xyxy": crop_xyxy, "detector_params": meta["params"]},
        "classification": cls,
        "contamination": cont, "condition": cond,
        "carbon_economic": imp,
        "explainability": {"status": gc_status, "error": gc_err, "method": "Grad-CAM", "target_class": cat,
                           "target_layer": gc["target_layer"] if gc else None, "normalization": gc["normalization"] if gc else None,
                           "target_score": gc["target_score"] if gc else None,
                           "region": explain.describe(gc["cam"]) if gc else None,
                           "limitation": "Grad-CAM gives a coarse spatial cue; it does not prove causal reasoning."},
        "recommendation": {"guidance": rec, "handling": hazard,
                           "note": "Based on an image prediction, not verified material composition."},
        "limitations": ["Image-only classification; composition is not verified.", cont["label"], cond["label"],
                        "Carbon/value factors are placeholders unless replaced with cited data."],
        "warnings": warnings,
        "timings": {"detect_crop_ms": round((t1 - t0) * 1000), "classify_explain_ms": round((t2 - t1) * 1000),
                    "total_ms": round((time.perf_counter() - t0) * 1000)},
        "artifacts": {k: f"/api/reports/{aid}/{k}" for k in ("annotated-image", "crop", "gradcam", "heatmap", "contour")},
    }
    store.save(report)
    return report


def explain_target(aid: str, target: str, alpha: float) -> dict:
    rep = store.get(aid)
    if rep is None:
        raise UserError("Unknown analysis id.")
    crop = Image.open(config.RUN_DIR / aid / "crop.png").convert("RGB")
    with manager.lock:
        cls = hier.classify(crop, manager, 0.0, 0.0, det_label=rep["detection"]["label"] if rep["detection"]["detection_id"] >= 0 else None)
        ctx = cls.pop("_ctx")
        if target == "stream":
            if manager.stream is None:
                raise UserError("Stream explanation needs a dedicated stream model (models/resnet18_stream.pt); the current stream decision is derived from fine-grained probabilities.")
            m, labels = manager.stream.model, manager.stream.classes
            idx = labels.index(cls["stream"]["prediction"])
            batch = ctx["batch"]
        else:
            m, labels, batch = ctx["head"].model, ctx["labels"], ctx["batch"]
            cands = [i for i, c in enumerate(labels) if hier.display(c) == target]
            if not cands:
                raise UserError(f"'{target}' is not a class of the routed model.")
            idx = cands[0]
            if ctx["mask_family"] and hier.family_of(labels[idx]) != ctx["mask_family"]:
                raise UserError(f"'{target}' belongs to a different stream than the routed one.")
        gc = explain.gradcam(m, batch, idx)
    heat, over = explain.render(crop, gc["cam"], alpha)
    safe = "".join(ch for ch in target if ch.isalnum())
    _save_png(heat, config.RUN_DIR / aid / f"heatmap_{safe}.png")
    _save_png(over, config.RUN_DIR / aid / f"gradcam_{safe}.png")
    return {"analysis_id": aid, "target": target, "target_layer": gc["target_layer"], "normalization": gc["normalization"],
            "target_score": gc["target_score"], "region": explain.describe(gc["cam"]),
            "overlay_url": f"/api/reports/{aid}/file/gradcam_{safe}.png", "heatmap_url": f"/api/reports/{aid}/file/heatmap_{safe}.png"}
