"""FastAPI application. Run: PYTHONPATH=src uvicorn vista.app.api:app --port 8000"""
import json
import logging
import statistics
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vista.app import APP_VERSION, config, exports, service, store
from vista.app.models import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vista.api")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app):
    manager.load_all()
    yield


app = FastAPI(title="VISTA API", version=APP_VERSION, lifespan=lifespan,
              description="Explainable dual-stream waste / e-waste analysis. Training is never started by the server.")


@app.exception_handler(service.UserError)
async def user_error(_req, exc: service.UserError):
    return JSONResponse(status_code=422, content={"error": str(exc)})


class AnalyzeReq(BaseModel):
    image_id: str
    detection_id: int | None = None
    box_xyxy: list[float] | None = Field(default=None, min_length=4, max_length=4)
    mass_kg: float | None = Field(default=None, gt=0, le=1000)
    fine_tau: float | None = Field(default=None, ge=0, le=1)
    stream_tau: float | None = Field(default=None, ge=0, le=1)


class ExplainReq(BaseModel):
    analysis_id: str
    target: str = "predicted"
    opacity: float = Field(default=0.5, ge=0, le=1)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/health")
def health_alias():
    return health()


@app.get("/api/models/status")
def models_status():
    return manager.status()


@app.get("/api/models/metadata")
def models_metadata():
    return {"status": manager.status(), "evaluation": _evaluation(),
            "thresholds": {"detect_conf": config.DETECT_CONF, "detect_iou": config.DETECT_IOU, "fine_tau": config.FINE_TAU, "stream_tau": config.STREAM_TAU},
            "split_note": "See models/eval/*/metrics.json for split strategy once training has been run."}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(config.DETECT_CONF), iou: float = Form(config.DETECT_IOU), live: bool = Form(False)):
    if not 0 < conf < 1 or not 0 < iou < 1:
        raise HTTPException(422, "conf and iou must be between 0 and 1")
    img = service.read_image(await file.read(), file.filename or "")
    return service.detect(img, file.filename or "capture.jpg", conf, iou, persist=not live)


@app.post("/predict")
async def predict_alias(file: UploadFile = File(...), mass_kg: float | None = Form(None)):
    """Architecture-slide endpoint (13a): image in -> detect -> analyse the top detection (or full frame) -> JSON report."""
    img = service.read_image(await file.read(), file.filename or "")
    d = service.detect(img, file.filename or "upload.jpg", config.DETECT_CONF, config.DETECT_IOU)
    best = max(d["detections"], key=lambda x: x["confidence"])["id"] if d["detections"] else None
    return service.analyze(d["image_id"], best, None, mass_kg)


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    return service.analyze(req.image_id, req.detection_id, req.box_xyxy, req.mass_kg, req.fine_tau, req.stream_tau)


@app.post("/api/explain/gradcam")
def explain_gradcam(req: ExplainReq):
    target = req.target
    if target == "predicted":
        rep = store.get(req.analysis_id)
        if rep is None:
            raise HTTPException(404, "Unknown analysis id")
        target = rep["classification"]["fine"]["prediction"]
    try:
        return service.explain_target(req.analysis_id, target, req.opacity)
    except service.UserError:
        raise
    except Exception as e:
        log.exception("gradcam failed")
        raise HTTPException(500, f"Grad-CAM failed: {e}")


# ---------------- history & reports ----------------------------------------------------------------
def _filters(stream, category, min_conf, max_conf, date_from, date_to):
    return dict(stream=stream, category=category, min_conf=min_conf, max_conf=max_conf, date_from=date_from, date_to=date_to)


@app.get("/api/history")
def history(stream: str | None = None, category: str | None = None, min_conf: float | None = None, max_conf: float | None = None,
            date_from: str | None = None, date_to: str | None = None, limit: int = Query(200, le=2000)):
    return store.query(**_filters(stream, category, min_conf, max_conf, date_from, date_to), limit=limit)


@app.delete("/api/history")
def clear_history():
    store.clear()
    return {"cleared": True}


@app.get("/api/history/export")
def history_export(format: str = "csv", stream: str | None = None, category: str | None = None, min_conf: float | None = None,
                   max_conf: float | None = None, date_from: str | None = None, date_to: str | None = None):
    recs = store.query(**_filters(stream, category, min_conf, max_conf, date_from, date_to), limit=5000)
    rows = [exports.flat_row(store.get(r["id"])) for r in recs]
    if format == "csv":
        return Response(exports.to_csv(rows), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="vista_history.csv"'})
    raise HTTPException(422, "format must be csv (use per-report Excel export for xlsx)")


@app.get("/api/history/{aid}")
def history_one(aid: str):
    r = store.get(aid)
    if r is None:
        raise HTTPException(404, "Unknown analysis id")
    return r


def _need(aid: str) -> dict:
    r = store.get(aid)
    if r is None:
        raise HTTPException(404, "Unknown analysis id")
    return r


@app.get("/api/reports/{aid}/json")
def rep_json(aid: str):
    return Response(exports.to_json(_need(aid)), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="vista_{aid}.json"'})


@app.get("/api/reports/{aid}/csv")
def rep_csv(aid: str):
    return Response(exports.to_csv([exports.flat_row(_need(aid))]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="vista_{aid}.csv"'})


@app.get("/api/reports/{aid}/excel")
def rep_xlsx(aid: str):
    return Response(exports.to_xlsx(_need(aid), _evaluation()), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="vista_{aid}.xlsx"'})


_FILES = {"annotated-image": "annotated.png", "crop": "crop.png", "gradcam": "gradcam_overlay.png", "heatmap": "heatmap.png", "contour": "contour.png"}


@app.get("/api/reports/{aid}/file/{name}")
def rep_named(aid: str, name: str):
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    p = config.RUN_DIR / aid / name
    if not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/reports/{aid}/{kind}")
def rep_image(aid: str, kind: str):
    if kind not in _FILES:
        raise HTTPException(404, "unknown artefact")
    p = config.RUN_DIR / aid / _FILES[kind]
    if not p.is_file():
        raise HTTPException(404, "not available for this analysis")
    return FileResponse(p, media_type="image/png", headers={"Content-Disposition": f'inline; filename="vista_{aid}_{kind}.png"'})


# ---------------- analytics --------------------------------------------------------------------------
def _recs(**kw):
    return store.query(**kw, limit=5000)


def _mean(v):
    v = [x for x in v if x is not None]
    return round(statistics.fmean(v), 4) if v else None


@app.get("/api/analytics/overview")
def overview(stream: str | None = None, category: str | None = None, min_conf: float | None = None, max_conf: float | None = None,
             date_from: str | None = None, date_to: str | None = None):
    r = _recs(**_filters(stream, category, min_conf, max_conf, date_from, date_to))
    if not r:
        return {"empty": True, "message": "No analyses yet. Upload an image or use the camera on the Analyze page."}
    co2 = [x["co2e"] for x in r if x["co2e"] is not None]
    val = [x["value_usd"] for x in r if x["value_usd"] is not None]
    return {"empty": False, "total_analyses": len(r), "images_analyzed": len({x["image_id"] for x in r}),
            "objects_analyzed": len(r), "msw": sum(x["stream"] == "MSW" for x in r), "e_waste": sum(x["stream"] == "E_WASTE" for x in r),
            "uncertain": sum(bool(x["uncertain"]) for x in r), "mean_detector_conf": _mean([x["det_conf"] for x in r]),
            "mean_stream_conf": _mean([x["stream_conf"] for x in r]), "mean_fine_conf": _mean([x["fine_conf"] for x in r]),
            "mean_contamination": _mean([x["contamination"] for x in r]), "mean_condition": _mean([x["condition_score"] for x in r]),
            "total_co2e_kg": round(sum(co2), 4), "total_value_usd": round(sum(val), 4),
            "basis": "Aggregates cover saved analyses matching the filters. Missing values (e.g. full-frame analyses have no detector confidence, "
                     "unavailable condition scores) are excluded from means. CO2e/value use placeholder factors and assumed masses unless mass was entered."}


@app.get("/api/analytics/distributions")
def distributions(stream: str | None = None, category: str | None = None, min_conf: float | None = None, max_conf: float | None = None,
                  date_from: str | None = None, date_to: str | None = None):
    r = _recs(**_filters(stream, category, min_conf, max_conf, date_from, date_to))
    cats: dict = {}
    for x in r:
        c = cats.setdefault(x["category"], {"count": 0, "co2e": 0.0, "value": 0.0})
        c["count"] += 1
        c["co2e"] += x["co2e"] or 0
        c["value"] += x["value_usd"] or 0
    return {"records": [{k: x[k] for k in ("id", "created_at", "stream", "category", "fine_conf", "stream_conf", "contamination", "condition_score", "co2e", "value_usd", "latency_ms", "uncertain")} for x in r],
            "by_category": cats}


def _evaluation() -> dict:
    out: dict = {}
    if config.EVAL_DIR.exists():
        for d in sorted(config.EVAL_DIR.iterdir()):
            if d.is_dir():
                e = {}
                for fn in ("metrics", "history"):
                    f = d / f"{fn}.json"
                    if f.is_file():
                        e[fn] = json.loads(f.read_text())
                if e:
                    out[d.name] = e
    return out


@app.get("/api/analytics/model-performance")
def model_performance():
    ev = _evaluation()
    if not ev:
        return {"empty": True, "message": "No evaluation artefacts found in models/eval/. Run scripts/train_classifier.py and scripts/evaluate.py; nothing is shown until real results exist."}
    return {"empty": False, "models": ev}


# ---------------- front end ----------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
