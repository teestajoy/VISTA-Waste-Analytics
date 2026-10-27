"""One-call pipeline: image -> per-item JSON-ready report."""

import time
from dataclasses import asdict
from typing import Any

from PIL import Image

from vista.assessment.quality import condition_score, contamination_score
from vista.classification.loader import WasteClassifier
from vista.impact.carbon import estimate_impact

STAGES = ["Validate", "Detect", "Crop", "Route", "Classify", "Quality", "Impact", "Explain", "Report"]


def analyze(image: Image.Image, classifier: WasteClassifier | None, *, with_gradcam: bool = True) -> dict[str, Any]:
    """Analyse one item crop. Quality scoring works without a model; the rest needs a classifier."""
    t0 = time.perf_counter()
    image = image.convert("RGB")
    cont, cond = contamination_score(image), condition_score(image)
    report: dict[str, Any] = {
        "contamination": {"score": cont.score, "verdict": cont.verdict},
        "condition": {"score": cond.score, "verdict": cond.verdict},
        "classification": None, "impact": None, "gradcam": None,
    }
    cam = None
    if classifier is not None:
        pred = classifier.predict(image)
        report["classification"] = {
            "family": pred.family, "class": pred.category or pred.label,
            "confidence": round(pred.confidence, 4), "top3": [[n, round(p, 4)] for n, p in pred.top3],
        }
        impact = estimate_impact(pred.category) if pred.category else None
        if impact:
            report["impact"] = {**asdict(impact), "note": "placeholder reference factors; estimate, not measurement"}
        if with_gradcam:
            from vista.explainability.gradcam import GradCAM, top_region
            cam = GradCAM(classifier.model, classifier.target_layer).heatmap(classifier.prepare(image))
            report["gradcam"] = {"layer": "layer4", "peak_region_xyxy": top_region(cam)}
    report["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    report["_cam"] = cam  # numpy map for the UI; stripped from JSON by strip_internal()
    return report


def strip_internal(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if not k.startswith("_")}
