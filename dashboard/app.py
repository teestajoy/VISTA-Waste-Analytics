"""VISTA's minimal Streamlit interface for provisional waste sorting."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from vista.detection.object_detector import Detection


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
MODEL_PATH = "yolo26n.pt"
CLASSIFIER_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "paper_cardboard_resnet18_pilot"
    / "best_model.pth"
)
PROVISIONAL_FAMILY_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "waste_family_provisional"
    / "best_model.pth"
)
PROVISIONAL_CATEGORY_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "waste_category_provisional"
    / "best_model.pth"
)
PILOT_CONFIDENCE_THRESHOLD = 0.70
PILOT_CLASSIFIER_NOTICE = (
    "Experimental paper/cardboard pilot classifier. The 0.70 confidence cutoff "
    "is provisional and is not calibrated."
)
CATEGORY_GUIDANCE = {
    "paper": (
        "Paper-based material; this image-level model does not verify material purity.",
        "Keep clean, dry paper separate. Follow local collection rules and keep it out of wet or food-contaminated recycling.",
    ),
    "cardboard": (
        "Cardboard-like packaging; coatings or mixed layers are not verified.",
        "Keep dry and flatten when practical. Separate heavily soiled or waxed packaging for local guidance.",
    ),
    "plastic": (
        "Plastic-like item; resin type and local acceptance are not identified.",
        "Check the local recycling rules for the item and material code. Do not assume every plastic is accepted.",
    ),
    "glass": (
        "Glass-like item; the model does not determine glass type or breakage risk.",
        "Handle carefully and use the local glass collection route. Keep broken glass contained and follow local safety guidance.",
    ),
    "organic": (
        "Organic-like image category; the folder-derived training label is not verified as food waste.",
        "Only place confirmed food scraps in a local organics program; keep packaging and non-food material out.",
    ),
    "battery": (
        "Battery-like item; chemistry, charge state, and damage are not determined.",
        "Keep batteries out of ordinary trash and use an approved battery collection point. Isolate damaged batteries and follow local safety advice.",
    ),
}
PROJECT_NAME = "VISTA"
APP_SUBTITLE = "Intelligent Waste Sorting"
PROJECT_DESCRIPTION = (
    "Capture or upload one waste item. VISTA analyzes the image and suggests a "
    "disposal route only when its experimental models are sufficiently certain and agree."
)
FAMILY_RESULT = "Uncertain"
FAMILY_STATUS = (
    "No completed family checkpoint is available for verified predictions; the "
    "separate folder-derived family model remains provisional"
)
VERIFIED_FAMILY_UNAVAILABLE_NOTICE = (
    "The verified family checkpoint is unavailable. The separate folder-derived "
    "provisional baseline below is not a verified substitute."
)
EXPLAINABILITY_NOTICE = (
    "Genuine Grad-CAM is not available yet. VISTA has no verified waste "
    "classifier to explain, so this panel does not display a fabricated heatmap."
)
GENERAL_DETECTION_NOTICE = (
    "General-purpose YOLO detections — not VISTA waste classifications."
)
GENERIC_DETECTION_WARNING = (
    "YOLO detects general objects such as people and household items. A detected "
    "box is not evidence that the image contains waste, and these boxes are never "
    "counted as waste items."
)
CATEGORY_GUIDANCE_NOTICE = (
    "General handling guidance only — not a confirmed material identification, "
    "local recyclability decision, or safety assessment."
)
DISPOSAL_BIN_DEFAULTS = {
    "paper": "Paper recycling bin",
    "cardboard": "Cardboard recycling bin",
    "plastic": "Plastic recycling bin — check local rules",
    "glass": "Glass recycling bin",
    "organic": "Organics / food-waste bin",
    "battery": "Battery / e-waste collection bin",
}
MANUAL_SORTING_MESSAGE = "Unable to classify reliably — manual sorting required."
ANALYTICS_UNAVAILABLE = {
    "Condition": "Unavailable — no validated condition assessment is implemented.",
    "Contamination": "Unavailable — no contamination detection logic is implemented.",
}
LIVE_WORKFLOW_STEPS = (
    "1. Capture one item or upload a clear image.",
    "2. Select Analyze image.",
    "3. Follow a bin recommendation only when both provisional models agree; otherwise sort manually.",
)


class FrameDetector(Protocol):
    """Small detector boundary for unit testing without YOLO weights."""

    def detect_frame(self, frame: np.ndarray) -> list[Detection]: ...


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a non-empty OpenCV BGR frame."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("The selected image is empty or could not be read.")

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if (
        frame is None
        or frame.size == 0
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
    ):
        raise ValueError("The selected file is not a supported, readable image.")
    return frame


def image_bytes_from_capture(capture) -> bytes | None:
    """Safely read the UploadedFile returned by Streamlit upload or camera input."""
    if capture is None:
        return None
    try:
        image_bytes = capture.getvalue()
    except Exception as error:
        raise ValueError(f"Could not read the captured image: {error}") from error
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("The captured image is empty or unavailable.")
    return image_bytes


def load_disposal_bin_mapping(raw_value: str | None = None) -> dict[str, str]:
    """Load category-to-bin names, allowing safe per-project environment overrides."""
    if raw_value is None:
        raw_value = os.environ.get("VISTA_DISPOSAL_BIN_MAP", "")
    mapping = dict(DISPOSAL_BIN_DEFAULTS)
    if not raw_value.strip():
        return mapping
    try:
        overrides = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError("VISTA_DISPOSAL_BIN_MAP must be a JSON object.") from error
    if not isinstance(overrides, dict):
        raise ValueError("VISTA_DISPOSAL_BIN_MAP must be a JSON object.")
    unknown = set(overrides) - set(DISPOSAL_BIN_DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported disposal-bin categories: {', '.join(sorted(unknown))}.")
    for category, bin_name in overrides.items():
        if not isinstance(bin_name, str) or not bin_name.strip():
            raise ValueError(f"Disposal-bin name for {category!r} must be a non-empty string.")
        mapping[category] = bin_name.strip()
    return mapping


def disposal_recommendation(
    family_prediction,
    category_prediction,
    bin_mapping: dict[str, str] | None,
    *,
    family_error: str | None = None,
    category_error: str | None = None,
    mapping_error: str | None = None,
) -> dict[str, object]:
    """Return a cautious recommendation only for certain, agreeing provisional models."""
    category = getattr(category_prediction, "category", None)
    candidate = getattr(category_prediction, "candidate_category", None)
    category_name = getattr(category, "value", category)
    candidate_name = getattr(candidate, "value", candidate)
    score = getattr(category_prediction, "score", None)

    def manual(reason: str) -> dict[str, object]:
        return {
            "category": category_name or candidate_name,
            "category_is_certain": (
                category_name is not None
                and not bool(getattr(category_prediction, "is_uncertain", True))
            ),
            "bin": None,
            "confidence": score,
            "manual_review_required": True,
            "message": MANUAL_SORTING_MESSAGE,
            "explanation": reason,
        }

    if mapping_error:
        return manual(f"The disposal-bin configuration is invalid: {mapping_error}")
    if family_error or category_error or family_prediction is None or category_prediction is None:
        return manual("A required provisional family or category model is unavailable.")
    family = family_prediction.get("family") if isinstance(family_prediction, dict) else None
    family_name = getattr(family, "value", family)
    if (
        family is None
        or bool(family_prediction.get("is_uncertain"))
        or bool(getattr(category_prediction, "is_uncertain", True))
        or category_name is None
    ):
        return manual("At least one provisional model is uncertain; no bin was selected.")
    category_family = getattr(category_prediction, "family_mapping", None)
    category_family_name = getattr(category_family, "value", category_family)
    if category_family_name is None or family_name != category_family_name:
        return manual("The provisional family and category outputs conflict or cannot be reconciled.")
    if category_name not in DISPOSAL_BIN_DEFAULTS or not bin_mapping or category_name not in bin_mapping:
        return manual("This predicted category has no supported disposal-bin mapping.")
    return {
        "category": category_name,
        "category_is_certain": True,
        "bin": bin_mapping[category_name],
        "confidence": score,
        "manual_review_required": False,
        "message": f"PROVISIONAL recommendation: {bin_mapping[category_name]}",
        "explanation": (
            "The experimental category and family outputs agree. Scores are raw and "
            "uncalibrated; confirm the local collection rules before disposal."
        ),
    }


def update_session_results(
    results: dict[str, dict[str, object]] | None,
    image_digest_value: str,
    recommendation: dict[str, object],
    impact_estimates: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Store one completed analysis per unique image digest, without rerun duplicates."""
    updated = dict(results or {})
    updated[image_digest_value] = {
        "category": (
            recommendation.get("category")
            if recommendation.get("category_is_certain")
            else None
        ),
        "bin": recommendation.get("bin"),
        "manual_review_required": bool(recommendation.get("manual_review_required")),
        "impact_estimates": impact_estimates,
    }
    return updated


def update_session_impact_estimates(
    results: dict[str, dict[str, object]] | None,
    image_digest_value: str,
    impact_estimates: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Attach an optional measured-weight estimate to one completed image record."""
    updated = dict(results or {})
    if image_digest_value in updated:
        item = dict(updated[image_digest_value])
        item["impact_estimates"] = impact_estimates
        updated[image_digest_value] = item
    return updated


def session_summary(results: dict[str, dict[str, object]] | None) -> dict[str, object]:
    """Summarize completed unique-image analyses held only in the current session."""
    items = list((results or {}).values())
    by_category: dict[str, int] = {}
    potential_co2e_values = []
    material_value_totals: dict[str, float] = {}
    measured_weight_items = 0
    for item in items:
        category = item.get("category")
        if category:
            by_category[category] = by_category.get(category, 0) + 1
        estimates = item.get("impact_estimates") or {}
        if estimates.get("weight_grams") is not None:
            measured_weight_items += 1
        carbon = estimates.get("carbon") or {}
        if isinstance(carbon.get("value"), (int, float)):
            potential_co2e_values.append(float(carbon["value"]))
        material_value = estimates.get("material_value") or {}
        value = material_value.get("value")
        currency = material_value.get("unit")
        if isinstance(value, (int, float)) and currency:
            material_value_totals[currency] = material_value_totals.get(currency, 0.0) + float(value)
    return {
        "analyzed_items": len(items),
        "recommended_items": sum(bool(item.get("bin")) for item in items),
        "manual_review_items": sum(bool(item.get("manual_review_required")) for item in items),
        "by_category": by_category,
        "measured_weight_items": measured_weight_items,
        "potential_co2e_estimate_count": len(potential_co2e_values),
        "potential_co2e_kg_total": sum(potential_co2e_values) if potential_co2e_values else None,
        "material_value_estimate_count": sum(
            1
            for item in items
            if isinstance((item.get("impact_estimates") or {}).get("material_value", {}).get("value"), (int, float))
        ),
        "material_value_totals_by_currency": material_value_totals,
    }


def image_digest(image_bytes: bytes) -> str:
    """Return a stable key so the UI can identify a newly selected image."""
    return hashlib.sha256(image_bytes).hexdigest()


def annotation_detections(
    detections: list[Detection], *, show_person_boxes: bool = False
) -> list[Detection]:
    """Filter only the image overlay; the full detector output remains available."""
    if show_person_boxes:
        return list(detections)
    return [
        detection
        for detection in detections
        if detection.class_label.strip().casefold() != "person"
    ]


def annotate_detections(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    show_person_boxes: bool = False,
) -> np.ndarray:
    """Draw generic YOLO boxes in orange, optionally including person boxes."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for detection in annotation_detections(
        detections, show_person_boxes=show_person_boxes
    ):
        box = detection.box
        x_min = max(0, min(width - 1, int(round(box.x_min))))
        y_min = max(0, min(height - 1, int(round(box.y_min))))
        x_max = max(0, min(width - 1, int(round(box.x_max))))
        y_max = max(0, min(height - 1, int(round(box.y_max))))
        if x_max <= x_min or y_max <= y_min:
            continue
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 165, 255), 2)
        label = f"YOLO {detection.class_label} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x_min, max(18, y_min - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def analyze_image(
    image_bytes: bytes, detector: FrameDetector
) -> tuple[np.ndarray, list[Detection]]:
    """Decode one image and run the supplied detector on its BGR frame."""
    frame = decode_image_bytes(image_bytes)
    return frame, detector.detect_frame(frame)


def classify_full_image(frame: np.ndarray, classifier):
    """Classify a complete OpenCV BGR frame using the pilot image transform."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.size == 0
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("Whole-image classification requires a non-empty BGR frame.")
    rgb_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return classifier.predict(rgb_image)


def classify_yolo_proposed_regions(
    image: Image.Image, detections: list[Detection], classifier
) -> list[dict[str, object]]:
    """Run the provisional category model on non-person YOLO-proposed crops.

    A YOLO box only proposes pixels to crop. Its generic class label is retained
    for traceability and is never passed to, or used as, the waste category.
    """
    from vista.detection.object_cropper import crop_detection

    results: list[dict[str, object]] = []
    rgb_image = image.convert("RGB")
    for detection in detections:
        proposal_label = detection.class_label.strip()
        if proposal_label.casefold() == "person":
            continue
        try:
            crop = crop_detection(rgb_image, detection)
        except ValueError as error:
            results.append(
                {
                    "detection": detection,
                    "crop": None,
                    "prediction": None,
                    "error": str(error),
                }
            )
            continue
        try:
            prediction = classifier.predict(crop)
        except Exception as error:
            results.append(
                {
                    "detection": detection,
                    "crop": crop,
                    "prediction": None,
                    "error": str(error),
                }
            )
            continue
        results.append(
            {
                "detection": detection,
                "crop": crop,
                "prediction": prediction,
                "error": None,
            }
        )
    return results


def category_crop_prediction_row(result: dict[str, object]) -> dict[str, str]:
    """Format a crop-level provisional category result without trusting YOLO labels."""
    detection = result["detection"]
    prediction = result.get("prediction")
    if result.get("error"):
        return {
            "proposal": detection.class_label,
            "category": "Unavailable — review crop",
            "raw_score": "Unavailable",
            "status": str(result["error"]),
        }
    if prediction is None:
        return {
            "proposal": detection.class_label,
            "category": "Unavailable — review crop",
            "raw_score": "Unavailable",
            "status": "The category model returned no prediction.",
        }
    uncertain = bool(getattr(prediction, "is_uncertain", True))
    category = getattr(prediction, "category", None)
    candidate = getattr(prediction, "candidate_category", None)
    category_name = getattr(category, "value", category)
    candidate_name = getattr(candidate, "value", candidate)
    return {
        "proposal": detection.class_label,
        "category": (
            f"Uncertain — top candidate {candidate_name}; manual review recommended"
            if uncertain or category_name is None
            else f"{category_name} · PROVISIONAL"
        ),
        "raw_score": f"{prediction.score:.1%} · uncalibrated",
        "status": prediction.status,
    }


def whole_image_recommendation_input(category_prediction, category_error=None):
    """Keep experimental YOLO-crop diagnostics out of disposal decisions."""
    return category_prediction, category_error, "whole-image"


def detection_table_rows(detections: list[Detection]) -> list[dict[str, object]]:
    """Format generic detector outputs for a compact UI table."""
    return [
        {
            "Generic label": detection.class_label,
            "Confidence": f"{detection.confidence:.1%}",
            "Class ID": detection.class_id,
        }
        for detection in detections
    ]


def status_card_values(
    has_result: bool, detection_count: int, has_error: bool = False
) -> tuple[tuple[str, str], ...]:
    """Return honest status-card labels and values for the current analysis."""
    return (
        (
            "Detection state",
            "Error" if has_error else "Complete" if has_result else "Ready",
        ),
        ("Generic YOLO boxes", str(detection_count) if has_result else "—"),
        ("Verified family model", "Unavailable"),
        ("Explainability", "Unavailable"),
    )


def main() -> None:
    # Streamlit is imported only at launch, keeping helpers testable without it.
    import streamlit as st

    if str(PROJECT_SRC) not in sys.path:
        sys.path.insert(0, str(PROJECT_SRC))
    from vista.detection.object_detector import YoloObjectDetector

    st.set_page_config(
        page_title="VISTA | Intelligent Waste Sorting",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
        <style>
        :root {
            --vista-bg: #07111f;
            --vista-panel: #0d1b2b;
            --vista-panel-soft: #102238;
            --vista-line: #203750;
            --vista-text: #eaf4f7;
            --vista-muted: #a7bbc9;
            --vista-mint: #69e4c3;
            --vista-blue: #65aaff;
        }
        .stApp { background: radial-gradient(ellipse at 50% -20%, #132944 0, var(--vista-bg) 55%); color: var(--vista-text); }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        html, body, [class*="css"] { font-family: Inter, "Segoe UI", Arial, sans-serif; }
        h1, h2, h3, h4 { font-family: "Space Grotesk", Inter, "Segoe UI", Arial, sans-serif; letter-spacing: -0.025em; }
        h1 { font-size: clamp(2rem, 4vw, 3rem) !important; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: var(--vista-panel); border-right: 1px solid var(--vista-line); }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--vista-line); border-radius: 1rem; background: color-mix(in srgb, var(--vista-panel) 88%, transparent); }
        div[data-testid="stFileUploader"] section { background: var(--vista-panel-soft); border: 1px dashed #3e6480; border-radius: .9rem; }
        div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label { color: var(--vista-text); }
        div.stButton > button[kind="primary"] { background: linear-gradient(100deg, #43d9b2, #4ba8ff); color: #06131e; border: 0; border-radius: .8rem; font-weight: 700; min-height: 3rem; }
        div.stButton > button[kind="primary"]:hover { border: 1px solid #a7f4e1; color: #06131e; filter: brightness(1.05); }
        .vista-brand { display:flex; align-items:center; gap:.8rem; margin:.1rem 0 .3rem; }
        .vista-mark { width:2.65rem; height:2.65rem; display:grid; place-items:center; border:1px solid #397d80; border-radius:.85rem; color:var(--vista-mint); background:linear-gradient(145deg,#123345,#0b1b2c); font-size:1.35rem; }
        .vista-brand-name { font:700 1.35rem/1.1 "Space Grotesk", Inter, "Segoe UI", sans-serif; letter-spacing:.04em; color:var(--vista-text); }
        .vista-brand-subtitle { color:var(--vista-muted); font:500 .92rem/1.35 Inter, "Segoe UI", Arial, sans-serif; }
        .vista-status { margin-left:auto; padding:.38rem .7rem; border:1px solid #276d70; border-radius:999px; color:var(--vista-mint); background:#0c292e; font:600 .72rem Inter, "Segoe UI", sans-serif; letter-spacing:.08em; }
        .vista-intro { color:var(--vista-muted); max-width:48rem; font-size:1.02rem; margin:.2rem 0 1.4rem; }
        .vista-result-label { color:var(--vista-mint); text-transform:uppercase; font-size:.72rem; font-weight:700; letter-spacing:.13em; }
        .vista-result-name { font:600 clamp(1.6rem, 3vw, 2.25rem)/1.15 "Space Grotesk", Inter, "Segoe UI", sans-serif; margin:.35rem 0 .7rem; color:var(--vista-text); }
        .vista-muted { color:var(--vista-muted); }
        .vista-accent { color:var(--vista-blue); }
        @media (max-width: 760px) {
            .block-container { padding: 1.15rem 1rem 2rem; }
            .vista-status { font-size:.62rem; padding:.32rem .48rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_detector() -> YoloObjectDetector:
        return YoloObjectDetector(model_path=MODEL_PATH)

    st.markdown(
        '<div class="vista-brand"><div class="vista-mark">♻</div>'
        '<div><div class="vista-brand-name">VISTA</div>'
        '<div class="vista-brand-subtitle">Intelligent Waste Sorting</div></div>'
        '<div class="vista-status">EXPERIMENTAL MODELS</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Sort an item")
    st.markdown(f'<p class="vista-intro">{PROJECT_DESCRIPTION}</p>', unsafe_allow_html=True)

    @st.cache_resource
    def load_classifier():
        from vista.classification.paper_cardboard_classifier import PaperCardboardClassifier

        return PaperCardboardClassifier(
            checkpoint_path=CLASSIFIER_CHECKPOINT,
            confidence_threshold=PILOT_CONFIDENCE_THRESHOLD,
        )

    @st.cache_resource
    def load_family_classifier():
        from vista.classification.waste_family_classifier import (
            WasteFamilyResNet18Classifier,
            UnavailableWasteFamilyClassifier,
        )

        try:
            return WasteFamilyResNet18Classifier()
        except (OSError, RuntimeError, ValueError) as error:
            return UnavailableWasteFamilyClassifier(str(error))

    @st.cache_resource
    def load_provisional_family_classifier():
        from vista.classification.provisional_waste_family_classifier import (
            ProvisionalWasteFamilyClassifier,
        )

        try:
            return ProvisionalWasteFamilyClassifier(
                checkpoint_path=PROVISIONAL_FAMILY_CHECKPOINT,
                confidence_threshold=0.70,
            ), None
        except (OSError, RuntimeError, ValueError) as error:
            return None, str(error)

    @st.cache_resource
    def load_provisional_category_classifier():
        from vista.classification.provisional_waste_category_classifier import (
            ProvisionalWasteCategoryClassifier,
        )

        try:
            return ProvisionalWasteCategoryClassifier(
                checkpoint_path=PROVISIONAL_CATEGORY_CHECKPOINT,
                confidence_threshold=0.70,
            ), None
        except (OSError, RuntimeError, ValueError) as error:
            return None, str(error)

    _render_detection(
        st, load_detector, load_classifier, load_family_classifier,
        load_provisional_family_classifier, load_provisional_category_classifier,
    )
    with st.expander("About VISTA"):
        _render_about(st)


def _render_overview(st) -> None:
    st.subheader("Project overview")
    st.write(
        "VISTA is being developed to triage municipal and electronic waste. "
        "This prototype includes general-purpose object detection and separate "
        "folder-derived provisional family/category classifiers; no waste label is verified."
    )
    cards = status_card_values(False, 0)
    _render_status_cards(st, cards)
    st.info("Choose **Live Detection** to upload an image or take a webcam snapshot.")
    st.markdown(f"**Current detection model:** {GENERAL_DETECTION_NOTICE}")
    _render_session_summary(st)


def _render_session_summary(st) -> None:
    summary = session_summary(st.session_state.get("vista_session_results", {}))
    st.subheader("This-session summary")
    st.caption("Counts completed image analyses once per unique image; this summary is not saved after the browser session.")
    columns = st.columns(3)
    columns[0].metric("Analyzed images", summary["analyzed_items"])
    columns[1].metric("With provisional bin recommendation", summary["recommended_items"])
    columns[2].metric("Uncertain / manual review", summary["manual_review_items"])
    for category, count in sorted(summary["by_category"].items()):
        st.write(f"**Certain provisional {category} category predictions:** {count}")


def _render_analytics(st) -> None:
    st.subheader("Session analytics")
    _render_session_summary(st)
    summary = session_summary(st.session_state.get("vista_session_results", {}))
    st.subheader("Optional weight-based estimates")
    st.caption(
        "Provisional potential estimates require a manually measured weight and documented "
        "material factors. The camera does not measure weight; these values are not measured savings or guaranteed prices."
    )
    carbon_value = summary["potential_co2e_kg_total"]
    carbon_label = (
        f"{carbon_value:.4f} kg CO2e potential"
        if carbon_value is not None else "Estimate unavailable"
    )
    value_totals = summary["material_value_totals_by_currency"]
    value_label = (
        ", ".join(f"{currency} {value:.2f}" for currency, value in sorted(value_totals.items()))
        if value_totals else "Estimate unavailable"
    )
    columns = st.columns(4)
    columns[0].metric("Items with measured weight", summary["measured_weight_items"])
    columns[1].metric("Potential CO2e total", carbon_label)
    columns[2].metric("CO2e estimates available", summary["potential_co2e_estimate_count"])
    columns[3].metric("Estimated material value", value_label)
    st.caption(
        f"Material-value estimates available: {summary['material_value_estimate_count']}. "
        "Only estimates actually computed from configured factors are included."
    )
    st.subheader("Additional waste analysis")
    _render_unavailable_assessments(st)


def primary_result_display(category_prediction, recommendation, category_error=None) -> dict[str, object]:
    """Prepare the main result without promoting uncertain candidates to labels."""
    category = getattr(category_prediction, "category", None)
    candidate = getattr(category_prediction, "candidate_category", None)
    category_name = getattr(category, "value", category)
    candidate_name = getattr(candidate, "value", candidate)
    uncertain = (
        bool(category_error)
        or category_prediction is None
        or bool(getattr(category_prediction, "is_uncertain", True))
        or category_name is None
    )
    score = getattr(category_prediction, "score", None)
    return {
        "category": "Uncertain" if uncertain else f"{category_name} · PROVISIONAL",
        "candidate": candidate_name if uncertain else None,
        "score": score,
        "score_label": f"{score:.1%} · raw, uncalibrated" if isinstance(score, (int, float)) else "Unavailable",
        "status": "Classifier unavailable" if category_error else "Below provisional 70% cutoff" if uncertain else "Folder-derived experimental model",
        "manual_review_required": uncertain or bool((recommendation or {}).get("manual_review_required", True)),
        "bin": (recommendation or {}).get("bin"),
        "explanation": (recommendation or {}).get("explanation", "A reliable recommendation is unavailable."),
    }


def classify_category_for_family(family_prediction, image, classifier):
    """Run stage two only after a certain family result, within supported classes."""
    from vista.classification.taxonomy import WasteFamily

    family = family_prediction.get("family") if isinstance(family_prediction, dict) else None
    family_name = getattr(family, "value", family)
    if (
        family is None
        or family_name not in {WasteFamily.MSW.value, WasteFamily.E_WASTE.value}
        or bool(family_prediction.get("is_uncertain"))
    ):
        return None, "Family result is uncertain or unavailable; category stage was not run."
    if classifier is None:
        return None, "The provisional category checkpoint is unavailable."

    prediction = classifier.predict(image)
    candidate = getattr(prediction, "candidate_category", None)
    candidate_name = getattr(candidate, "value", candidate)
    mapped_family = getattr(prediction, "family_mapping", None)
    mapped_family_name = getattr(mapped_family, "value", mapped_family)
    if family_name == WasteFamily.MSW.value and mapped_family_name != WasteFamily.MSW.value:
        return None, (
            f"The category model's {candidate_name or 'top'} candidate is outside the MSW branch; "
            "manual review is required."
        )
    if family_name == WasteFamily.E_WASTE.value and candidate_name != "battery":
        return None, (
            "The current E-waste branch supports only the provisional battery category. "
            "Other electronic categories require a suitable trained model; manual review is required."
        )
    return prediction, None


def render_generic_detection_diagnostics(st, analysis) -> None:
    """Render optional generic YOLO details independently from sorting results."""
    with st.expander("Technical details · generic object detections", expanded=False):
        st.caption(GENERAL_DETECTION_NOTICE)
        st.warning(GENERIC_DETECTION_WARNING)
        if analysis["detections"]:
            show_person_boxes = st.checkbox(
                "Show generic person boxes in the diagnostic image",
                value=False,
                key="vista_show_generic_person_boxes",
            )
            st.image(
                annotate_detections(
                    analysis["frame"], analysis["detections"],
                    show_person_boxes=show_person_boxes,
                ),
                channels="BGR",
                caption="Orange boxes are generic YOLO proposals, not waste items.",
                width="stretch",
            )
            for row in detection_table_rows(analysis["detections"]):
                st.caption(f"{row['Generic label']} · {row['Confidence']} · generic class")
        else:
            st.caption(
                "No generic objects were detected. This does not determine whether an image contains waste."
            )


def render_optional_generic_diagnostics(st, analysis) -> None:
    """Keep failures in nonessential YOLO UI from breaking the sorting result."""
    try:
        render_generic_detection_diagnostics(st, analysis)
    except Exception as error:
        st.caption(f"Optional generic-object diagnostics unavailable: {error}")


def _render_detection(
    st, load_detector, load_classifier, load_family_classifier,
    load_provisional_family_classifier=None, load_provisional_category_classifier=None,
) -> None:
    from vista.detection.image_workflow import classify_detections

    st.markdown("#### 1 · Add an image")
    st.caption("Use a clear image with one item in view. Capture a still frame or upload a photo.")
    current_bytes = None
    camera_error = None
    with st.container(border=True):
        source = st.radio(
            "Image source",
            ("Upload image", "Camera capture"),
            horizontal=True,
            key="vista_image_source",
        )
        if source == "Upload image":
            selected_image = st.file_uploader(
                "Upload a photo",
                type=("jpg", "jpeg", "png", "bmp", "webp"),
                key="vista_upload",
                label_visibility="collapsed",
            )
        else:
            try:
                selected_image = st.camera_input("Capture one item", key="vista_camera")
            except Exception as error:
                selected_image = None
                camera_error = str(error)
                st.error("Camera unavailable. Allow browser camera access or switch to image upload.")
                st.caption(camera_error)

        if selected_image is not None:
            try:
                current_bytes = image_bytes_from_capture(selected_image)
                preview_frame = decode_image_bytes(current_bytes)
            except ValueError as error:
                st.error(str(error))
                current_bytes = None
            else:
                st.image(preview_frame, channels="BGR", width="stretch")
        elif source == "Upload image":
            st.info("Choose a JPG, PNG, BMP, or WebP image to continue.")
        elif camera_error is None:
            st.info("Allow camera access, capture one item, then analyze the captured still image.")

        with st.expander("Optional experimental diagnostics", expanded=False):
            st.caption(
                "These optional experiments do not replace the primary whole-image category "
                "or change the disposal recommendation."
            )
            classify_whole_image = st.checkbox(
                "Run paper/cardboard pilot on the whole image",
                value=False,
                key="vista_classify_whole_image",
            )
            classify_yolo_regions = st.checkbox(
                "Run category and paper/cardboard pilots on detected crops",
                value=False,
                key="vista_classify_yolo_regions",
            )

        analyze_clicked = st.button(
            "Analyze image",
            type="primary",
            disabled=current_bytes is None,
            width="stretch",
        )

    current_digest = image_digest(current_bytes) if current_bytes is not None else None
    if analyze_clicked and current_bytes is not None:
        try:
            with st.spinner("Analyzing image…"):
                frame, detections = analyze_image(current_bytes, load_detector())
            source_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            family_prediction = None
            family_error = None
            provisional_family_prediction = None
            provisional_family_error = None
            provisional_category_prediction = None
            provisional_category_error = None
            category_classifier = None
            category_crop_results = []
            category_crop_error = None
            classified_objects = []
            classifier_error = None
            whole_image_prediction = None
            whole_image_error = None
            gradcam_result = None
            gradcam_error = None

            try:
                family_prediction = load_family_classifier().predict(source_image)
            except Exception as error:
                family_error = str(error)

            if load_provisional_family_classifier is not None:
                provisional_classifier, provisional_family_error = load_provisional_family_classifier()
                if provisional_classifier is not None:
                    try:
                        provisional_family_prediction = provisional_classifier.predict(source_image)
                    except Exception as error:
                        provisional_family_error = str(error)

            if load_provisional_category_classifier is not None:
                category_classifier, category_load_error = load_provisional_category_classifier()
                if category_load_error:
                    provisional_category_error = category_load_error
                elif category_classifier is not None:
                    try:
                        provisional_category_prediction, provisional_category_error = (
                            classify_category_for_family(
                                provisional_family_prediction,
                                source_image,
                                category_classifier,
                            )
                        )
                    except Exception as error:
                        provisional_category_error = str(error)
                else:
                    provisional_category_error = "The provisional category classifier is unavailable."

            if category_classifier is not None and provisional_category_prediction is not None:
                try:
                    from vista.explainability.gradcam import generate_gradcam

                    gradcam_result = generate_gradcam(category_classifier, source_image)
                except Exception as error:
                    gradcam_error = str(error)
            elif provisional_category_error:
                gradcam_error = provisional_category_error

            if classify_yolo_regions:
                if category_classifier is None:
                    category_crop_error = provisional_category_error or "Provisional category model is unavailable."
                else:
                    try:
                        category_crop_results = classify_yolo_proposed_regions(
                            source_image, detections, category_classifier
                        )
                    except Exception as error:
                        category_crop_error = str(error)
                non_person_detections = [
                    item for item in detections
                    if item.class_label.strip().casefold() != "person"
                ]
                if non_person_detections:
                    try:
                        classified_objects = classify_detections(
                            source_image, non_person_detections, load_classifier()
                        )
                    except Exception as error:
                        classifier_error = str(error)

            if classify_whole_image:
                try:
                    whole_image_prediction = classify_full_image(frame, load_classifier())
                except Exception as error:
                    whole_image_error = str(error)

            try:
                bin_mapping = load_disposal_bin_mapping()
                mapping_error = None
            except ValueError as error:
                bin_mapping = None
                mapping_error = str(error)

            recommendation = disposal_recommendation(
                provisional_family_prediction,
                provisional_category_prediction,
                bin_mapping,
                family_error=provisional_family_error,
                category_error=provisional_category_error,
                mapping_error=mapping_error,
            )
            st.session_state["vista_analysis"] = {
                "digest": current_digest,
                "whole_image_requested": classify_whole_image,
                "frame": frame,
                "detections": detections,
                "classified_objects": classified_objects,
                "classifier_error": classifier_error,
                "whole_image_prediction": whole_image_prediction,
                "whole_image_error": whole_image_error,
                "family_prediction": family_prediction,
                "family_error": family_error,
                "provisional_family_prediction": provisional_family_prediction,
                "provisional_family_error": provisional_family_error,
                "provisional_category_prediction": provisional_category_prediction,
                "provisional_category_error": provisional_category_error,
                "category_recommendation_prediction": provisional_category_prediction,
                "category_recommendation_error": provisional_category_error,
                "category_recommendation_source": "whole-image",
                "category_crop_requested": classify_yolo_regions,
                "category_crop_results": category_crop_results,
                "category_crop_error": category_crop_error,
                "gradcam_result": gradcam_result,
                "gradcam_error": gradcam_error,
                "disposal_recommendation": recommendation,
                "disposal_mapping_error": mapping_error,
                "error": None,
            }
            st.session_state["vista_session_results"] = update_session_results(
                st.session_state.get("vista_session_results", {}),
                current_digest,
                recommendation,
            )
        except Exception as error:
            st.session_state["vista_analysis"] = {
                "digest": current_digest,
                "whole_image_requested": classify_whole_image,
                "frame": None,
                "detections": [],
                "classified_objects": [],
                "classifier_error": None,
                "whole_image_prediction": None,
                "whole_image_error": None,
                "family_prediction": None,
                "family_error": None,
                "provisional_family_prediction": None,
                "provisional_family_error": None,
                "provisional_category_prediction": None,
                "provisional_category_error": None,
                "category_recommendation_prediction": None,
                "category_recommendation_error": None,
                "category_recommendation_source": "unavailable",
                "category_crop_requested": classify_yolo_regions,
                "category_crop_results": [],
                "category_crop_error": None,
                "gradcam_result": None,
                "gradcam_error": None,
                "disposal_recommendation": {
                    "category": None,
                    "bin": None,
                    "confidence": None,
                    "manual_review_required": True,
                    "message": MANUAL_SORTING_MESSAGE,
                    "explanation": "Image analysis did not complete.",
                },
                "disposal_mapping_error": None,
                "error": str(error),
            }

    analysis = st.session_state.get("vista_analysis")
    is_current_result = (
        analysis is not None
        and analysis.get("digest") == current_digest
        and analysis.get("whole_image_requested", False) == classify_whole_image
        and analysis.get("category_crop_requested", False) == classify_yolo_regions
    )
    st.markdown("#### 2 · Sorting result")
    if current_bytes is None:
        st.info("Your image preview and sorting result will appear here.")
    elif not is_current_result:
        st.info("Image ready. Select **Analyze image** to see its provisional result.")
    elif analysis.get("error"):
        st.error(f"Image analysis could not be completed: {analysis['error']}")
        st.warning(MANUAL_SORTING_MESSAGE)
    else:
        image_column, result_column = st.columns((1.05, 0.95), gap="large")
        with image_column:
            with st.container(border=True):
                st.image(analysis["frame"], channels="BGR", caption="Analyzed image", width="stretch")
        with result_column:
            with st.container(border=True):
                st.markdown('<div class="vista-result-label">PROVISIONAL CATEGORY</div>', unsafe_allow_html=True)
                display = primary_result_display(
                    analysis.get("provisional_category_prediction"),
                    analysis.get("disposal_recommendation"),
                    analysis.get("provisional_category_error"),
                )
                st.markdown(
                    f'<div class="vista-result-name">{display["category"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(display["score_label"])
                if display["candidate"]:
                    st.caption(f"Top candidate only — not confirmed: {display['candidate']}")
                st.caption(display["status"])

                family_row = provisional_family_prediction_row(
                    analysis.get("provisional_family_prediction"),
                    error=analysis.get("provisional_family_error"),
                )
                st.markdown("**Waste family**")
                st.write(f"{family_row['PROVISIONAL family']} · PROVISIONAL")
                st.caption("Family and category outputs are experimental and are not verified ground truth.")
                if analysis.get("provisional_category_error"):
                    st.caption(f"Category stage: {analysis['provisional_category_error']}")

                if display["manual_review_required"]:
                    st.warning(MANUAL_SORTING_MESSAGE)
                elif display["bin"]:
                    st.success(f"Recommended disposal: {display['bin']}")
                else:
                    st.warning(MANUAL_SORTING_MESSAGE)
                st.caption(display["explanation"])
                st.caption("Bin guidance is configurable and may differ from local collection rules.")

                category_prediction = analysis.get("provisional_category_prediction")
                if category_prediction is not None and not category_prediction.is_uncertain and category_prediction.category is not None:
                    guidance = category_guidance(category_prediction.category.value)
                    with st.expander("General handling guidance"):
                        st.write(guidance["handling"])
                        st.caption(guidance["notice"])

        render_optional_generic_diagnostics(st, analysis)

        with st.expander("Why this prediction?", expanded=False):
            gradcam = analysis.get("gradcam_result")
            if gradcam is None:
                st.info(
                    "A Grad-CAM explanation is unavailable for this result. "
                    + str(analysis.get("gradcam_error") or "No supported category result was produced.")
                )
            else:
                original_column, heatmap_column = st.columns(2)
                with original_column:
                    st.image(
                        analysis["frame"], channels="BGR",
                        caption="Original image", width="stretch",
                    )
                with heatmap_column:
                    st.image(gradcam.overlay, caption="Grad-CAM overlay", width="stretch")
                st.caption(
                    f"Grad-CAM highlights image regions that influenced the model's "
                    f"{gradcam.target_class} output. It does not prove that the classification is correct."
                )

        if classify_whole_image or classify_yolo_regions:
            with st.expander("Experimental pilot outputs", expanded=False):
                st.caption(PILOT_CLASSIFIER_NOTICE)
                if classify_whole_image:
                    if analysis.get("whole_image_error"):
                        st.warning(f"Whole-image pilot unavailable: {analysis['whole_image_error']}")
                    elif analysis.get("whole_image_prediction") is not None:
                        row = whole_image_prediction_row(analysis["whole_image_prediction"])
                        st.write(f"**Whole-image pilot:** {row['VISTA pilot category']} · {row['Raw confidence']}")
                if classify_yolo_regions:
                    st.markdown("**YOLO-proposed crop diagnostics**")
                    _render_category_crop_results(st, analysis, requested=True)
                    if analysis.get("classifier_error"):
                        st.warning(f"Paper/cardboard crop pilot unavailable: {analysis['classifier_error']}")
                    for row in classification_table_rows(analysis.get("classified_objects", [])):
                        st.caption(
                            f"Paper/cardboard crop {row['Object crop']}: "
                            f"{row['VISTA pilot category']} · {row['Raw confidence']}"
                        )



def _render_status_cards(st, cards: tuple[tuple[str, str], ...]) -> None:
    for start in range(0, len(cards), 2):
        columns = st.columns(2)
        for column, (label, value) in zip(columns, cards[start : start + 2]):
            with column:
                st.metric(label, value)


def _render_disposal_recommendation(st, recommendation, mapping_error=None) -> None:
    st.markdown("##### Disposal recommendation")
    recommendation = recommendation or {
        "category": None,
        "bin": None,
        "confidence": None,
        "manual_review_required": True,
        "message": MANUAL_SORTING_MESSAGE,
        "explanation": "A reliable recommendation is unavailable.",
    }
    if recommendation.get("manual_review_required"):
        st.warning(MANUAL_SORTING_MESSAGE)
    else:
        st.success(recommendation["message"])
    category = recommendation.get("category")
    if category:
        st.write(f"**Provisional category:** {category}")
    confidence = recommendation.get("confidence")
    st.write(
        "**Raw category score:** "
        + (f"{confidence:.1%} · uncalibrated" if isinstance(confidence, (int, float)) else "Unavailable")
    )
    st.caption(recommendation.get("explanation", ""))
    st.caption(
        "Bin names are configurable and are general guidance; follow your local waste-collection rules."
    )
    if mapping_error:
        st.error(f"Disposal-bin configuration error: {mapping_error}")


def _render_weight_estimates(st, analysis, image_digest_value: str) -> dict[str, object]:
    """Collect an optional measured weight and render only supported estimates."""
    from vista.impact.material_factors import estimate_material_impacts

    st.markdown("##### Optional weight-based analysis")
    st.caption(
        "VISTA does not measure weight from the camera. Enter a value only if you have "
        "weighed this item; all resulting figures are provisional estimates."
    )
    has_measured_weight = st.checkbox(
        "I have a measured weight for this item",
        key=f"vista_weight_enabled_{image_digest_value[:16]}",
    )
    weight_grams = None
    if has_measured_weight:
        entered_weight = st.number_input(
            "Measured item weight (grams)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.1f",
            key=f"vista_weight_grams_{image_digest_value[:16]}",
        )
        if entered_weight > 0:
            weight_grams = float(entered_weight)
        else:
            st.caption("Enter a value greater than zero to use the measured weight.")

    prediction = analysis.get("provisional_category_prediction")
    category = getattr(prediction, "category", None)
    category = getattr(category, "value", category)
    category_is_certain = (
        prediction is not None
        and not bool(getattr(prediction, "is_uncertain", True))
        and category is not None
    )
    estimates = estimate_material_impacts(
        category,
        weight_grams,
        category_is_certain=category_is_certain,
    )

    carbon = estimates["carbon"]
    material_value = estimates["material_value"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Potential recycling CO2e impact · PROVISIONAL**")
        if carbon["value"] is None:
            st.info(carbon["status"])
        else:
            st.metric(carbon["unit"], f"{carbon['value']:.4f}")
            st.caption(f"{carbon['status']} Formula: {carbon['formula']}. Source: {carbon['source']}")
    with right:
        st.markdown("**Estimated material value · PROVISIONAL**")
        if material_value["value"] is None:
            st.info(material_value["status"])
        else:
            st.metric(material_value["unit"], f"{material_value['value']:.2f}")
            st.caption(
                f"{material_value['status']} Formula: {material_value['formula']}. "
                f"Source: {material_value['source']}"
            )
    return estimates


def _render_unavailable_assessments(st) -> None:
    """Show unavailable visual assessments instead of implying they were measured."""
    st.markdown("##### Visual condition and contamination")
    st.caption("These properties are not inferred from the current image-analysis pipeline.")
    columns = st.columns(2)
    for column, (name, status) in zip(columns, ANALYTICS_UNAVAILABLE.items()):
        with column:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.info(status)


def _render_category_crop_results(st, analysis, *, requested: bool) -> None:
    st.markdown("##### Optional category predictions on YOLO-proposed crops")
    st.caption(
        "The generic YOLO class is used only to identify a crop region. The separate six-way "
        "category model predicts the crop label; its output remains provisional."
    )
    if not requested:
        st.info("Crop-level category analysis was not selected for this run.")
        return
    if analysis.get("category_crop_error"):
        st.warning(f"Crop-level category analysis unavailable: {analysis['category_crop_error']}")
        return
    results = analysis.get("category_crop_results", [])
    if not results:
        st.info("No non-person YOLO regions were available to crop; no crop-level category prediction was made.")
        return

    for index, result in enumerate(results, start=1):
        row = category_crop_prediction_row(result)
        detection = result["detection"]
        with st.expander(f"Region {index} · generic YOLO proposal: {detection.class_label}"):
            st.caption(
                f"Generic YOLO proposal score: {detection.confidence:.1%}. "
                "This score and label are not waste-category evidence."
            )
            crop = result.get("crop")
            if crop is not None:
                st.image(crop, caption="Cropped pixels sent to the provisional category model", width="stretch")
            st.write(f"**VISTA provisional crop category:** {row['category']}")
            st.write(f"**Raw category score:** {row['raw_score']}")
            st.caption(row["status"])
    st.caption(
        "The whole-image category is the input to this recommendation. YOLO-crop category results are "
        "experimental diagnostics only and do not override it. Generic YOLO labels remain proposals, not waste labels."
    )


def _render_summary_cards(st, summary: dict[str, str]) -> None:
    columns = st.columns(2)
    for column, (label, value) in zip(columns, summary.items()):
        with column:
            st.metric(label, value)


def _render_compact_result_card(st, title: str, label: str, score: str, status: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        left, right = st.columns((2, 1))
        with left:
            st.write(label)
        with right:
            st.metric("Raw score", score)
        st.caption(status)


def classification_table_rows(classified_objects) -> list[dict[str, str]]:
    """Format pilot crop predictions separately from generic YOLO detections."""
    rows = []
    for index, item in enumerate(classified_objects, start=1):
        if item.error:
            category = "Unavailable — review crop"
            status = item.error
            confidence = "—"
        elif item.prediction is None:
            category = "Uncertain — manual review recommended"
            status = "No pilot prediction"
            confidence = "—"
        elif item.prediction.is_uncertain:
            category = "Uncertain — manual review recommended"
            status = "Below provisional threshold"
            confidence = f"{item.prediction.confidence:.1%}"
        else:
            category = item.prediction.label or "Uncertain — manual review recommended"
            status = "Experimental pilot prediction"
            confidence = f"{item.prediction.confidence:.1%}"
        rows.append(
            {
                "Object crop": str(index),
                "VISTA pilot category": category,
                "Raw confidence": confidence,
                "Status": status,
            }
        )
    return rows


def whole_image_prediction_row(prediction) -> dict[str, str]:
    """Format a full-frame pilot output without conflating it with a YOLO crop."""
    if prediction.is_uncertain:
        label = "Uncertain — manual review recommended"
        status = "Below provisional threshold"
    else:
        label = prediction.label or "Uncertain — manual review recommended"
        status = "Experimental whole-image pilot prediction"
    return {
        "Image scope": "Whole image",
        "VISTA pilot category": label,
        "Raw confidence": f"{prediction.confidence:.1%}",
        "Status": status,
    }


def family_prediction_table_row(prediction=None, *, error: str | None = None) -> dict[str, str]:
    """Format only explicit family-model output; never infer from detections."""
    unavailable_label = "Uncertain — verified MSW/E-waste model not available"
    if error:
        return {
            "Waste family": unavailable_label,
            "Raw model score": "Unavailable",
            "Status": f"Family model unavailable: {error}",
        }
    if prediction is None or prediction.is_uncertain or prediction.family is None:
        return {
            "Waste family": unavailable_label,
            "Raw model score": (
                f"{prediction.score:.1%}" if prediction and prediction.score is not None
                else "Unavailable"
            ),
            "Status": prediction.status if prediction is not None else FAMILY_STATUS,
        }
    return {
        "Waste family": prediction.family.value,
        "Raw model score": f"{prediction.score:.1%}",
        "Status": f"Experimental — {prediction.status}",
    }


def provisional_family_prediction_row(prediction: dict | None = None, *, error: str | None = None) -> dict[str, str]:
    """Format a separate folder-derived output without implying verified status."""
    if error:
        return {
            "PROVISIONAL family": "Unavailable",
            "Raw score": "Unavailable",
            "Status": f"Provisional model unavailable: {error}",
        }
    if prediction is None:
        return {
            "PROVISIONAL family": "Uncertain",
            "Raw score": "Unavailable",
            "Status": "No provisional prediction is available.",
        }
    family = prediction.get("family")
    score = prediction.get("score")
    uncertain = bool(prediction.get("is_uncertain")) or family is None
    return {
        "PROVISIONAL family": "Uncertain — manual review recommended" if uncertain else f"{family.value if hasattr(family, 'value') else family} · PROVISIONAL",
        "Raw score": f"{score:.1%}" if isinstance(score, (int, float)) else "Unavailable",
        "Status": str(prediction.get("status") or "Folder-derived provisional label; not verified."),
    }


def provisional_category_prediction_row(prediction) -> dict[str, str]:
    """Format an image-level category prediction without implying object counts."""
    candidate = getattr(prediction, "candidate_category", None)
    category = getattr(prediction, "category", None)
    score = getattr(prediction, "score", None)
    uncertain = bool(getattr(prediction, "is_uncertain", True)) or category is None
    candidate_name = getattr(candidate, "value", candidate) or "Unavailable"
    category_name = getattr(category, "value", category)
    return {
        "PROVISIONAL image category": (
            f"Uncertain — top candidate {candidate_name} · manual review"
            if uncertain else f"{category_name} · PROVISIONAL"
        ),
        "Raw score": f"{score:.1%}" if isinstance(score, (int, float)) else "Unavailable",
        "Status": str(getattr(prediction, "status", "Folder-derived output; not verified.")),
    }


def provisional_category_diagnostic_rows(prediction) -> list[dict[str, str]]:
    """Format the classifier's raw top-three candidates for experimental review."""
    candidates = getattr(prediction, "top_candidates", ())
    rows = []
    for rank, candidate in enumerate(candidates[:3], start=1):
        category, score = candidate
        name = getattr(category, "value", category)
        rows.append({"Rank": str(rank), "Candidate": str(name), "Raw score": f"{score:.1%}"})
    return rows


def provisional_model_consistency(family_prediction, category_prediction) -> str:
    """Compare independent provisional outputs only when both pass thresholds."""
    if not isinstance(family_prediction, dict) or category_prediction is None:
        return "inconclusive"
    family = family_prediction.get("family")
    if family is None or family_prediction.get("is_uncertain") or getattr(category_prediction, "is_uncertain", True):
        return "inconclusive"
    family_name = getattr(family, "value", family)
    category_family = getattr(category_prediction, "family_mapping", None)
    category_family_name = getattr(category_family, "value", category_family)
    if category_family_name is None:
        return "inconclusive"
    return "aligned" if family_name == category_family_name else "conflict"


def generic_detection_breakdown(detections) -> list[dict[str, object]]:
    """Count generic YOLO labels only when multiple boxes were detected."""
    if len(detections) < 2:
        return []
    counts = {}
    for detection in detections:
        counts[detection.class_label] = counts.get(detection.class_label, 0) + 1
    return [
        {"Generic YOLO label": label, "Detected boxes": count}
        for label, count in sorted(counts.items())
    ]


def result_summary(family_prediction, category_prediction, detections, *, family_error=None, category_error=None):
    """Summarize the separate model outputs and keep uncertainty visible."""
    family = family_prediction.get("family") if isinstance(family_prediction, dict) else None
    family_name = getattr(family, "value", family)
    family_uncertain = (
        bool(family_error)
        or not isinstance(family_prediction, dict)
        or bool(family_prediction.get("is_uncertain"))
        or family is None
    )
    family_text = "Uncertain" if family_uncertain else f"{family_name} · PROVISIONAL"

    category = getattr(category_prediction, "category", None)
    candidate = getattr(category_prediction, "candidate_category", None)
    category_uncertain = (
        bool(category_error)
        or category_prediction is None
        or bool(getattr(category_prediction, "is_uncertain", True))
        or category is None
    )
    category_name = getattr(category, "value", category)
    candidate_name = getattr(candidate, "value", candidate)
    if category_uncertain:
        category_text = f"Uncertain — top candidate {candidate_name}" if candidate_name else "Uncertain"
    else:
        category_text = f"{category_name} · PROVISIONAL"

    detection_count = len(detections or ())
    detection_text = (
        "No generic YOLO boxes"
        if detection_count == 0
        else f"{detection_count} generic YOLO box(es) — not waste items"
    )
    uncertain = family_uncertain or category_uncertain
    uncertainty_text = (
        "Yes — manual review recommended"
        if uncertain
        else "No score below cutoff; outputs remain provisional"
    )
    return {
        "Family prediction": family_text,
        "Category prediction": category_text,
        "Generic YOLO detections": detection_text,
        "Uncertain": uncertainty_text,
    }


def category_guidance(category: str) -> dict[str, str]:
    """Return general handling information without claiming recyclability."""
    try:
        meaning, handling = CATEGORY_GUIDANCE[category]
    except KeyError as error:
        raise ValueError(f"No supported guidance is available for category {category!r}.") from error
    return {
        "meaning": meaning,
        "handling": handling,
        "notice": CATEGORY_GUIDANCE_NOTICE,
    }


def _render_unavailable_page(st, title: str, status: str, message: str) -> None:
    st.subheader(title)
    st.warning(status)
    st.info(message)


def _render_about(st) -> None:
    st.markdown(f"**{APP_SUBTITLE}**")
    st.write(PROJECT_DESCRIPTION)
    st.markdown(f"**Object detector:** {GENERAL_DETECTION_NOTICE}")
    st.markdown(
        "**Classification status:** Family and six-category outputs are provisional, "
        "trained with source-folder-derived labels, and not verified ground truth."
    )
    st.markdown(f"**Verified family model:** {FAMILY_RESULT} — {FAMILY_STATUS}.")
    st.caption(
        "The 0.70 cutoff is provisional and uncalibrated. Recommendations require certain, "
        "agreeing provisional family/category results. Follow local disposal rules. "
        "Condition, contamination, Grad-CAM, sourced carbon estimates, and economic estimates are unavailable."
    )


if __name__ == "__main__":
    main()
