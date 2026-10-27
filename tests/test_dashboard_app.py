from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dashboard_app(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)
    from dashboard import app

    return app


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.frames = []

    def detect_frame(self, frame):
        self.frames.append(frame.copy())
        return self.detections


class FakeClassifier:
    def __init__(self, prediction=None):
        self.prediction = prediction or SimpleNamespace(
            label="paper", confidence=0.82, threshold=0.70, is_uncertain=False
        )
        self.images = []

    def predict(self, image):
        self.images.append(image.copy())
        return self.prediction


def _png_bytes(color=(12, 80, 200)):
    image = Image.new("RGB", (5, 4), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_bytes_returns_three_channel_bgr_frame(dashboard_app):
    frame = dashboard_app.decode_image_bytes(_png_bytes((10, 40, 220)))

    assert frame.shape == (4, 5, 3)
    assert frame.dtype == np.uint8
    assert tuple(frame[0, 0]) == (220, 40, 10)


@pytest.mark.parametrize("image_bytes", [b"", b"not an image"])
def test_decode_image_bytes_rejects_empty_or_invalid_data(dashboard_app, image_bytes):
    with pytest.raises(ValueError, match="image"):
        dashboard_app.decode_image_bytes(image_bytes)


def test_image_bytes_from_capture_handles_camera_results_and_read_errors(dashboard_app):
    class Capture:
        def getvalue(self):
            return b"captured-frame"

    class BrokenCapture:
        def getvalue(self):
            raise OSError("camera stream unavailable")

    assert dashboard_app.image_bytes_from_capture(None) is None
    assert dashboard_app.image_bytes_from_capture(Capture()) == b"captured-frame"
    with pytest.raises(ValueError, match="Could not read the captured image"):
        dashboard_app.image_bytes_from_capture(BrokenCapture())


def test_disposal_bin_mapping_has_safe_defaults_and_validated_overrides(dashboard_app):
    defaults = dashboard_app.load_disposal_bin_mapping("")
    assert set(defaults) == {"paper", "cardboard", "plastic", "glass", "organic", "battery"}
    assert defaults["battery"] == "Battery / e-waste collection bin"

    overridden = dashboard_app.load_disposal_bin_mapping('{"paper":"Blue paper stream"}')
    assert overridden["paper"] == "Blue paper stream"
    assert overridden["glass"] == defaults["glass"]
    with pytest.raises(ValueError, match="JSON object"):
        dashboard_app.load_disposal_bin_mapping("not json")
    with pytest.raises(ValueError, match="Unsupported disposal-bin categories"):
        dashboard_app.load_disposal_bin_mapping('{"metal":"Metal bin"}')
    with pytest.raises(ValueError, match="non-empty string"):
        dashboard_app.load_disposal_bin_mapping('{"paper":"  "}')


def test_disposal_recommendation_requires_certain_agreeing_provisional_models(dashboard_app):
    from vista.classification.taxonomy import WasteCategory, WasteFamily

    family = {"family": WasteFamily.MSW, "score": 0.91, "is_uncertain": False}
    category = SimpleNamespace(
        category=WasteCategory.PAPER,
        candidate_category=WasteCategory.PAPER,
        family_mapping=WasteFamily.MSW,
        score=0.84,
        is_uncertain=False,
    )
    mapping = dashboard_app.load_disposal_bin_mapping("")
    result = dashboard_app.disposal_recommendation(family, category, mapping)
    assert result["bin"] == mapping["paper"]
    assert result["category"] == "paper"
    assert result["confidence"] == 0.84
    assert result["manual_review_required"] is False

    uncertain = SimpleNamespace(
        category=None,
        candidate_category=WasteCategory.PAPER,
        family_mapping=WasteFamily.MSW,
        score=0.69,
        is_uncertain=True,
    )
    uncertain_result = dashboard_app.disposal_recommendation(family, uncertain, mapping)
    assert uncertain_result["bin"] is None
    assert uncertain_result["message"] == dashboard_app.MANUAL_SORTING_MESSAGE
    assert uncertain_result["manual_review_required"] is True

    conflict = dashboard_app.disposal_recommendation(
        {"family": WasteFamily.E_WASTE, "score": 0.9, "is_uncertain": False},
        category,
        mapping,
    )
    assert conflict["bin"] is None
    assert conflict["manual_review_required"] is True


def test_disposal_recommendation_handles_missing_and_unsupported_categories(dashboard_app):
    from vista.classification.taxonomy import WasteFamily

    mapping = dashboard_app.load_disposal_bin_mapping("")
    unavailable = dashboard_app.disposal_recommendation(
        None, None, mapping, category_error="checkpoint missing"
    )
    assert unavailable["bin"] is None
    assert unavailable["message"] == dashboard_app.MANUAL_SORTING_MESSAGE

    unsupported_category = SimpleNamespace(
        category="textile",
        candidate_category="textile",
        family_mapping=WasteFamily.MSW,
        score=0.99,
        is_uncertain=False,
    )
    unsupported = dashboard_app.disposal_recommendation(
        {"family": WasteFamily.MSW, "is_uncertain": False},
        unsupported_category,
        mapping,
    )
    assert unsupported["bin"] is None
    assert unsupported["manual_review_required"] is True


def test_session_analytics_count_completed_unique_images_and_manual_reviews(dashboard_app):
    recommended = {
        "category": "paper",
        "category_is_certain": True,
        "bin": "Paper recycling bin",
        "manual_review_required": False,
    }
    manual = {
        "category": "glass",
        "bin": None,
        "manual_review_required": True,
    }
    results = dashboard_app.update_session_results({}, "image-a", recommended)
    results = dashboard_app.update_session_results(results, "image-a", recommended)
    results = dashboard_app.update_session_results(results, "image-b", manual)

    assert dashboard_app.session_summary(results) == {
        "analyzed_items": 2,
        "recommended_items": 1,
        "manual_review_items": 1,
        "by_category": {"paper": 1},
        "measured_weight_items": 0,
        "potential_co2e_estimate_count": 0,
        "potential_co2e_kg_total": None,
        "material_value_estimate_count": 0,
        "material_value_totals_by_currency": {},
    }


def test_condition_contamination_and_impact_analytics_are_explicitly_unavailable(dashboard_app):
    assert set(dashboard_app.ANALYTICS_UNAVAILABLE) == {"Condition", "Contamination"}
    assert all(value.startswith("Unavailable") for value in dashboard_app.ANALYTICS_UNAVAILABLE.values())


def test_live_demo_workflow_and_visual_assessment_messages_are_explicit(dashboard_app):
    steps = " ".join(dashboard_app.LIVE_WORKFLOW_STEPS)
    assert "Capture one item" in steps
    assert "Analyze image" in steps
    assert "sort manually" in steps
    assert "no validated condition assessment" in dashboard_app.ANALYTICS_UNAVAILABLE["Condition"]
    assert "no contamination detection logic" in dashboard_app.ANALYTICS_UNAVAILABLE["Contamination"]
    assert "never counted as waste items" in dashboard_app.GENERIC_DETECTION_WARNING


def test_live_unavailable_assessment_panel_displays_both_limitations(dashboard_app):
    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def markdown(self, value):
            rendered.append(value)

        def info(self, value):
            rendered.append(value)

    class FakeStreamlit:
        def columns(self, count):
            return [Container() for _ in range(count)]

        def container(self, **kwargs):
            return Container()

        def markdown(self, value):
            rendered.append(value)

        def caption(self, value):
            rendered.append(value)

        def info(self, value):
            rendered.append(value)

    rendered = []
    dashboard_app._render_unavailable_assessments(FakeStreamlit())
    text = " ".join(rendered)
    assert "Visual condition and contamination" in text
    assert "Condition" in text and "Contamination" in text
    assert "no validated condition assessment" in text
    assert "no contamination detection logic" in text


def test_session_analytics_aggregate_only_available_weight_estimates(dashboard_app):
    first_estimates = {
        "weight_grams": 250.0,
        "carbon": {"value": 0.3},
        "material_value": {"value": 0.2, "unit": "USD"},
    }
    second_estimates = {
        "weight_grams": 500.0,
        "carbon": {"value": None},
        "material_value": {"value": None, "unit": "currency not configured"},
    }
    results = dashboard_app.update_session_results(
        {}, "image-a",
        {"category": "paper", "category_is_certain": True, "bin": "Paper bin", "manual_review_required": False},
        first_estimates,
    )
    results = dashboard_app.update_session_results(
        results, "image-b",
        {"category": "glass", "category_is_certain": False, "bin": None, "manual_review_required": True},
        second_estimates,
    )

    summary = dashboard_app.session_summary(results)
    assert summary["analyzed_items"] == 2
    assert summary["by_category"] == {"paper": 1}
    assert summary["manual_review_items"] == 1
    assert summary["measured_weight_items"] == 2
    assert summary["potential_co2e_estimate_count"] == 1
    assert summary["potential_co2e_kg_total"] == pytest.approx(0.3)
    assert summary["material_value_estimate_count"] == 1
    assert summary["material_value_totals_by_currency"] == {"USD": pytest.approx(0.2)}


def test_session_impact_update_preserves_other_images_and_replaces_same_image(dashboard_app):
    initial = {
        "image-a": {"category": "paper", "impact_estimates": None},
        "image-b": {"category": None, "impact_estimates": None},
    }
    updated = dashboard_app.update_session_impact_estimates(
        initial, "image-a", {"weight_grams": 100.0, "carbon": {"value": None}}
    )

    assert updated["image-a"]["impact_estimates"]["weight_grams"] == 100.0
    assert updated["image-b"] == initial["image-b"]
    assert initial["image-a"]["impact_estimates"] is None


def test_analyze_image_passes_decoded_bgr_frame_to_fake_detector(dashboard_app):
    detector = FakeDetector(detections=[])

    frame, detections = dashboard_app.analyze_image(_png_bytes((10, 40, 220)), detector)

    assert detections == []
    assert len(detector.frames) == 1
    assert np.array_equal(detector.frames[0], frame)
    assert tuple(detector.frames[0][0, 0]) == (220, 40, 10)


def test_whole_image_classification_converts_bgr_frame_to_rgb_pillow_image(
    dashboard_app,
):
    frame = dashboard_app.decode_image_bytes(_png_bytes((10, 40, 220)))
    classifier = FakeClassifier()

    prediction = dashboard_app.classify_full_image(frame, classifier)

    assert prediction.label == "paper"
    assert len(classifier.images) == 1
    assert classifier.images[0].mode == "RGB"
    assert classifier.images[0].size == (5, 4)
    assert classifier.images[0].getpixel((0, 0)) == (10, 40, 220)


def test_category_crop_classification_uses_clamped_yolo_proposal_crop(dashboard_app):
    from vista.detection.object_detector import BoundingBox, Detection

    class CropClassifier:
        def __init__(self):
            self.images = []

        def predict(self, image):
            self.images.append(image.copy())
            return SimpleNamespace(
                category="plastic",
                candidate_category="plastic",
                score=0.73,
                is_uncertain=False,
                status="PROVISIONAL folder-derived category.",
            )

    image = Image.new("RGB", (10, 8), (14, 80, 150))
    detection = Detection(BoundingBox(-2, 1, 5, 6), 39, "bottle", 0.86)
    classifier = CropClassifier()

    results = dashboard_app.classify_yolo_proposed_regions(image, [detection], classifier)

    assert len(results) == 1
    assert results[0]["detection"] is detection
    assert results[0]["crop"].size == (5, 5)
    assert classifier.images[0].getpixel((0, 0)) == (14, 80, 150)
    row = dashboard_app.category_crop_prediction_row(results[0])
    assert row["proposal"] == "bottle"
    assert row["category"] == "plastic · PROVISIONAL"
    assert row["raw_score"] == "73.0% · uncalibrated"


def test_category_crop_classification_handles_no_detections_without_inference(
    dashboard_app,
):
    classifier = FakeClassifier()
    results = dashboard_app.classify_yolo_proposed_regions(
        Image.new("RGB", (8, 8)), [], classifier
    )
    assert results == []
    assert classifier.images == []


def test_category_crop_classification_reports_invalid_crop_without_model_call(
    dashboard_app,
):
    from vista.detection.object_detector import BoundingBox, Detection

    classifier = FakeClassifier()
    detection = Detection(BoundingBox(20, 20, 30, 30), 39, "bottle", 0.8)
    results = dashboard_app.classify_yolo_proposed_regions(
        Image.new("RGB", (8, 8)), [detection], classifier
    )

    assert len(results) == 1
    assert results[0]["crop"] is None
    assert "no visible area" in results[0]["error"]
    assert classifier.images == []
    assert dashboard_app.category_crop_prediction_row(results[0])["category"] == (
        "Unavailable — review crop"
    )


def test_category_crop_classification_skips_person_boxes(dashboard_app):
    from vista.detection.object_detector import BoundingBox, Detection

    classifier = FakeClassifier()
    person = Detection(BoundingBox(0, 0, 6, 6), 0, "person", 0.99)
    results = dashboard_app.classify_yolo_proposed_regions(
        Image.new("RGB", (8, 8)), [person], classifier
    )
    assert results == []
    assert classifier.images == []


def test_uncertain_category_crop_keeps_candidate_withheld_below_existing_cutoff(
    dashboard_app,
):
    from vista.classification.taxonomy import WasteCategory
    from vista.detection.object_detector import BoundingBox, Detection

    prediction = SimpleNamespace(
        category=None,
        candidate_category=WasteCategory.GLASS,
        score=0.625,
        is_uncertain=True,
        status="PROVISIONAL score below 70%; manual review recommended.",
    )

    row = dashboard_app.category_crop_prediction_row(
        {
            "detection": Detection(BoundingBox(0, 0, 8, 8), 39, "bottle", 0.9),
            "crop": Image.new("RGB", (8, 8)),
            "prediction": prediction,
            "error": None,
        }
    )

    assert row["proposal"] == "bottle"
    assert row["category"] == "Uncertain — top candidate glass; manual review recommended"
    assert row["raw_score"] == "62.5% · uncalibrated"
    assert "below 70%" in row["status"]


def test_yolo_crop_diagnostics_never_replace_whole_image_recommendation_input(dashboard_app):
    whole_image_prediction = SimpleNamespace(category="glass", is_uncertain=True)
    category_prediction, error, source = dashboard_app.whole_image_recommendation_input(
        whole_image_prediction, "whole-image classifier error"
    )
    assert category_prediction is whole_image_prediction
    assert error == "whole-image classifier error"
    assert source == "whole-image"


def test_whole_image_prediction_is_labeled_separately_and_keeps_uncertainty(
    dashboard_app,
):
    prediction = SimpleNamespace(
        label=None, confidence=0.64, threshold=0.70, is_uncertain=True
    )

    row = dashboard_app.whole_image_prediction_row(prediction)

    assert row["Image scope"] == "Whole image"
    assert row["VISTA pilot category"] == (
        "Uncertain — manual review recommended"
    )
    assert row["Raw confidence"] == "64.0%"
    assert row["Status"] == "Below provisional threshold"
    assert "provisional" in dashboard_app.PILOT_CLASSIFIER_NOTICE


def test_annotate_detections_keeps_source_frame_unchanged(dashboard_app):
    from vista.detection.object_detector import BoundingBox, Detection

    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    original = frame.copy()
    detection = Detection(
        box=BoundingBox(5, 5, 20, 20),
        class_id=39,
        class_label="bottle",
        confidence=0.875,
    )

    annotated = dashboard_app.annotate_detections(frame, [detection])

    assert np.array_equal(frame, original)
    assert tuple(annotated[5, 5]) == (0, 165, 255)
    assert dashboard_app.FAMILY_RESULT == "Uncertain"
    assert "No completed family checkpoint" in dashboard_app.FAMILY_STATUS
    assert "Grad-CAM" in dashboard_app.EXPLAINABILITY_NOTICE


def test_annotation_hides_person_boxes_by_default_but_preserves_full_yolo_results(
    dashboard_app,
):
    from vista.detection.object_detector import BoundingBox, Detection

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    detections = [
        Detection(BoundingBox(2, 2, 12, 12), 0, "person", 0.9),
        Detection(BoundingBox(22, 22, 35, 35), 39, "bottle", 0.8),
    ]

    visible_default = dashboard_app.annotation_detections(detections)
    annotated_default = dashboard_app.annotate_detections(frame, detections)
    annotated_with_people = dashboard_app.annotate_detections(
        frame, detections, show_person_boxes=True
    )

    assert [detection.class_label for detection in visible_default] == ["bottle"]
    assert tuple(annotated_default[2, 2]) == (0, 0, 0)
    assert tuple(annotated_default[22, 22]) == (0, 165, 255)
    assert tuple(annotated_with_people[2, 2]) == (0, 165, 255)
    # The displayed generic detection results still contain every YOLO result.
    assert [row["Generic label"] for row in dashboard_app.detection_table_rows(detections)] == [
        "person", "bottle"
    ]


def test_detection_table_rows_show_generic_labels_and_confidence(dashboard_app):
    from vista.detection.object_detector import BoundingBox, Detection

    detections = [
        Detection(
            box=BoundingBox(1, 2, 3, 4),
            class_id=39,
            class_label="bottle",
            confidence=0.875,
        )
    ]

    assert dashboard_app.detection_table_rows(detections) == [
        {"Generic label": "bottle", "Confidence": "87.5%", "Class ID": 39}
    ]


def test_status_cards_are_honest_for_ready_complete_and_error_states(dashboard_app):
    ready = dict(dashboard_app.status_card_values(False, 0))
    complete = dict(dashboard_app.status_card_values(True, 2))
    error = dict(dashboard_app.status_card_values(False, 0, has_error=True))

    assert ready["Detection state"] == "Ready"
    assert complete["Detection state"] == "Complete"
    assert complete["Generic YOLO boxes"] == "2"
    assert error["Detection state"] == "Error"
    assert complete["Verified family model"] == "Unavailable"
    assert complete["Explainability"] == "Unavailable"


def test_dashboard_messages_do_not_claim_waste_classification(dashboard_app):
    assert dashboard_app.FAMILY_RESULT == "Uncertain"
    assert "No completed family checkpoint" in dashboard_app.FAMILY_STATUS
    assert "not VISTA waste classifications" in dashboard_app.GENERAL_DETECTION_NOTICE
    assert "not available yet" in dashboard_app.EXPLAINABILITY_NOTICE


def test_family_prediction_display_is_explicit_and_fail_closed(dashboard_app):
    from vista.classification.taxonomy import WasteFamily
    from vista.classification.waste_family_classifier import (
        UnavailableWasteFamilyClassifier,
        WasteFamilyPrediction,
    )

    unavailable = UnavailableWasteFamilyClassifier().predict(Image.new("RGB", (4, 4)))
    assert dashboard_app.family_prediction_table_row(unavailable) == {
        "Waste family": "Uncertain — verified MSW/E-waste model not available",
        "Raw model score": "Unavailable",
        "Status": unavailable.status,
    }
    assert dashboard_app.family_prediction_table_row(
        WasteFamilyPrediction(
            WasteFamily.E_WASTE, 0.61, False, "A future model result."
        )
    ) == {
        "Waste family": "E_WASTE",
        "Raw model score": "61.0%",
        "Status": "Experimental — A future model result.",
    }
    assert dashboard_app.family_prediction_table_row(error="checkpoint missing") == {
        "Waste family": "Uncertain — verified MSW/E-waste model not available",
        "Raw model score": "Unavailable",
        "Status": "Family model unavailable: checkpoint missing",
    }
    assert dashboard_app.FAMILY_RESULT == "Uncertain"
    assert "No completed family checkpoint" in dashboard_app.FAMILY_STATUS


def test_provisional_family_display_is_separate_and_never_claims_verified_status(dashboard_app):
    assert dashboard_app.provisional_family_prediction_row(error="checkpoint missing") == {
        "PROVISIONAL family": "Unavailable",
        "Raw score": "Unavailable",
        "Status": "Provisional model unavailable: checkpoint missing",
    }
    assert dashboard_app.provisional_family_prediction_row({
        "family": None, "score": 0.62, "is_uncertain": True,
        "status": "PROVISIONAL score below 70%; manual review recommended.",
    }) == {
        "PROVISIONAL family": "Uncertain — manual review recommended",
        "Raw score": "62.0%",
        "Status": "PROVISIONAL score below 70%; manual review recommended.",
    }
    assert dashboard_app.provisional_family_prediction_row({
        "family": "MSW", "score": 0.91, "is_uncertain": False,
        "status": "PROVISIONAL folder-derived output; not verified or calibrated.",
    })["PROVISIONAL family"] == "MSW · PROVISIONAL"
    assert dashboard_app.FAMILY_RESULT == "Uncertain"


def test_provisional_category_display_includes_confidence_and_withholds_uncertain_label(dashboard_app):
    from vista.classification.provisional_waste_category_classifier import (
        ProvisionalCategoryPrediction,
    )
    from vista.classification.taxonomy import WasteCategory, WasteFamily

    certain = ProvisionalCategoryPrediction(
        WasteCategory.BATTERY, WasteCategory.BATTERY, WasteFamily.E_WASTE,
        0.91, False, "PROVISIONAL folder-derived output.",
    )
    uncertain = ProvisionalCategoryPrediction(
        None, WasteCategory.PAPER, WasteFamily.MSW,
        0.48, True, "PROVISIONAL low score; manual review recommended.",
    )
    assert dashboard_app.provisional_category_prediction_row(certain) == {
        "PROVISIONAL image category": "battery · PROVISIONAL",
        "Raw score": "91.0%",
        "Status": "PROVISIONAL folder-derived output.",
    }
    assert "Uncertain" in dashboard_app.provisional_category_prediction_row(uncertain)["PROVISIONAL image category"]
    assert dashboard_app.provisional_category_prediction_row(uncertain)["Raw score"] == "48.0%"
    assert "battery collection point" in dashboard_app.CATEGORY_GUIDANCE["battery"][1]


def test_provisional_category_diagnostics_show_top_three_raw_scores(dashboard_app):
    from vista.classification.taxonomy import WasteCategory

    prediction = SimpleNamespace(
        top_candidates=(
            (WasteCategory.CARDBOARD, 0.522),
            (WasteCategory.PAPER, 0.201),
            (WasteCategory.PLASTIC, 0.117),
            (WasteCategory.GLASS, 0.08),
        )
    )
    rows = dashboard_app.provisional_category_diagnostic_rows(prediction)

    assert rows == [
        {"Rank": "1", "Candidate": "cardboard", "Raw score": "52.2%"},
        {"Rank": "2", "Candidate": "paper", "Raw score": "20.1%"},
        {"Rank": "3", "Candidate": "plastic", "Raw score": "11.7%"},
    ]


def test_verified_family_unavailable_is_distinguished_from_provisional_baseline(dashboard_app):
    row = dashboard_app.family_prediction_table_row(error="checkpoint missing")
    assert row["Waste family"] == "Uncertain — verified MSW/E-waste model not available"
    assert "PROVISIONAL" in dashboard_app.provisional_family_prediction_row({
        "family": "MSW", "score": 1.0, "is_uncertain": False,
        "status": "Folder-derived provisional output.",
    })["PROVISIONAL family"]
    assert "checkpoint is unavailable" in dashboard_app.VERIFIED_FAMILY_UNAVAILABLE_NOTICE
    assert "not a verified substitute" in dashboard_app.VERIFIED_FAMILY_UNAVAILABLE_NOTICE


def test_provisional_family_category_disagreement_is_explicit(dashboard_app):
    from vista.classification.provisional_waste_category_classifier import (
        ProvisionalCategoryPrediction,
    )
    from vista.classification.taxonomy import WasteCategory, WasteFamily

    category = ProvisionalCategoryPrediction(
        WasteCategory.BATTERY, WasteCategory.BATTERY, WasteFamily.E_WASTE,
        0.89, False, "PROVISIONAL.",
    )
    assert dashboard_app.provisional_model_consistency(
        {"family": WasteFamily.MSW, "is_uncertain": False}, category
    ) == "conflict"
    assert dashboard_app.provisional_model_consistency(
        {"family": WasteFamily.E_WASTE, "is_uncertain": False}, category
    ) == "aligned"
    assert dashboard_app.provisional_model_consistency(
        {"family": None, "is_uncertain": True}, category
    ) == "inconclusive"


def test_generic_detection_counts_only_appear_for_multiple_yolo_boxes(dashboard_app):
    from vista.detection.object_detector import BoundingBox, Detection

    single = [Detection(BoundingBox(1, 1, 4, 4), 39, "bottle", 0.8)]
    multiple = single + [Detection(BoundingBox(5, 1, 8, 4), 39, "bottle", 0.7)]
    assert dashboard_app.generic_detection_breakdown(single) == []
    assert dashboard_app.generic_detection_breakdown(multiple) == [
        {"Generic YOLO label": "bottle", "Detected boxes": 2}
    ]


def test_result_summary_keeps_family_category_yolo_and_uncertainty_separate(dashboard_app):
    from vista.classification.provisional_waste_category_classifier import (
        ProvisionalCategoryPrediction,
    )
    from vista.classification.taxonomy import WasteCategory, WasteFamily
    from vista.detection.object_detector import BoundingBox, Detection

    category = ProvisionalCategoryPrediction(
        None, WasteCategory.GLASS, WasteFamily.MSW, 0.485, True,
        "PROVISIONAL score below 70%; manual review recommended.",
    )
    detections = [Detection(BoundingBox(1, 1, 4, 4), 0, "person", 1.0)]
    summary = dashboard_app.result_summary(
        {"family": WasteFamily.MSW, "score": 1.0, "is_uncertain": False},
        category,
        detections,
    )
    assert summary == {
        "Family prediction": "MSW · PROVISIONAL",
        "Category prediction": "Uncertain — top candidate glass",
        "Generic YOLO detections": "1 generic YOLO box(es) — not waste items",
        "Uncertain": "Yes — manual review recommended",
    }
    assert "not evidence" in dashboard_app.GENERIC_DETECTION_WARNING
    assert "never counted as waste items" in dashboard_app.GENERIC_DETECTION_WARNING


def test_category_guidance_is_available_for_all_supported_classes_and_disclaimed(dashboard_app):
    expected = {"paper", "cardboard", "plastic", "glass", "organic", "battery"}
    assert set(dashboard_app.CATEGORY_GUIDANCE) == expected
    for category in expected:
        guidance = dashboard_app.category_guidance(category)
        assert guidance["meaning"]
        assert guidance["handling"]
        assert "not a confirmed" in guidance["notice"]
    with pytest.raises(ValueError, match="No supported guidance"):
        dashboard_app.category_guidance("metal")


def test_pilot_classifier_display_is_separate_and_marks_uncertainty(dashboard_app):
    predicted = SimpleNamespace(
        error=None,
        prediction=SimpleNamespace(
            label="paper", confidence=0.83, is_uncertain=False
        ),
    )
    uncertain = SimpleNamespace(
        error=None,
        prediction=SimpleNamespace(
            label=None, confidence=0.62, is_uncertain=True
        ),
    )
    invalid_crop = SimpleNamespace(
        error="Bounding box has no visible area inside the image.",
        prediction=None,
    )

    rows = dashboard_app.classification_table_rows(
        [predicted, uncertain, invalid_crop]
    )

    assert rows[0]["VISTA pilot category"] == "paper"
    assert rows[0]["Raw confidence"] == "83.0%"
    assert rows[1]["VISTA pilot category"] == (
        "Uncertain — manual review recommended"
    )
    assert rows[1]["Raw confidence"] == "62.0%"
    assert rows[2]["VISTA pilot category"] == "Unavailable — review crop"
    assert "provisional" in dashboard_app.PILOT_CLASSIFIER_NOTICE
    assert dashboard_app.FAMILY_RESULT == "Uncertain"
    assert "No completed family checkpoint" in dashboard_app.FAMILY_STATUS
    assert "Grad-CAM" in dashboard_app.EXPLAINABILITY_NOTICE


def test_main_page_uses_compact_vista_brand_and_sorting_copy(dashboard_app):
    assert dashboard_app.PROJECT_NAME == "VISTA"
    assert dashboard_app.APP_SUBTITLE == "Intelligent Waste Sorting"
    assert "Capture or upload one waste item" in dashboard_app.PROJECT_DESCRIPTION
    assert "agree" in dashboard_app.PROJECT_DESCRIPTION
    assert not hasattr(dashboard_app, "NAV_ITEMS")


def test_primary_result_display_keeps_uncertainty_and_candidate_separate(dashboard_app):
    from vista.classification.taxonomy import WasteCategory, WasteFamily
    from vista.classification.provisional_waste_category_classifier import (
        ProvisionalCategoryPrediction,
    )

    prediction = ProvisionalCategoryPrediction(
        None,
        WasteCategory.GLASS,
        WasteFamily.MSW,
        0.625,
        True,
        "PROVISIONAL score below 70%; manual review recommended.",
    )
    row = dashboard_app.primary_result_display(
        prediction,
        {"manual_review_required": True, "bin": None},
    )

    assert row["category"] == "Uncertain"
    assert row["candidate"] == "glass"
    assert row["score_label"] == "62.5% · raw, uncalibrated"
    assert row["manual_review_required"] is True
    assert row["bin"] is None


def test_primary_result_display_does_not_hide_family_category_conflict(dashboard_app):
    from vista.classification.taxonomy import WasteCategory, WasteFamily
    from vista.classification.provisional_waste_category_classifier import (
        ProvisionalCategoryPrediction,
    )

    prediction = ProvisionalCategoryPrediction(
        WasteCategory.BATTERY,
        WasteCategory.BATTERY,
        WasteFamily.E_WASTE,
        0.91,
        False,
        "PROVISIONAL folder-derived category.",
    )
    row = dashboard_app.primary_result_display(
        prediction,
        {
            "manual_review_required": True,
            "bin": None,
            "explanation": "The provisional family and category outputs conflict.",
        },
    )

    assert row["category"] == "battery · PROVISIONAL"
    assert row["score_label"] == "91.0% · raw, uncalibrated"
    assert row["manual_review_required"] is True
    assert row["bin"] is None


def test_category_stage_waits_for_certain_family_prediction(dashboard_app):
    class CategoryModel:
        def __init__(self):
            self.calls = 0

        def predict(self, _image):
            self.calls += 1
            return SimpleNamespace(
                category="plastic", candidate_category="plastic",
                family_mapping="MSW", is_uncertain=False,
            )

    classifier = CategoryModel()
    prediction, error = dashboard_app.classify_category_for_family(
        {"family": None, "is_uncertain": True}, Image.new("RGB", (4, 4)), classifier
    )

    assert prediction is None
    assert "category stage was not run" in error
    assert classifier.calls == 0


@pytest.mark.parametrize(
    ("family", "category", "category_family", "expected", "error_text"),
    [
        ("MSW", "plastic", "MSW", "plastic", None),
        ("MSW", "battery", "E_WASTE", None, "outside the MSW branch"),
        ("E_WASTE", "battery", "E_WASTE", "battery", None),
        ("E_WASTE", "glass", "MSW", None, "supports only the provisional battery category"),
    ],
)
def test_category_stage_routes_only_supported_branch_categories(
    dashboard_app, family, category, category_family, expected, error_text
):
    class CategoryModel:
        def predict(self, _image):
            return SimpleNamespace(
                category=category,
                candidate_category=category,
                family_mapping=category_family,
                is_uncertain=False,
            )

    prediction, error = dashboard_app.classify_category_for_family(
        {"family": family, "is_uncertain": False}, Image.new("RGB", (4, 4)), CategoryModel()
    )
    if expected is None:
        assert prediction is None
        assert error_text in error
    else:
        assert prediction.category == expected
        assert error is None


def test_optional_yolo_diagnostics_render_without_undefined_notice(dashboard_app):
    class FakeUI:
        def __init__(self):
            self.messages = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def expander(self, *_args, **_kwargs):
            return self

        def caption(self, value):
            self.messages.append(str(value))

        def warning(self, value):
            self.messages.append(str(value))

    ui = FakeUI()
    dashboard_app.render_optional_generic_diagnostics(ui, {"detections": []})

    assert dashboard_app.GENERAL_DETECTION_NOTICE in ui.messages
    assert any("never counted as waste items" in message for message in ui.messages)


def test_optional_yolo_diagnostic_failure_is_contained(dashboard_app):
    class BrokenUI:
        def expander(self, *_args, **_kwargs):
            raise RuntimeError("diagnostic renderer failed")

        def caption(self, value):
            self.message = str(value)

    ui = BrokenUI()
    dashboard_app.render_optional_generic_diagnostics(ui, {"detections": []})

    assert "Optional generic-object diagnostics unavailable" in ui.message
    assert "diagnostic renderer failed" in ui.message
