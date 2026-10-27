import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def comparison(monkeypatch):
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "src"))
    monkeypatch.syspath_prepend(str(project / "scripts"))
    import compare_clip_category

    return compare_clip_category


def _manifest_row(path, digest, *, split="test", status="provisional_include", canonical=None):
    return {
        "source_relative_path": path,
        "sha256": digest,
        "group_status": status,
        "canonical_path": canonical or path,
        "split": split,
    }


def _prediction(path, digest, label="paper", provenance="explicit human-reviewed category decision"):
    return {
        "task": "category",
        "source_relative_path": path,
        "sha256": digest,
        "expected_label": label,
        "label_provenance": provenance,
    }


def _write_inputs(tmp_path, rows, predictions):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"predictions": predictions}), encoding="utf-8")
    return report, manifest


def test_collect_uses_only_matching_canonical_heldout_images_and_excludes_train_leakage(comparison, tmp_path):
    root = tmp_path / "dataset"
    (root / "paper").mkdir(parents=True)
    image_path = root / "paper" / "one.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    eligible = root / "paper" / "two.png"
    Image.new("RGB", (8, 8), "black").save(eligible)
    eligible_digest = hashlib.sha256(eligible.read_bytes()).hexdigest()
    rows = [
        _manifest_row("paper/one.png", digest),
        _manifest_row("plastic/leaked.png", digest, split="train"),
        _manifest_row("paper/two.png", eligible_digest),
    ]
    report, manifest = _write_inputs(tmp_path, rows, [
        _prediction("paper/one.png", digest), _prediction("paper/two.png", eligible_digest)
    ])

    cases, excluded = comparison.collect_reviewed_test_images(report, manifest, root)

    assert len(cases) == 1
    assert cases[0]["source_relative_path"] == "paper/two.png"
    assert any("training or validation" in row["reason"] for row in excluded)


@pytest.mark.parametrize("unsafe", ["../outside.png", "C:\\outside.png", "/outside.png"])
def test_safe_relative_path_rejects_traversal_and_absolute_paths(comparison, unsafe):
    with pytest.raises(comparison.ComparisonError, match="Unsafe"):
        comparison.safe_relative_path(unsafe)


def test_collect_excludes_wrong_hash_noncanonical_and_nonhuman_label(comparison, tmp_path):
    image_path = tmp_path / "dataset" / "paper" / "one.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    rows = [
        _manifest_row("paper/one.png", digest),
        _manifest_row("paper/copy.png", digest, canonical="paper/one.png"),
    ]
    predictions = [
        _prediction("paper/one.png", "0" * 64),
        _prediction("paper/copy.png", digest),
        _prediction("paper/one.png", digest, provenance="folder-derived label"),
    ]
    report, manifest = _write_inputs(tmp_path, rows, predictions)

    cases, excluded = comparison.collect_reviewed_test_images(report, manifest, image_path.parents[2])

    assert cases == []
    assert len(excluded) == 3


class _Category:
    value = "cardboard"


class _Candidate:
    value = "cardboard"


class _BaselinePrediction:
    category = _Category()
    candidate_category = _Candidate()
    score = 0.669
    is_uncertain = True


class _Baseline:
    def predict(self, image):
        return _BaselinePrediction()


class _Clip:
    def predict(self, image):
        return {"top_category": "cardboard", "top_similarity": 0.31, "top2_margin": 0.01}


def test_evaluation_preserves_uncertain_baseline_and_treats_clip_as_ranking_only(comparison, tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "gray").save(image_path)
    report = comparison.evaluate_comparison(
        [{"source_relative_path": "paper/sample.png", "sha256": "a" * 64,
          "expected_label": "cardboard", "image_path": image_path}],
        _Baseline(), _Clip(),
    )

    assert report["current_model_uncertain_count"] == 1
    assert report["current_model_top1"]["confusion_matrix"]["cardboard"]["cardboard"] == 1
    assert report["current_model_operating_point"]["confusion_matrix"]["cardboard"]["Uncertain"] == 1
    assert report["current_model_uncertainty"]["coverage"] == 0.0
    assert report["clip_top1"]["confusion_matrix"]["cardboard"]["cardboard"] == 1
    assert "not probabilities" in report["clip_score_note"]
    assert "Not calibrated" in report["clip_uncertainty"]


def test_crop_proposal_ignores_person_and_selects_largest_region(comparison):
    from vista.detection.object_detector import BoundingBox, Detection

    detections = [
        Detection(BoundingBox(0, 0, 8, 8), 0, "person", 0.99),
        Detection(BoundingBox(1, 1, 4, 4), 39, "bottle", 0.55),
        Detection(BoundingBox(2, 2, 7, 7), 41, "cup", 0.40),
    ]

    selected = comparison.select_largest_non_person_detection(detections)

    assert selected.class_label == "cup"
    assert selected.class_label != "person"


def test_crop_diagnostics_keep_no_proposal_and_invalid_crop_uncertain(comparison, tmp_path):
    from vista.detection.object_detector import BoundingBox, Detection

    image_path = tmp_path / "test.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    class Detector:
        def __init__(self, detections):
            self.detections = iter(detections)

        def detect(self, _path):
            return next(self.detections)

    class CategoryClassifier:
        def predict(self, _image):
            raise AssertionError("Invalid crops should not reach the classifier.")

    cases = [
        {"source_relative_path": "paper/no-proposal.png", "expected_label": "paper", "image_path": image_path},
        {"source_relative_path": "paper/outside.png", "expected_label": "paper", "image_path": image_path},
    ]
    result = comparison.evaluate_yolo_crop_diagnostics(
        cases,
        Detector([[], [Detection(BoundingBox(20, 20, 30, 30), 39, "bottle", 0.8)]]),
        CategoryClassifier(),
    )

    assert result["images_without_non_person_proposal"] == 1
    assert result["crop_or_inference_errors"] == 1
    assert result["uncertain_count_including_missing_crops"] == 2
    assert all(row["category_prediction"] == "Uncertain" for row in result["predictions"])


def test_crop_diagnostics_classify_pixels_only_and_keep_generic_label_as_metadata(comparison, tmp_path):
    from vista.detection.object_detector import BoundingBox, Detection

    image_path = tmp_path / "test.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    class Detector:
        def detect(self, _path):
            return [Detection(BoundingBox(2, 2, 6, 6), 39, "bottle", 0.9)]

    class CategoryClassifier:
        def predict(self, image):
            assert image.size == (4, 4)
            return type(
                "Prediction", (), {
                    "is_uncertain": False,
                    "category": type("Category", (), {"value": "paper"})(),
                    "candidate_category": type("Category", (), {"value": "paper"})(),
                    "score": 0.8,
                }
            )()

    result = comparison.evaluate_yolo_crop_diagnostics(
        [{"source_relative_path": "paper/test.png", "expected_label": "paper", "image_path": image_path}],
        Detector(), CategoryClassifier(),
    )

    row = result["predictions"][0]
    assert row["proposal_label_generic_only"] == "bottle"
    assert row["category_prediction"] == "paper"
    assert "ROI proposal" in result["generic_label_use"]
    assert result["metrics_with_uncertain_abstentions"]["accuracy"] == 1.0
