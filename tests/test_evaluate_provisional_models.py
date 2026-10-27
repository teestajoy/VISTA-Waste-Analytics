import csv
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_provisional_models.py"
SPEC = importlib.util.spec_from_file_location("evaluate_provisional_models", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class FakeFamilyClassifier:
    def predict(self, image):
        return {"family": "MSW", "score": 0.64, "is_uncertain": True}


class FakeCategoryClassifier:
    def predict(self, image):
        return SimpleNamespace(
            category=None,
            candidate_category="cardboard",
            score=0.522,
            is_uncertain=True,
        )


class CertainFamilyClassifier:
    def predict(self, image):
        return {"family": "E_WASTE", "score": 0.92, "is_uncertain": False}


class CertainCategoryClassifier:
    def predict(self, image):
        return SimpleNamespace(
            category="paper",
            candidate_category="paper",
            score=0.93,
            is_uncertain=False,
        )


def _write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), "red").save(path, format="PNG")


def test_evaluation_reports_provisional_predictions_and_independent_metrics(tmp_path):
    image_path = tmp_path / "images" / "cup.png"
    _write_image(image_path)
    rows = [{
        "filename": "cup.png",
        "expected_category": "cardboard",
        "expected_family": "MSW",
    }]

    result = evaluation.evaluate(
        image_path.parent, rows, FakeFamilyClassifier(), FakeCategoryClassifier(), ()
    )

    record = result["records"][0]
    assert record["provisional_family"] == "Uncertain"
    assert record["family_raw_top_score"] == 0.64
    assert record["family_uncertain"] is True
    assert record["provisional_category"] == "Uncertain"
    assert record["category_top_candidate"] == "cardboard"
    assert record["category_raw_top_score"] == 0.522
    assert result["family_metrics"]["accuracy"] == 0.0
    assert result["category_metrics"]["accuracy"] == 0.0
    assert result["category_metrics"]["confusion_matrix"]["cardboard"]["Uncertain"] == 1


def test_no_expected_labels_means_no_aggregate_metrics(tmp_path):
    image = tmp_path / "sample.png"
    _write_image(image)
    rows = [{"filename": "sample.png", "expected_category": "", "expected_family": ""}]

    result = evaluation.evaluate(
        tmp_path, rows, FakeFamilyClassifier(), FakeCategoryClassifier(), ()
    )

    assert result["category_metrics"] is None
    assert result["family_metrics"] is None
    assert result["unlabeled_count"] == 1


def test_manifest_hash_overlap_is_reported(tmp_path):
    image = tmp_path / "sample.png"
    _write_image(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256"])
        writer.writeheader()
        writer.writerow({"sha256": digest})

    result = evaluation.evaluate(
        tmp_path,
        [{"filename": "sample.png", "expected_category": "", "expected_family": ""}],
        FakeFamilyClassifier(), FakeCategoryClassifier(), (manifest,),
    )

    assert result["overlap_files"] == ["sample.png"]
    assert result["records"][0]["overlaps_existing_manifest"] is True


def test_csv_rejects_invalid_expected_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("filename,expected_category\nitem.jpg,metal\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid expected_category"):
        evaluation.read_expected_labels(path)


@pytest.mark.parametrize("filename", ["../escape.png", "C:\\outside.png"])
def test_image_path_rejects_traversal_and_absolute_paths(tmp_path, filename):
    with pytest.raises(ValueError, match="Unsafe image path"):
        evaluation.resolve_image_path(tmp_path, filename)


def _manifest_row(path, digest, split="test", canonical=None, status="provisional_include"):
    return {
        "source_relative_path": path,
        "sha256": digest,
        "canonical_path": canonical if canonical is not None else path,
        "split": split,
        "group_status": status,
    }


def _decision(path, status="include", label="paper", task="category"):
    if task == "family":
        return {
            "source_relative_path": path,
            "review_status": status,
            "reviewed_family": label,
            "reason": "independently reviewed",
        }
    return {
        "source_relative_path": path,
        "decision": status,
        "reviewed_label": label,
        "reason": "independently reviewed",
    }


def test_reviewed_decision_matches_only_canonical_heldout_test_path_and_hash(tmp_path):
    image = tmp_path / "images" / "paper" / "one.png"
    _write_image(image)
    relative = "paper/one.png"
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    cases, excluded = evaluation.collect_reviewed_test_cases(
        image.parents[1], [], [_decision(relative)],
        [], [_manifest_row(relative, digest)],
    )

    assert len(cases) == 1
    assert cases[0]["expected_label"] == "paper"
    assert cases[0]["sha256"] == digest
    assert excluded == []


def test_reviewed_test_hash_also_in_train_or_validation_is_excluded(tmp_path):
    image = tmp_path / "images" / "one.png"
    _write_image(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    cases, excluded = evaluation.collect_reviewed_test_cases(
        image.parent, [], [_decision("one.png")],
        [], [_manifest_row("one.png", digest), _manifest_row("other.png", digest, "validation")],
    )

    assert cases == []
    assert "training or validation" in excluded[0]["reason"]


def test_duplicate_test_hashes_are_excluded_as_a_group(tmp_path):
    first = tmp_path / "images" / "one.png"
    second = tmp_path / "images" / "two.png"
    _write_image(first)
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(first.read_bytes())
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    decisions = [_decision("one.png"), _decision("two.png", label="cardboard")]
    manifest = [_manifest_row("one.png", digest), _manifest_row("two.png", digest)]

    cases, excluded = evaluation.collect_reviewed_test_cases(
        first.parent, [], decisions, [], manifest
    )

    assert cases == []
    assert len(excluded) == 2
    assert all("duplicate" in row["reason"] for row in excluded)


def test_held_and_missing_label_decisions_are_excluded(tmp_path):
    held_image = tmp_path / "held.png"
    blank_image = tmp_path / "blank.png"
    _write_image(held_image)
    _write_image(blank_image)
    decisions = [
        _decision("held.png", status="hold", label=""),
        _decision("blank.png", status="include", label=""),
    ]
    cases, excluded = evaluation.collect_reviewed_test_cases(
        tmp_path, [], decisions,
        [], [
            _manifest_row("held.png", hashlib.sha256(held_image.read_bytes()).hexdigest()),
            _manifest_row("blank.png", hashlib.sha256(blank_image.read_bytes()).hexdigest()),
        ],
    )

    assert cases == []
    assert "decision is hold" in excluded[0]["reason"]
    assert "missing or invalid" in excluded[1]["reason"]


def test_uncertain_prediction_is_reported_and_counts_as_uncertain(tmp_path):
    image = tmp_path / "one.png"
    _write_image(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    case = {
        "task": "category", "source_relative_path": "one.png", "sha256": digest,
        "expected_label": "cardboard", "image_path": image,
    }

    result = evaluation.evaluate_reviewed_test_cases(
        [case], FakeFamilyClassifier(), FakeCategoryClassifier()
    )

    assert len(result["uncertain_predictions"]) == 1
    assert result["uncertain_predictions"][0]["provisional_prediction"] == "Uncertain"
    assert result["category_metrics"]["confusion_matrix"]["cardboard"]["Uncertain"] == 1
    assert result["independently_reviewed_category_examples"] == 1


def test_test_batch_decisions_are_adapted_only_after_queue_path_hash_match(tmp_path):
    digest = "a" * 64
    queue = tmp_path / "queue.csv"
    decisions = tmp_path / "decisions.csv"
    queue.write_text(
        f"source_relative_path,sha256\npaper/item.jpg,{digest}\n", encoding="utf-8"
    )
    decisions.write_text(
        "source_relative_path,sha256,review_type,decision,reviewed_label,reason\n"
        f"paper/item.jpg,{digest},family,include,MSW,\n"
        f"paper/item.jpg,{digest},category,include,paper,\n",
        encoding="utf-8",
    )

    family, category = evaluation.read_test_batch_decisions(decisions, queue)

    assert family[0]["review_status"] == "include"
    assert family[0]["reviewed_family"] == "MSW"
    assert family[0]["_review_sha256"] == digest
    assert category[0]["decision"] == "include"
    assert category[0]["reviewed_label"] == "paper"


def test_test_batch_decisions_reject_queue_hash_mismatch(tmp_path):
    queue = tmp_path / "queue.csv"
    decisions = tmp_path / "decisions.csv"
    queue.write_text(f"source_relative_path,sha256\none.jpg,{'a' * 64}\n", encoding="utf-8")
    decisions.write_text(
        "source_relative_path,sha256,review_type,decision,reviewed_label,reason\n"
        f"one.jpg,{'b' * 64},family,include,MSW,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the saved review queue"):
        evaluation.read_test_batch_decisions(decisions, queue)


def test_test_batch_decisions_require_reason_for_holds(tmp_path):
    digest = "c" * 64
    queue = tmp_path / "queue.csv"
    decisions = tmp_path / "decisions.csv"
    queue.write_text(f"source_relative_path,sha256\none.jpg,{digest}\n", encoding="utf-8")
    decisions.write_text(
        "source_relative_path,sha256,review_type,decision,reviewed_label,reason\n"
        f"one.jpg,{digest},family,hold,, \n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a reason"):
        evaluation.read_test_batch_decisions(decisions, queue)


def test_batch_merge_replaces_pending_in_memory_but_rejects_explicit_conflict():
    pending = [{
        "source_relative_path": "one.jpg", "review_status": "pending", "reviewed_family": ""
    }]
    batch = [{
        "source_relative_path": "one.jpg", "review_status": "include", "reviewed_family": "MSW"
    }]
    merged = evaluation.merge_review_rows(pending, batch, "family")
    assert len(merged) == 1
    assert merged[0]["reviewed_family"] == "MSW"

    explicit = [{
        "source_relative_path": "one.jpg", "review_status": "include", "reviewed_family": "E_WASTE"
    }]
    with pytest.raises(ValueError, match="Conflicting explicit family review decisions"):
        evaluation.merge_review_rows(explicit, batch, "family")


def test_previous_and_followup_batch_decisions_combine_without_overwriting(tmp_path):
    digest_a, digest_b = "a" * 64, "b" * 64
    previous_queue, previous_decisions = tmp_path / "previous_queue.csv", tmp_path / "previous.csv"
    followup_queue, followup_decisions = tmp_path / "followup_queue.csv", tmp_path / "followup.csv"
    previous_queue.write_text(
        f"source_relative_path,sha256\none.jpg,{digest_a}\n", encoding="utf-8"
    )
    followup_queue.write_text(
        f"source_relative_path,sha256\ntwo.jpg,{digest_b}\n", encoding="utf-8"
    )
    header = "source_relative_path,sha256,review_type,decision,reviewed_label,reason\n"
    previous_decisions.write_text(header + f"one.jpg,{digest_a},family,include,MSW,\n", encoding="utf-8")
    followup_decisions.write_text(header + f"two.jpg,{digest_b},family,include,E_WASTE,\n", encoding="utf-8")

    previous_family, previous_category = evaluation.read_test_batch_decisions(
        previous_decisions, previous_queue
    )
    next_family, next_category = evaluation.read_test_batch_decisions(
        followup_decisions, followup_queue
    )
    merged_family = evaluation.merge_review_rows(previous_family, next_family, "family")

    assert [row["source_relative_path"] for row in merged_family] == ["one.jpg", "two.jpg"]
    assert [row["reviewed_family"] for row in merged_family] == ["MSW", "E_WASTE"]
    assert previous_decisions.read_text(encoding="utf-8").endswith(f"one.jpg,{digest_a},family,include,MSW,\n")
    assert previous_category == next_category == []


def test_three_test_review_batches_route_family_and_category_decisions(tmp_path):
    header = "source_relative_path,sha256,review_type,decision,reviewed_label,reason\n"
    entries = (
        ("one.jpg", "1" * 64, "family", "MSW"),
        ("two.jpg", "2" * 64, "family", "E_WASTE"),
        ("three.jpg", "3" * 64, "category", "glass"),
    )
    inputs = []
    for index, (path, digest, task, label) in enumerate(entries):
        queue, decisions = tmp_path / f"queue{index}.csv", tmp_path / f"decisions{index}.csv"
        queue.write_text(f"source_relative_path,sha256\n{path},{digest}\n", encoding="utf-8")
        decisions.write_text(header + f"{path},{digest},{task},include,{label},\n", encoding="utf-8")
        inputs.append((decisions, queue))

    family, category = evaluation.merge_test_batch_inputs([], [], inputs)

    assert [(row["source_relative_path"], row["reviewed_family"]) for row in family] == [
        ("one.jpg", "MSW"), ("two.jpg", "E_WASTE")
    ]
    assert [(row["source_relative_path"], row["reviewed_label"]) for row in category] == [
        ("three.jpg", "glass")
    ]


def test_test_batch_digest_must_match_heldout_manifest(tmp_path):
    image = tmp_path / "one.png"
    _write_image(image)
    actual = hashlib.sha256(image.read_bytes()).hexdigest()
    decision = _decision("one.png", task="family", label="MSW")
    decision["_review_sha256"] = "f" * 64
    cases, excluded = evaluation.collect_reviewed_test_cases(
        tmp_path, [decision], [],
        [_manifest_row("one.png", actual)], [],
    )

    assert cases == []
    assert "review decision SHA-256 differs" in excluded[0]["reason"]
