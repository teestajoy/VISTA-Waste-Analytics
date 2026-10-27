"""Evaluate provisional VISTA models against existing human review decisions.

The command reads the family and paper/cardboard review CSVs, then includes only
explicit include decisions that match canonical, hash-verified held-out test
rows. Manifest folder labels and model predictions are never used as expected
labels. All model outputs and metrics remain provisional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
DEFAULT_FAMILY_CHECKPOINT = PROJECT_ROOT / "models" / "waste_family_provisional" / "best_model.pth"
DEFAULT_CATEGORY_CHECKPOINT = PROJECT_ROOT / "models" / "waste_category_provisional" / "best_model.pth"
DEFAULT_TRAINING_MANIFESTS = (
    PROJECT_ROOT / "data" / "processed" / "family_provisional_split_manifest.csv",
    PROJECT_ROOT / "models" / "waste_category_provisional" / "split_manifest.csv",
)
DEFAULT_FAMILY_REVIEW = PROJECT_ROOT / "data" / "processed" / "family_review_decisions.csv"
DEFAULT_CATEGORY_REVIEW = PROJECT_ROOT / "data" / "processed" / "paper_cardboard_pilot_review.csv"
DEFAULT_FAMILY_MANIFEST = PROJECT_ROOT / "data" / "processed" / "family_provisional_split_manifest.csv"
DEFAULT_CATEGORY_MANIFEST = PROJECT_ROOT / "models" / "waste_category_provisional" / "split_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "reviewed_test_model_evaluation.json"
FAMILIES = ("MSW", "E_WASTE")
CATEGORIES = ("paper", "cardboard", "plastic", "glass", "organic", "battery")
MINIMUM_CLASS_SUPPORT_WARNING = 30


def read_expected_labels(csv_path: Path) -> list[dict[str, str]]:
    """Read independently assigned labels; never derive expected labels."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "filename" not in fields:
            raise ValueError("Label CSV must contain a 'filename' column.")
        if not fields.intersection({"expected_category", "expected_family"}):
            raise ValueError(
                "Label CSV must contain expected_category and/or expected_family; "
                "leave values blank only for images without an independent label."
            )
        rows = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"Label CSV row {row_number} has a blank filename.")
            if filename in seen:
                raise ValueError(f"Label CSV contains duplicate filename {filename!r}.")
            seen.add(filename)
            expected_category = (row.get("expected_category") or "").strip()
            expected_family = (row.get("expected_family") or "").strip()
            if expected_category and expected_category not in CATEGORIES:
                raise ValueError(
                    f"Invalid expected_category {expected_category!r} on row {row_number}; "
                    f"choose one of {', '.join(CATEGORIES)}."
                )
            if expected_family and expected_family not in FAMILIES:
                raise ValueError(
                    f"Invalid expected_family {expected_family!r} on row {row_number}; "
                    f"choose one of {', '.join(FAMILIES)}."
                )
            rows.append({
                "filename": filename,
                "expected_category": expected_category,
                "expected_family": expected_family,
            })
    return rows


def resolve_image_path(images_root: Path, filename: str) -> Path:
    """Resolve a CSV image path without permitting traversal or symlink escape."""
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image path in label CSV: {filename!r}")
    root = images_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Image path resolves outside the image folder: {filename!r}") from error
    if not candidate.is_file():
        raise ValueError(f"Image path is not a file: {filename!r}")
    return candidate


def load_manifest_hashes(manifest_paths: tuple[Path, ...]) -> set[str]:
    """Collect hashes from known model data manifests for overlap warnings."""
    hashes = set()
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                digest = (row.get("sha256") or "").strip().lower()
                if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
                    hashes.add(digest)
    return hashes


def confusion_matrix(expected: list[str], predicted: list[str], labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Build a small dependency-free matrix, retaining Uncertain predictions."""
    columns = (*labels, "Uncertain")
    matrix = {label: {column: 0 for column in columns} for label in labels}
    for actual, result in zip(expected, predicted):
        matrix[actual][result if result in columns else "Uncertain"] += 1
    return matrix


def accuracy(expected: list[str], predicted: list[str]) -> float | None:
    if not expected:
        return None
    return sum(actual == result for actual, result in zip(expected, predicted)) / len(expected)


def evaluate(
    images_root: Path,
    expected_rows: list[dict[str, str]],
    family_classifier,
    category_classifier,
    training_manifests: tuple[Path, ...] = DEFAULT_TRAINING_MANIFESTS,
) -> dict[str, object]:
    """Evaluate images and compare predictions only with supplied expected labels."""
    known_hashes = load_manifest_hashes(training_manifests)
    records = []
    overlaps = []
    category_expected, category_predicted = [], []
    family_expected, family_predicted = [], []

    for row in expected_rows:
        image_path = resolve_image_path(images_root, row["filename"])
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        overlap = digest in known_hashes
        if overlap:
            overlaps.append(row["filename"])
        try:
            with Image.open(image_path) as source:
                source.load()
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            records.append({
                **row,
                "sha256": digest,
                "overlaps_existing_manifest": overlap,
                "error": f"Image could not be read: {error}",
            })
            continue

        family_result = family_classifier.predict(image)
        category_result = category_classifier.predict(image)
        family = family_result.get("family")
        family_name = getattr(family, "value", family)
        family_uncertain = bool(family_result.get("is_uncertain")) or family_name is None
        family_prediction = "Uncertain" if family_uncertain else family_name
        category_value = getattr(category_result.category, "value", category_result.category)
        category_candidate = getattr(
            category_result.candidate_category, "value", category_result.candidate_category
        )
        category_uncertain = bool(category_result.is_uncertain) or category_value is None
        category_final = "Uncertain" if category_uncertain else category_value
        record = {
            **row,
            "sha256": digest,
            "overlaps_existing_manifest": overlap,
            "provisional_family": family_prediction,
            "family_raw_top_score": family_result.get("score"),
            "family_uncertain": family_uncertain,
            "provisional_category": category_final,
            "category_top_candidate": category_candidate,
            "category_raw_top_score": category_result.score,
            "category_uncertain": category_uncertain,
        }
        records.append(record)
        if row["expected_family"]:
            family_expected.append(row["expected_family"])
            family_predicted.append(family_prediction)
        if row["expected_category"]:
            category_expected.append(row["expected_category"])
            category_predicted.append(category_final)

    return {
        "records": records,
        "overlap_files": overlaps,
        "family_metrics": {
            "labeled_count": len(family_expected),
            "accuracy": accuracy(family_expected, family_predicted),
            "confusion_matrix": confusion_matrix(family_expected, family_predicted, FAMILIES),
        } if family_expected else None,
        "category_metrics": {
            "labeled_count": len(category_expected),
            "accuracy": accuracy(category_expected, category_predicted),
            "confusion_matrix": confusion_matrix(category_expected, category_predicted, CATEGORIES),
        } if category_expected else None,
        "unlabeled_count": sum(
            not row["expected_family"] and not row["expected_category"] for row in expected_rows
        ),
        "unreadable_count": sum("error" in record for record in records),
    }


def _load_classifiers(family_checkpoint: Path, category_checkpoint: Path):
    if str(PROJECT_SRC) not in sys.path:
        sys.path.insert(0, str(PROJECT_SRC))
    from vista.classification.provisional_waste_category_classifier import (
        ProvisionalWasteCategoryClassifier,
    )
    from vista.classification.provisional_waste_family_classifier import (
        ProvisionalWasteFamilyClassifier,
    )

    return (
        ProvisionalWasteFamilyClassifier(checkpoint_path=family_checkpoint),
        ProvisionalWasteCategoryClassifier(checkpoint_path=category_checkpoint),
    )


def read_review_decisions(family_csv: Path, category_csv: Path) -> tuple[list[dict], list[dict]]:
    """Read explicit human decisions from the existing family and category files."""
    with family_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        family_rows = list(csv.DictReader(handle))
    with category_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        category_rows = list(csv.DictReader(handle))
    if family_rows and not {"source_relative_path", "review_status", "reviewed_family"}.issubset(family_rows[0]):
        raise ValueError("Family review CSV is missing its expected decision columns.")
    if category_rows and not {"source_relative_path", "decision", "reviewed_label"}.issubset(category_rows[0]):
        raise ValueError("Category review CSV is missing its expected decision columns.")
    return family_rows, category_rows


def read_test_batch_decisions(
    decisions_csv: Path, queue_csv: Path
) -> tuple[list[dict], list[dict]]:
    """Adapt explicit Pilot Test Review decisions after validating their queue hashes."""
    with queue_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        queue_reader = csv.DictReader(handle)
        if not {"source_relative_path", "sha256"}.issubset(set(queue_reader.fieldnames or ())):
            raise ValueError("Test review queue must contain source_relative_path and sha256 columns.")
        queue_rows = list(queue_reader)
    queue_by_path = {}
    for row in queue_rows:
        path = (row.get("source_relative_path") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        if not path or not _valid_sha256(digest):
            raise ValueError("Test review queue contains a blank path or invalid SHA-256.")
        if path in queue_by_path:
            raise ValueError(f"Test review queue contains duplicate path {path!r}.")
        queue_by_path[path] = digest

    with decisions_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_relative_path", "sha256", "review_type", "decision", "reviewed_label", "reason"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("Test review decisions CSV is missing required decision columns.")
        rows = list(reader)

    adapted = {"family": [], "category": []}
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        path = (row.get("source_relative_path") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        task = (row.get("review_type") or "").strip().lower()
        decision = (row.get("decision") or "").strip().lower()
        label = (row.get("reviewed_label") or "").strip()
        reason = (row.get("reason") or "").strip()
        if not path:
            raise ValueError(f"Test review decision row {row_number} has a blank source-relative path.")
        if task not in adapted:
            raise ValueError(f"Invalid review_type {task!r} on test review decision row {row_number}.")
        if decision not in {"include", "hold", "exclude"}:
            raise ValueError(
                f"Invalid or non-persistable decision {decision!r} on row {row_number}; "
                "only include, hold, and exclude decisions may be evaluated."
            )
        if not _valid_sha256(digest):
            raise ValueError(f"Test review decision row {row_number} has an invalid SHA-256.")
        if queue_by_path.get(path) != digest:
            raise ValueError(
                f"Test review decision path/hash does not match the saved review queue: {path}"
            )
        key = (task, path)
        if key in seen:
            raise ValueError(f"Duplicate {task} decision row for {path!r}.")
        seen.add(key)
        if decision in {"hold", "exclude"} and not reason:
            raise ValueError(f"{decision.title()} decision requires a reason for {path!r}.")
        labels = FAMILIES if task == "family" else CATEGORIES
        if decision == "include" and label not in labels:
            raise ValueError(
                f"Included {task} decision for {path!r} has an invalid or missing label {label!r}."
            )
        if decision != "include" and label and label not in labels:
            raise ValueError(f"Invalid reviewed label {label!r} for {path!r}.")

        if task == "family":
            adapted[task].append({
                "source_relative_path": path,
                "review_status": decision,
                "reviewed_family": label,
                "reason": reason,
                "_review_sha256": digest,
            })
        else:
            adapted[task].append({
                "source_relative_path": path,
                "decision": decision,
                "reviewed_label": label,
                "reason": reason,
                "_review_sha256": digest,
            })
    return adapted["family"], adapted["category"]


def merge_review_rows(existing: list[dict], batch: list[dict], task: str) -> list[dict]:
    """Merge batch decisions in memory; explicit decisions may replace only pending rows."""
    status_field = "review_status" if task == "family" else "decision"
    label_field = "reviewed_family" if task == "family" else "reviewed_label"
    merged = {}
    for row in existing:
        path = (row.get("source_relative_path") or "").strip()
        if path:
            merged[path] = row
    for row in batch:
        path = row["source_relative_path"]
        previous = merged.get(path)
        if previous:
            old_status = (previous.get(status_field) or "").strip().lower()
            old_label = (previous.get(label_field) or "").strip()
            new_status = row[status_field]
            new_label = row[label_field]
            if old_status not in {"", "pending", "unreviewed"} and (
                old_status != new_status or old_label != new_label
            ):
                raise ValueError(
                    f"Conflicting explicit {task} review decisions exist for {path!r}; "
                    "resolve the conflict before evaluation."
                )
        merged[path] = row
    return list(merged.values())


def merge_test_batch_inputs(
    family_decisions: list[dict],
    category_decisions: list[dict],
    batch_inputs: list[tuple[Path, Path]],
) -> tuple[list[dict], list[dict]]:
    """Read and merge any number of separate test-review batches in memory."""
    for decisions_path, queue_path in batch_inputs:
        batch_family, batch_category = read_test_batch_decisions(decisions_path, queue_path)
        family_decisions = merge_review_rows(family_decisions, batch_family, "family")
        category_decisions = merge_review_rows(category_decisions, batch_category, "category")
    return family_decisions, category_decisions


def read_manifest(path: Path) -> list[dict[str, str]]:
    required = {"source_relative_path", "sha256", "canonical_path", "split", "group_status"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError(f"Manifest {path} is missing required path/hash/split columns.")
        return list(reader)


def _valid_sha256(value: str) -> bool:
    value = (value or "").lower()
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def collect_reviewed_test_cases(
    dataset_root: Path,
    family_decisions: list[dict],
    category_decisions: list[dict],
    family_manifest: list[dict],
    category_manifest: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Match reviewed include decisions to canonical test rows and reject leakage.

    Only the decision CSV's explicit reviewed label is used as expected truth.
    Folder-derived manifest labels are deliberately ignored.
    """
    all_manifests = (family_manifest, category_manifest)
    train_validation_hashes = {
        row["sha256"].lower()
        for manifest in all_manifests
        for row in manifest
        if row.get("split") in {"train", "validation"} and _valid_sha256(row.get("sha256", ""))
    }
    manifest_by_task = {"family": family_manifest, "category": category_manifest}
    task_rows = {"family": family_decisions, "category": category_decisions}
    included_candidates: list[dict] = []
    excluded: list[dict] = []

    for task, decision_rows in task_rows.items():
        manifest = manifest_by_task[task]
        manifest_by_path = defaultdict(list)
        test_hash_paths = defaultdict(set)
        for row in manifest:
            path = row.get("source_relative_path", "")
            manifest_by_path[path].append(row)
            if row.get("split") == "test" and row.get("canonical_path") == path and _valid_sha256(row.get("sha256", "")):
                test_hash_paths[row["sha256"].lower()].add(path)
        duplicate_test_hashes = {
            digest for digest, paths in test_hash_paths.items() if len(paths) > 1
        }

        seen_paths = set()
        for decision in decision_rows:
            path = (decision.get("source_relative_path") or "").strip()
            raw_status = decision.get("review_status" if task == "family" else "decision") or ""
            status = raw_status.strip().lower()
            label_field = "reviewed_family" if task == "family" else "reviewed_label"
            expected_label = (decision.get(label_field) or "").strip()
            valid_labels = FAMILIES if task == "family" else CATEGORIES
            base = {"task": task, "source_relative_path": path, "decision": status}

            if not path:
                excluded.append({**base, "reason": "blank source-relative path"})
                continue
            if path in seen_paths:
                excluded.append({**base, "reason": "duplicate decision row for this path"})
                continue
            seen_paths.add(path)
            if status != "include":
                excluded.append({**base, "reason": f"decision is {status or 'blank'}, not include"})
                continue
            if expected_label not in valid_labels:
                excluded.append({**base, "reason": "include decision has a missing or invalid reviewed label"})
                continue

            rows = manifest_by_path.get(path, [])
            if not rows:
                excluded.append({**base, "reason": "path is not present in the model manifest"})
                continue
            if len(rows) != 1:
                excluded.append({**base, "reason": "manifest contains an ambiguous repeated path"})
                continue
            manifest_row = rows[0]
            if manifest_row.get("split") != "test":
                excluded.append({**base, "reason": f"path is assigned to {manifest_row.get('split') or 'no'} split, not held-out test"})
                continue
            if manifest_row.get("canonical_path") != path:
                excluded.append({**base, "reason": "path is not the canonical duplicate-group member"})
                continue
            if "include" not in (manifest_row.get("group_status") or "").lower():
                excluded.append({**base, "reason": "manifest does not mark this canonical group as included"})
                continue
            digest = (manifest_row.get("sha256") or "").lower()
            if not _valid_sha256(digest):
                excluded.append({**base, "reason": "manifest SHA-256 is missing or invalid"})
                continue
            reviewed_digest = (decision.get("_review_sha256") or "").lower()
            if reviewed_digest and reviewed_digest != digest:
                excluded.append({**base, "sha256": digest, "reason": "review decision SHA-256 differs from held-out test manifest"})
                continue
            if digest in duplicate_test_hashes:
                excluded.append({**base, "sha256": digest, "reason": "duplicate hash occurs at multiple canonical test paths"})
                continue
            if digest in train_validation_hashes:
                excluded.append({**base, "sha256": digest, "reason": "hash also appears in training or validation data"})
                continue
            try:
                image_path = resolve_image_path(dataset_root, path)
                current_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            except (OSError, ValueError) as error:
                excluded.append({**base, "sha256": digest, "reason": f"source path unavailable or unsafe: {error}"})
                continue
            if current_hash != digest:
                excluded.append({**base, "sha256": digest, "reason": "current source SHA-256 differs from saved test manifest"})
                continue
            candidate = {
                **base,
                "sha256": digest,
                "expected_label": expected_label,
                "image_path": image_path,
            }
            included_candidates.append(candidate)

    # A hash may enter the review through both CSVs (same image, separate task),
    # which is fine. Distinct paths sharing a hash are duplicate evaluation items.
    paths_by_hash = defaultdict(set)
    for candidate in included_candidates:
        paths_by_hash[candidate["sha256"]].add(candidate["source_relative_path"])
    duplicate_hashes = {
        digest for digest, paths in paths_by_hash.items() if len(paths) > 1
    }
    for candidate in included_candidates:
        if candidate["sha256"] in duplicate_hashes:
            excluded.append({
                "task": candidate["task"],
                "source_relative_path": candidate["source_relative_path"],
                "decision": candidate["decision"],
                "sha256": candidate["sha256"],
                "reason": "duplicate image hash is represented by another reviewed test path",
            })
    included_candidates = [row for row in included_candidates if row["sha256"] not in duplicate_hashes]
    return included_candidates, excluded


def _class_metrics(expected: list[str], predicted: list[str], labels: tuple[str, ...]) -> dict:
    matrix = confusion_matrix(expected, predicted, labels)
    per_class = {}
    f1_values = []
    for label in labels:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in labels)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            f1_values.append(f1)
    support_warning = any(
        per_class[label]["support"] < MINIMUM_CLASS_SUPPORT_WARNING for label in labels
    )
    return {
        "independently_reviewed_examples": len(expected),
        "accuracy": accuracy(expected, predicted),
        "macro_f1_over_supported_classes": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "reliability_warning": (
            f"At least one class has fewer than {MINIMUM_CLASS_SUPPORT_WARNING} independently reviewed examples; "
            "these results are too small for reliable conclusions."
            if support_warning else None
        ),
    }


def evaluate_reviewed_test_cases(
    cases: list[dict], family_classifier, category_classifier
) -> dict:
    """Run provisional models only on validated, independently reviewed test cases."""
    predictions = []
    evaluation_excluded = []
    family_expected, family_predicted = [], []
    category_expected, category_predicted = [], []
    for case in cases:
        try:
            with Image.open(case["image_path"]) as source:
                source.load()
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            unreadable = {
                "task": case["task"],
                "source_relative_path": case["source_relative_path"],
                "expected_label": case["expected_label"],
                "sha256": case["sha256"],
                "excluded_reason": f"image is unreadable: {error}",
            }
            predictions.append(unreadable)
            evaluation_excluded.append(unreadable)
            continue

        if case["task"] == "family":
            result = family_classifier.predict(image)
            family = result.get("family")
            family_name = getattr(family, "value", family)
            uncertain = bool(result.get("is_uncertain")) or family_name is None
            predicted = "Uncertain" if uncertain else family_name
            family_expected.append(case["expected_label"])
            family_predicted.append(predicted)
            predictions.append({
                "task": "family",
                "source_relative_path": case["source_relative_path"],
                "sha256": case["sha256"],
                "expected_label": case["expected_label"],
                "provisional_prediction": predicted,
                "raw_top_class_score": result.get("score"),
                "uncertain": uncertain,
                "label_provenance": "explicit human-reviewed family decision",
            })
        else:
            result = category_classifier.predict(image)
            category = getattr(result.category, "value", result.category)
            candidate = getattr(result.candidate_category, "value", result.candidate_category)
            uncertain = bool(result.is_uncertain) or category is None
            predicted = "Uncertain" if uncertain else category
            category_expected.append(case["expected_label"])
            category_predicted.append(predicted)
            predictions.append({
                "task": "category",
                "source_relative_path": case["source_relative_path"],
                "sha256": case["sha256"],
                "expected_label": case["expected_label"],
                "provisional_prediction": predicted,
                "top_candidate": candidate,
                "raw_top_class_score": result.score,
                "uncertain": uncertain,
                "label_provenance": "explicit human-reviewed category decision",
            })

    return {
        "evaluation_status": "PROVISIONAL models evaluated only against explicit human-reviewed labels in held-out test rows",
        "verified_performance_claim": False,
        "predictions": predictions,
        "uncertain_predictions": [row for row in predictions if row.get("uncertain")],
        "family_metrics": _class_metrics(family_expected, family_predicted, FAMILIES) if family_expected else None,
        "category_metrics": _class_metrics(category_expected, category_predicted, CATEGORIES) if category_expected else None,
        "independently_reviewed_family_examples": len(family_expected),
        "independently_reviewed_category_examples": len(category_expected),
        "excluded_records": evaluation_excluded,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset"))
    parser.add_argument("--family-decisions", type=Path, default=DEFAULT_FAMILY_REVIEW)
    parser.add_argument("--category-decisions", type=Path, default=DEFAULT_CATEGORY_REVIEW)
    parser.add_argument("--test-batch-decisions", type=Path, help="Optional separate Pilot Test Review decisions CSV.")
    parser.add_argument("--test-batch-queue", type=Path, help="Queue CSV paired with --test-batch-decisions.")
    parser.add_argument("--followup-test-batch-decisions", type=Path, help="Optional follow-up Pilot Test Review decisions CSV.")
    parser.add_argument("--followup-test-batch-queue", type=Path, help="Queue CSV paired with --followup-test-batch-decisions.")
    parser.add_argument(
        "--additional-test-batch", nargs=2, action="append", type=Path, default=[],
        metavar=("DECISIONS_CSV", "QUEUE_CSV"),
        help="Another decisions/queue CSV pair; may be repeated.",
    )
    parser.add_argument("--family-test-manifest", type=Path, default=DEFAULT_FAMILY_MANIFEST)
    parser.add_argument("--category-test-manifest", type=Path, default=DEFAULT_CATEGORY_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New JSON report path; existing files are never overwritten.")
    parser.add_argument("--family-checkpoint", type=Path, default=DEFAULT_FAMILY_CHECKPOINT)
    parser.add_argument("--category-checkpoint", type=Path, default=DEFAULT_CATEGORY_CHECKPOINT)
    args = parser.parse_args(argv)
    if bool(args.test_batch_decisions) != bool(args.test_batch_queue):
        parser.error("--test-batch-decisions and --test-batch-queue must be supplied together.")
    if bool(args.followup_test_batch_decisions) != bool(args.followup_test_batch_queue):
        parser.error("--followup-test-batch-decisions and --followup-test-batch-queue must be supplied together.")
    if args.output.exists():
        parser.error(f"report already exists; choose a new --output path: {args.output}")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist or is not a directory: {args.dataset_root}")
    family_decisions, category_decisions = read_review_decisions(
        args.family_decisions, args.category_decisions
    )
    batch_inputs = [
        (args.test_batch_decisions, args.test_batch_queue),
        (args.followup_test_batch_decisions, args.followup_test_batch_queue),
    ]
    batch_inputs = [(decisions, queue) for decisions, queue in batch_inputs if decisions]
    batch_inputs.extend(tuple(pair) for pair in args.additional_test_batch)
    family_decisions, category_decisions = merge_test_batch_inputs(
        family_decisions, category_decisions, batch_inputs
    )
    family_manifest = read_manifest(args.family_test_manifest)
    category_manifest = read_manifest(args.category_test_manifest)
    cases, excluded = collect_reviewed_test_cases(
        args.dataset_root, family_decisions, category_decisions,
        family_manifest, category_manifest,
    )
    results = evaluate_reviewed_test_cases(
        cases,
        *_load_classifiers(args.family_checkpoint, args.category_checkpoint),
    ) if cases else {
        "evaluation_status": "No eligible independently reviewed held-out test examples.",
        "verified_performance_claim": False,
        "predictions": [], "uncertain_predictions": [],
        "family_metrics": None, "category_metrics": None,
        "independently_reviewed_family_examples": 0,
        "independently_reviewed_category_examples": 0,
    }
    results["excluded_records"] = excluded + results.get("excluded_records", [])
    results["excluded_record_count"] = len(results["excluded_records"])
    results["too_small_for_reliable_conclusions"] = any(
        metrics and metrics["reliability_warning"]
        for metrics in (results["family_metrics"], results["category_metrics"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved provisional reviewed-test evaluation report: {args.output.resolve()}")
    print(f"Eligible family examples: {results['independently_reviewed_family_examples']}")
    print(f"Eligible category examples: {results['independently_reviewed_category_examples']}")
    print(f"Excluded review records: {results['excluded_record_count']}")
    if results["too_small_for_reliable_conclusions"]:
        print("WARNING: independently reviewed sample is too small for reliable conclusions.")
    return 0


def _format_score(score) -> str:
    return f"{score:.1%}" if isinstance(score, (int, float)) else "unavailable"


def _print_metrics(title: str, metrics) -> None:
    if metrics is None:
        print(f"{title} metrics: not calculated (no independent expected labels supplied).")
        return
    print(f"{title} accuracy (Uncertain counts as incorrect): {metrics['accuracy']:.3f} n={metrics['labeled_count']}")
    print(f"{title} confusion matrix (rows=expected, columns=predicted): {json.dumps(metrics['confusion_matrix'])}")


if __name__ == "__main__":
    raise SystemExit(main())
