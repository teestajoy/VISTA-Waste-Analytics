"""Run a validation-only, versioned MSW augmentation experiment.

This standalone runner deliberately never constructs test records, opens test
images, or writes test metrics. It reuses the existing category train/validation
membership and keeps the production classifier and checkpoint untouched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import prepare_provisional_family as family_data  # noqa: E402
import train_waste_category_provisional as category_training  # noqa: E402
import train_waste_family_provisional as family_training  # noqa: E402

BASELINE_DIR = PROJECT_ROOT / "models" / "waste_category_provisional"
MODELS_DIR = PROJECT_ROOT / "models"
# v1 is preserved as a completed artifact. Future runs default to a new version.
OUTPUT_DIR = MODELS_DIR / "waste_category_augmentation_v2"
LEGACY_V1_OUTPUT_DIR = MODELS_DIR / "waste_category_augmentation_v1"
CATEGORY_MANIFEST = BASELINE_DIR / "split_manifest.csv"
SOURCE_MANIFEST = PROJECT_ROOT / "data" / "processed" / "family_provisional_split_manifest.csv"
PENDING_REVIEW = PROJECT_ROOT / "docs" / "category_disagreement_re_review_v1.csv"
CLASS_NAMES = category_training.CLASS_NAMES
CATEGORY_TO_FAMILY = category_training.CATEGORY_TO_FAMILY
ARCHITECTURE = "torchvision ResNet18; frozen backbone; trained linear category head"
CLASS_COUNT = len(CLASS_NAMES)
TRAIN_VALIDATION = ("train", "validation")
SEED = 42
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
THRESHOLD = 0.70
AUGMENTATION = {
    "random_resized_crop": {"size": [224, 224], "scale": [0.90, 1.0], "ratio": [0.95, 1.05]},
    "brightness": 0.10,
    "contrast": 0.10,
    "color_jitter_probability": 0.80,
    "train_only": True,
}


def read_train_validation_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Read only canonical train/validation rows from the existing category manifest."""
    selected: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "source_relative_path", "sha256", "provisional_category", "group_status",
            "canonical_path", "split", "label_provenance", "source_folder",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Category manifest missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            # Ignore test and excluded rows before inspecting their labels or paths.
            split = (row.get("split") or "").strip()
            if split not in TRAIN_VALIDATION:
                continue
            relative = (row.get("source_relative_path") or "").strip().replace("\\", "/")
            digest = (row.get("sha256") or "").strip().lower()
            category = (row.get("provisional_category") or "").strip()
            canonical = (row.get("canonical_path") or "").strip().replace("\\", "/")
            if row.get("group_status") != "provisional_include" or canonical != relative:
                raise ValueError(f"Noncanonical included {split} row: {relative!r}.")
            if category not in CLASS_NAMES or row.get("source_folder") != category:
                raise ValueError(f"Unsupported category/source folder for {relative!r}.")
            if row.get("label_provenance") != category_training.PROVENANCE:
                raise ValueError(f"Unexpected label provenance for {relative!r}.")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid SHA-256 for {relative!r}.")
            if not relative or relative in seen_paths or digest in seen_hashes:
                raise ValueError(f"Duplicate or blank train/validation path/hash: {relative!r}.")
            seen_paths.add(relative)
            seen_hashes.add(digest)
            selected.append({
                "source_relative_path": relative,
                "sha256": digest,
                "source_folder": category,
                "provisional_category": category,
                "label_provenance": row["label_provenance"],
                "duplicate_group_id": row.get("duplicate_group_id", ""),
                "canonical_path": canonical,
                "group_status": row["group_status"],
                "split": split,
                "exclusion_reason": row.get("exclusion_reason", ""),
            })
    counts = Counter((row["split"], row["provisional_category"]) for row in selected)
    missing_counts = [f"{s}/{c}" for s in TRAIN_VALIDATION for c in CLASS_NAMES if not counts[s, c]]
    if missing_counts:
        raise ValueError("Train/validation manifest is missing classes: " + ", ".join(missing_counts))
    return selected


def verify_existing_membership(rows: list[dict[str, str]], source_manifest: Path, metadata: dict[str, Any]) -> None:
    """Cross-check the category membership against its recorded source, without test images."""
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source family manifest not found: {source_manifest}")
    source_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    if source_hash != metadata.get("source_manifest_sha256"):
        raise ValueError("Source family manifest hash differs from the baseline training metadata.")

    expected: set[tuple[str, str, str, str]] = set()
    source_rows: list[dict[str, str]] = []
    with source_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            # Rows outside the selected splits are not materialized or validated here.
            if (row.get("split") or "").strip() not in TRAIN_VALIDATION:
                continue
            folder = (row.get("source_folder") or "").strip()
            relative = (row.get("source_relative_path") or "").strip().replace("\\", "/")
            canonical = (row.get("canonical_path") or "").strip().replace("\\", "/")
            if (folder in CLASS_NAMES and row.get("group_status") == "provisional_include"
                    and relative and canonical == relative):
                source_rows.append(row)
    all_folders_by_hash: dict[str, set[str]] = {}
    for row in source_rows:
        digest = (row.get("sha256") or "").strip().lower()
        if digest:
            all_folders_by_hash.setdefault(digest, set()).add(row["source_folder"])
    conflicts = {digest for digest, folders in all_folders_by_hash.items() if len(folders) > 1}
    for row in source_rows:
        digest = (row.get("sha256") or "").strip().lower()
        if digest not in conflicts:
            expected.add((
                row["source_relative_path"].replace("\\", "/"), digest,
                row["source_folder"], row["split"],
            ))
    actual = {
        (row["source_relative_path"], row["sha256"], row["provisional_category"], row["split"])
        for row in rows
    }
    if actual != expected:
        raise ValueError(
            "Existing category train/validation membership differs from the source manifest "
            f"(category rows={len(actual)}, source projection={len(expected)})."
        )


def verify_pending_reviews(rows: list[dict[str, str]], pending_path: Path) -> None:
    if not pending_path.is_file():
        raise FileNotFoundError(f"Pending disagreement worksheet not found: {pending_path}")
    selected = {row["source_relative_path"] for row in rows}
    with pending_path.open("r", encoding="utf-8-sig", newline="") as stream:
        pending = {row["source_relative_path"].replace("\\", "/") for row in csv.DictReader(stream)
                   if (row.get("review_status") or "").strip().lower() == "pending"}
    overlap = selected & pending
    if overlap:
        raise ValueError("Pending review cases appeared in train/validation: " + ", ".join(sorted(overlap)))


def rows_to_records(rows: list[dict[str, str]]) -> list[category_training.CategoryImage]:
    return [category_training.CategoryImage(
        row["source_relative_path"], row["sha256"], row["provisional_category"], row["split"],
        row["source_folder"],
    ) for row in rows]


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            (224, 224), scale=(0.90, 1.0), ratio=(0.95, 1.05),
        ),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
        ], p=0.80),
        transforms.ToTensor(),
        transforms.Normalize(
            family_training.reviewed_trainer.IMAGENET_MEAN,
            family_training.reviewed_trainer.IMAGENET_STD,
        ),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            family_training.reviewed_trainer.IMAGENET_MEAN,
            family_training.reviewed_trainer.IMAGENET_STD,
        ),
    ])
    return train_transform, validation_transform


def _metrics_from_confusion(confusion: np.ndarray, supports: list[int] | None = None) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, category in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(supports[index] if supports is not None else confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[category] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        f1_values.append(f1)
    total = int(sum(supports) if supports is not None else confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def metrics_from_predictions(
    targets: list[str], predictions: list[str], confidences: list[float], threshold: float = THRESHOLD,
) -> dict[str, Any]:
    if not targets or len(targets) != len(predictions) or len(targets) != len(confidences):
        raise ValueError("Targets, predictions, and confidences must have the same nonzero length.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be in (0, 1).")
    if any(value not in CLASS_NAMES for value in [*targets, *predictions]):
        raise ValueError("Prediction rows contain an unsupported class name.")
    target_array = np.asarray([CLASS_NAMES.index(value) for value in targets], dtype=np.int64)
    top = np.asarray([CLASS_NAMES.index(value) for value in predictions], dtype=np.int64)
    confidence = np.asarray(confidences, dtype=np.float64)
    if not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError("Confidences must be finite values between zero and one.")
    supports = [int(np.sum(target_array == i)) for i in range(CLASS_COUNT)]
    top_confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual, predicted in zip(target_array, top):
        top_confusion[actual, predicted] += 1
    accepted = confidence >= threshold
    operating_confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES) + 1), dtype=np.int64)
    for actual, predicted, is_accepted in zip(target_array, top, accepted):
        operating_confusion[actual, predicted if is_accepted else len(CLASS_NAMES)] += 1
    accepted_count = int(accepted.sum())
    accepted_correct = int(np.sum((top == target_array) & accepted))
    glass, plastic = CLASS_NAMES.index("glass"), CLASS_NAMES.index("plastic")
    return {
        "examples": len(target_array),
        "top1": {
            **_metrics_from_confusion(top_confusion),
            "confusion_matrix": {
                actual: {predicted: int(top_confusion[i, j]) for j, predicted in enumerate(CLASS_NAMES)}
                for i, actual in enumerate(CLASS_NAMES)
            },
        },
        "operating_point": {
            "threshold": threshold,
            "coverage": accepted_count / len(target_array),
            "accepted": accepted_count,
            "uncertain": len(target_array) - accepted_count,
            "accepted_accuracy": accepted_correct / accepted_count if accepted_count else 0.0,
            "accuracy_counting_uncertain_as_wrong": accepted_correct / len(target_array),
            "per_class": _metrics_from_confusion(operating_confusion[:, :len(CLASS_NAMES)], supports)["per_class"],
            "confusion_matrix": {
                actual: {
                    **{predicted: int(operating_confusion[i, j]) for j, predicted in enumerate(CLASS_NAMES)},
                    "Uncertain": int(operating_confusion[i, len(CLASS_NAMES)]),
                }
                for i, actual in enumerate(CLASS_NAMES)
            },
        },
        "glass_to_plastic": {
            "top1_count": int(top_confusion[glass, plastic]),
            "accepted_count": int(operating_confusion[glass, plastic]),
            "glass_support": supports[glass],
        },
        "plastic_to_glass": {
            "top1_count": int(top_confusion[plastic, glass]),
            "accepted_count": int(operating_confusion[plastic, glass]),
            "plastic_support": supports[plastic],
        },
    }


def metrics_from_prediction_rows(rows: list[dict[str, Any]], threshold: float = THRESHOLD) -> dict[str, Any]:
    """Recompute saved validation metrics using only the prediction CSV fields."""
    return metrics_from_predictions(
        [row["reviewed_label"] for row in rows],
        [row["top_candidate"] for row in rows],
        [float(row["confidence"]) for row in rows],
        threshold,
    )


def metrics_from_probabilities(targets: list[int], probabilities: list[list[float]], threshold: float = THRESHOLD) -> dict[str, Any]:
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.shape != (len(target_array), CLASS_COUNT) or not len(target_array):
        raise ValueError("Targets/probabilities have an invalid shape or are empty.")
    if not np.isfinite(probability_array).all():
        raise ValueError("Probabilities must be finite.")
    if np.any(target_array < 0) or np.any(target_array >= CLASS_COUNT):
        raise ValueError("Target indices contain an unsupported class.")
    top = probability_array.argmax(axis=1)
    return metrics_from_predictions(
        [CLASS_NAMES[int(value)] for value in target_array],
        [CLASS_NAMES[int(value)] for value in top],
        [float(value) for value in probability_array.max(axis=1)],
        threshold,
    )


def validate_prediction_pair(
    baseline_rows: list[dict[str, Any]], augmented_rows: list[dict[str, Any]],
) -> None:
    """Require exact row-by-row validation cohort and label identity."""
    fields = ("source_relative_path", "reviewed_label", "split")
    if len(baseline_rows) != len(augmented_rows) or not baseline_rows:
        raise ValueError("Baseline and augmented prediction files have different or empty row counts.")
    for index, (base_row, augmented_row) in enumerate(zip(baseline_rows, augmented_rows), start=1):
        if any(base_row.get(field) != augmented_row.get(field) for field in fields):
            raise ValueError(f"Baseline/augmented validation row mismatch at row {index}.")
        if base_row.get("split") != "validation":
            raise ValueError(f"Prediction row {index} is not from the validation split.")
        if base_row.get("reviewed_label") not in CLASS_NAMES:
            raise ValueError(f"Unsupported reviewed label in prediction row {index}.")
    base_paths = [row["source_relative_path"] for row in baseline_rows]
    augmented_paths = [row["source_relative_path"] for row in augmented_rows]
    if len(set(base_paths)) != len(base_paths) or len(set(augmented_paths)) != len(augmented_paths):
        raise ValueError("Prediction files contain duplicate validation paths.")


def validate_checkpoint_metadata(checkpoint: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Fail if a new checkpoint and its sidecar disagree on inference/training identity."""
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    pairs = (
        (checkpoint.get("architecture"), metadata.get("architecture"), "architecture"),
        (checkpoint.get("class_count"), metadata.get("class_count"), "class count"),
        (list(checkpoint.get("class_names", ())), metadata.get("class_names"), "class order"),
        (checkpoint.get("class_to_index"), metadata.get("class_to_index"), "class mapping"),
        (checkpoint.get("seed"), metadata.get("seed"), "seed"),
        (checkpoint.get("confidence_threshold"), metadata.get("confidence_threshold"), "confidence threshold"),
        (checkpoint.get("input_preprocessing"), metadata.get("input_preprocessing"), "input preprocessing"),
        (checkpoint.get("pretrained_weights"), metadata.get("pretrained_weights"), "pretrained weights"),
        (checkpoint.get("augmentation"), metadata.get("augmentation"), "augmentation policy"),
    )
    for checkpoint_value, metadata_value, label in pairs:
        if checkpoint_value != metadata_value:
            raise ValueError(f"Checkpoint and metadata disagree on {label}.")
    if (metadata.get("architecture") != ARCHITECTURE
            or metadata.get("class_count") != CLASS_COUNT
            or metadata.get("class_names") != list(CLASS_NAMES)
            or list(checkpoint.get("class_names", ())) != list(CLASS_NAMES)
            or metadata.get("class_to_index") != expected_mapping
            or metadata.get("seed") != SEED
            or metadata.get("confidence_threshold") != THRESHOLD
            or metadata.get("input_preprocessing") != category_training.PREPROCESSING):
        raise ValueError("Checkpoint metadata does not match the experiment class definition.")
    state = checkpoint.get("state_dict", {})
    if tuple(state.get("fc.weight", torch.empty(0)).shape) != (CLASS_COUNT, 512):
        raise ValueError("Checkpoint linear head shape does not match its class count/ResNet18 backbone.")
    if tuple(state.get("fc.bias", torch.empty(0)).shape) != (CLASS_COUNT,):
        raise ValueError("Checkpoint linear head bias shape does not match its class count.")
    weight_hash = checkpoint.get("pretrained_weights", {}).get("sha256", "")
    if len(weight_hash) != 64 or any(char not in "0123456789abcdef" for char in weight_hash.lower()):
        raise ValueError("Checkpoint pretrained-weight SHA-256 is missing or malformed.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(model, loader, device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    targets: list[int] = []
    probabilities: list[list[float]] = []
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for target, row in zip(labels.tolist(), probs.tolist()):
                targets.append(int(target))
                probabilities.append([float(value) for value in row])
    metrics = metrics_from_probabilities(targets, probabilities)
    top = np.asarray(probabilities).argmax(axis=1)
    confidence = np.asarray(probabilities).max(axis=1)
    # The validation loader is sequential and records are supplied in that same order.
    records = loader.dataset.records
    for record, actual, predicted, score in zip(records, targets, top, confidence):
        predictions.append({
            "source_relative_path": record.source_relative_path,
            "reviewed_label": record.category,
            "top_candidate": CLASS_NAMES[int(predicted)],
            "confidence": float(score),
            "uncertain_at_0_70": bool(score < THRESHOLD),
            "split": record.split,
        })
    return metrics, predictions


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["source_relative_path", "reviewed_label", "top_candidate", "confidence", "uncertain_at_0_70", "split"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_prediction_csv(path: Path) -> list[dict[str, str]]:
    required = {"source_relative_path", "reviewed_label", "top_candidate", "confidence", "uncertain_at_0_70", "split"}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Prediction CSV missing columns: " + ", ".join(sorted(missing)))
        return [dict(row) for row in reader]


def resolve_output_path(path: Path) -> Path:
    """Resolve a fresh output directory constrained to the models directory."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Experiment output already exists; refusing to overwrite: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(MODELS_DIR.resolve())
    except ValueError as error:
        raise ValueError(f"Experiment outputs must be under {MODELS_DIR}.") from error
    if resolved in {BASELINE_DIR.resolve(), LEGACY_V1_OUTPUT_DIR.resolve()}:
        raise ValueError(f"Refusing to use a protected existing artifact directory: {resolved}")
    if not resolved.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {resolved.parent}")
    return resolved


def create_output_directory(path: Path) -> Path:
    """Create an experiment directory exclusively; existing paths are never reused."""
    resolved = resolve_output_path(path)
    resolved.mkdir(parents=False, exist_ok=False)
    return resolved


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir = resolve_output_path(output_dir)
    for required_path in (CATEGORY_MANIFEST, SOURCE_MANIFEST, PENDING_REVIEW,
                          BASELINE_DIR / "training_metadata.json", BASELINE_DIR / "class_mapping.json",
                          BASELINE_DIR / "best_model.pth"):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required baseline/split artifact not found: {required_path}")

    metadata = json.loads((BASELINE_DIR / "training_metadata.json").read_text(encoding="utf-8"))
    mapping = json.loads((BASELINE_DIR / "class_mapping.json").read_text(encoding="utf-8"))
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if metadata.get("class_to_index") != expected_mapping or mapping.get("class_to_index") != expected_mapping:
        raise ValueError("Baseline class mapping differs from the current six-class mapping.")
    if metadata.get("class_names") != list(CLASS_NAMES) or metadata.get("confidence_threshold") != THRESHOLD:
        raise ValueError("Baseline metadata class names or confidence threshold differ from this experiment.")
    if Path(metadata.get("source_manifest", "")).resolve() != SOURCE_MANIFEST.resolve():
        raise ValueError("Baseline metadata points to an unexpected source split manifest.")

    rows = read_train_validation_rows(CATEGORY_MANIFEST)
    verify_existing_membership(rows, SOURCE_MANIFEST, metadata)
    verify_pending_reviews(rows, PENDING_REVIEW)
    train_records = rows_to_records([r for r in rows if r["split"] == "train"])
    validation_records = rows_to_records([r for r in rows if r["split"] == "validation"])
    train_counts = Counter(record.category for record in train_records)
    validation_counts = Counter(record.category for record in validation_records)
    actual_counts = {
        "train": {c: train_counts[c] for c in CLASS_NAMES},
        "validation": {c: validation_counts[c] for c in CLASS_NAMES},
    }
    expected_counts = {s: metadata["split_counts_by_category"][s] for s in TRAIN_VALIDATION}
    if actual_counts != expected_counts:
        raise ValueError(f"Train/validation counts differ from baseline metadata: {actual_counts} != {expected_counts}")

    dataset_root = family_data.workflow.DATASET_DIR
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Existing dataset root not found: {dataset_root}")
    train_paths = {r.source_relative_path: category_training._safe_image_path(dataset_root, r) for r in train_records}
    validation_paths = {r.source_relative_path: category_training._safe_image_path(dataset_root, r) for r in validation_records}

    base_transform = build_transforms()[1]
    validation_loader = DataLoader(
        category_training.CategoryImageDataset(validation_records, validation_paths, base_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = family_training.reviewed_trainer.resolve_cached_weights()
    pretrained_weights = {
        "path": str(weights_path.resolve()),
        "sha256": sha256_file(weights_path),
        "hash_algorithm": "SHA-256",
    }
    baseline_model = family_training.reviewed_trainer.build_model(weights_path)
    baseline_model.fc = nn.Linear(baseline_model.fc.in_features, len(CLASS_NAMES))
    baseline_checkpoint = torch.load(BASELINE_DIR / "best_model.pth", map_location="cpu", weights_only=True)
    if tuple(baseline_checkpoint.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("Baseline checkpoint class names do not match its mapping.")
    baseline_model.load_state_dict(baseline_checkpoint["state_dict"], strict=True)
    baseline_model.to(device)
    baseline_metrics, baseline_predictions = evaluate(baseline_model, validation_loader, device)

    # Create this fresh versioned directory exclusively after all source checks.
    output_dir = create_output_directory(output_dir)
    with (output_dir / "experiment.log").open("x", encoding="utf-8", newline="\n") as log:
        log.write(f"started_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"device={device}\ntrain_images={len(train_records)}\nvalidation_images={len(validation_records)}\n")
    _write_predictions(output_dir / "baseline_validation_predictions.csv", baseline_predictions)
    _write_json_exclusive(output_dir / "baseline_validation_metrics.json", baseline_metrics)
    manifest_fields = list(rows[0])
    with (output_dir / "train_validation_manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(rows)

    family_training.reviewed_trainer.set_reproducible_seed(SEED)
    train_transform, validation_transform = build_transforms()
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        category_training.CategoryImageDataset(train_records, train_paths, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0,
    )
    experiment_validation_loader = DataLoader(
        category_training.CategoryImageDataset(validation_records, validation_paths, validation_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    model = family_training.reviewed_trainer.build_model(weights_path).to(device)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    class_weights = torch.tensor(
        [len(train_records) / (len(CLASS_NAMES) * train_counts[name]) for name in CLASS_NAMES],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_state = None
    best_epoch = 0
    log_path = output_dir / "experiment.log"
    history_path = output_dir / "training_history.csv"
    with history_path.open("x", encoding="utf-8", newline="") as history_stream:
        fields = ["epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1"]
        writer = csv.DictWriter(history_stream, fieldnames=fields)
        writer.writeheader()
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            for epoch in range(1, EPOCHS + 1):
                model.train()
                loss_total = 0.0
                sample_total = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                    loss_total += float(loss.item()) * labels.size(0)
                    sample_total += labels.size(0)
                val_metrics, _ = evaluate(model, experiment_validation_loader, device)
                val_loss = 0.0
                # Compute deterministic validation loss without touching any test examples.
                model.eval()
                with torch.inference_mode():
                    for inputs, labels in experiment_validation_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        val_loss += float(criterion(model(inputs), labels).item()) * labels.size(0)
                val_loss /= len(validation_records)
                item = {
                    "epoch": epoch,
                    "train_loss": loss_total / max(sample_total, 1),
                    "validation_loss": val_loss,
                    "validation_accuracy": val_metrics["top1"]["accuracy"],
                    "validation_macro_f1": val_metrics["top1"]["macro_f1"],
                }
                history.append(item)
                writer.writerow(item)
                history_stream.flush()
                line = (f"epoch={epoch}/{EPOCHS} train_loss={item['train_loss']:.5f} "
                        f"validation_loss={val_loss:.5f} validation_macro_f1={item['validation_macro_f1']:.5f}")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                key = (item["validation_macro_f1"], -val_loss)
                if best_key is None or key > best_key:
                    best_key, best_epoch = key, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without a validation-selected checkpoint.")
    model.load_state_dict(best_state, strict=True)
    augmented_metrics, augmented_predictions = evaluate(model, experiment_validation_loader, device)
    _write_predictions(output_dir / "augmented_validation_predictions.csv", augmented_predictions)
    saved_baseline_predictions = read_prediction_csv(output_dir / "baseline_validation_predictions.csv")
    saved_augmented_predictions = read_prediction_csv(output_dir / "augmented_validation_predictions.csv")
    validate_prediction_pair(saved_baseline_predictions, saved_augmented_predictions)
    baseline_from_predictions = metrics_from_prediction_rows(saved_baseline_predictions)
    augmented_from_predictions = metrics_from_prediction_rows(saved_augmented_predictions)
    if baseline_from_predictions != baseline_metrics or augmented_from_predictions != augmented_metrics:
        raise ValueError("Validation metrics do not reproduce from the saved prediction CSV files.")
    _write_json_exclusive(output_dir / "augmented_validation_metrics.json", augmented_from_predictions)
    comparison = {
        "baseline": baseline_from_predictions,
        "augmentation_v1": augmented_from_predictions,
        "top1_delta": {
            "accuracy": augmented_from_predictions["top1"]["accuracy"] - baseline_from_predictions["top1"]["accuracy"],
            "macro_f1": augmented_from_predictions["top1"]["macro_f1"] - baseline_from_predictions["top1"]["macro_f1"],
            "per_class_f1": {
                c: augmented_from_predictions["top1"]["per_class"][c]["f1"] - baseline_from_predictions["top1"]["per_class"][c]["f1"]
                for c in CLASS_NAMES
            },
        },
        "glass_to_plastic_count_delta": (
            augmented_from_predictions["glass_to_plastic"]["top1_count"] - baseline_from_predictions["glass_to_plastic"]["top1_count"]
        ),
        "plastic_to_glass_count_delta": (
            augmented_from_predictions["plastic_to_glass"]["top1_count"] - baseline_from_predictions["plastic_to_glass"]["top1_count"]
        ),
        "selection_basis": "validation macro-F1, then validation loss; locked test set not evaluated",
        "metrics_recomputed_from_saved_predictions": True,
    }
    _write_json_exclusive(output_dir / "validation_comparison.json", comparison)

    checkpoint_path = output_dir / "best_model.pth"
    checkpoint_payload = {
        "state_dict": best_state,
        "architecture": ARCHITECTURE,
        "class_count": CLASS_COUNT,
        "class_names": list(CLASS_NAMES),
        "class_to_index": expected_mapping,
        "category_to_family": CATEGORY_TO_FAMILY,
        "seed": SEED,
        "confidence_threshold": THRESHOLD,
        "input_preprocessing": category_training.PREPROCESSING,
        "preprocessing": category_training.PREPROCESSING,
        "validation_preprocessing": category_training.PREPROCESSING,
        "pretrained_weights": pretrained_weights,
        "augmentation": AUGMENTATION,
        "provisional": True,
        "label_provenance": category_training.PROVENANCE,
        "best_epoch": best_epoch,
    }
    with checkpoint_path.open("xb") as checkpoint_file:
        torch.save(checkpoint_payload, checkpoint_file)
    metadata_out = {
        "experiment": "MSW category training-only augmentation v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output_dir.resolve()),
        "evaluation_scope": "train/validation only; locked test images were not opened or evaluated",
        "baseline_checkpoint": str((BASELINE_DIR / "best_model.pth").resolve()),
        "baseline_checkpoint_sha256": hashlib.sha256((BASELINE_DIR / "best_model.pth").read_bytes()).hexdigest(),
        "category_manifest": str(CATEGORY_MANIFEST.resolve()),
        "category_manifest_sha256": hashlib.sha256(CATEGORY_MANIFEST.read_bytes()).hexdigest(),
        "source_manifest": str(SOURCE_MANIFEST.resolve()),
        "source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        "train_validation_manifest_sha256": sha256_file(output_dir / "train_validation_manifest.csv"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_prediction_sha256": {
            "baseline": sha256_file(output_dir / "baseline_validation_predictions.csv"),
            "augmentation_v1": sha256_file(output_dir / "augmented_validation_predictions.csv"),
        },
        "dataset_root": str(dataset_root.resolve()),
        "train_validation_counts": actual_counts,
        "class_count": CLASS_COUNT,
        "class_names": list(CLASS_NAMES),
        "class_to_index": expected_mapping,
        "category_to_family": CATEGORY_TO_FAMILY,
        "label_provenance": category_training.PROVENANCE,
        "verified_ground_truth": False,
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "class_weighted_loss": True,
        "augmentation": AUGMENTATION,
        "input_preprocessing": category_training.PREPROCESSING,
        "validation_preprocessing": category_training.PREPROCESSING,
        "pretrained_weights": pretrained_weights,
        "confidence_threshold": THRESHOLD,
        "best_epoch": best_epoch,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": np.__version__,
    }
    validate_checkpoint_metadata(checkpoint_payload, metadata_out)
    _write_json_exclusive(output_dir / "experiment_metadata.json", metadata_out)
    print(f"Validation-only results saved to {output_dir}")
    return {"output_dir": output_dir, "baseline": baseline_from_predictions, "experiment": augmented_from_predictions,
            "comparison": comparison, "metadata": metadata_out}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"new experiment directory under {MODELS_DIR} (must not already exist)",
    )
    args = parser.parse_args()
    result = run_experiment(args.output_dir)
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
