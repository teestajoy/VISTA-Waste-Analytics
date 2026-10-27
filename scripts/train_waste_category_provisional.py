"""Train a separate six-category, folder-derived provisional classifier.

The input is the existing provisional family manifest. Its deterministic
hash-group splits are reused unchanged, while unsupported and conflicting
duplicate groups remain excluded. All category targets are folder proxies,
not verified labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import prepare_provisional_family as family_data  # noqa: E402
import train_waste_family_provisional as family_trainer  # noqa: E402

DEFAULT_SOURCE_MANIFEST = family_data.OUTPUT_PATH
OUTPUT_DIR = PROJECT_ROOT / "models" / "waste_category_provisional"
CLASS_NAMES = ("paper", "cardboard", "plastic", "glass", "organic", "battery")
CATEGORY_TO_FAMILY = {
    "paper": "MSW",
    "cardboard": "MSW",
    "plastic": "MSW",
    "glass": "MSW",
    "organic": "MSW",
    "battery": "E_WASTE",
}
PROVENANCE = family_data.PROVENANCE
IMAGE_SIZE = family_trainer.IMAGE_SIZE
PREPROCESSING = family_trainer.PREPROCESSING
SPLITS = ("train", "validation", "test")
CATEGORY_MANIFEST_FIELDS = (
    "source_relative_path", "sha256", "source_folder", "provisional_category",
    "provisional_family", "label_provenance", "duplicate_group_id",
    "canonical_path", "group_status", "split", "exclusion_reason",
)


class CategoryTrainingError(ValueError):
    """Raised for an invalid category manifest or unsafe training condition."""


@dataclass(frozen=True)
class CategoryImage:
    source_relative_path: str
    sha256: str
    category: str
    split: str
    source_folder: str


def build_category_manifest(
    source_manifest: Path,
) -> tuple[list[dict[str, str]], list[CategoryImage]]:
    """Project supported categories from the existing deterministic family manifest."""
    if not Path(source_manifest).is_file():
        raise FileNotFoundError(f"Provisional family manifest not found: {source_manifest}")
    with Path(source_manifest).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = set(family_data.MANIFEST_FIELDS)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise CategoryTrainingError("Source manifest missing columns: " + ", ".join(sorted(missing)))
        source_rows = [dict(row) for row in reader]

    candidate_rows = [
        row for row in source_rows if (row.get("source_folder") or "").strip() in CLASS_NAMES
    ]
    by_hash_folders: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        digest = (row.get("sha256") or "").strip().lower()
        folder = (row.get("source_folder") or "").strip()
        if digest:
            by_hash_folders[digest].add(folder)
    conflicting_hashes = {
        digest for digest, folders in by_hash_folders.items() if len(folders) > 1
    }

    manifest_rows: list[dict[str, str]] = []
    included: list[CategoryImage] = []
    seen_included_hashes: set[str] = set()
    for row in candidate_rows:
        relative = (row.get("source_relative_path") or "").strip().replace("\\", "/")
        digest = (row.get("sha256") or "").strip().lower()
        folder = (row.get("source_folder") or "").strip()
        source_status = (row.get("group_status") or "").strip()
        canonical = (row.get("canonical_path") or "").strip().replace("\\", "/")
        split = (row.get("split") or "").strip()
        reason = (row.get("exclusion_reason") or "").strip()
        eligible = source_status == "provisional_include" and canonical == relative
        if digest in conflicting_hashes:
            eligible = False
            split = ""
            reason = "cross-category exact-hash conflict inherited from family audit"
        if eligible:
            if row.get("label_provenance") != PROVENANCE:
                raise CategoryTrainingError(f"Invalid label provenance for {relative!r}.")
            if folder not in CATEGORY_TO_FAMILY or row.get("provisional_family") != CATEGORY_TO_FAMILY[folder]:
                raise CategoryTrainingError(f"Unsupported category/family mapping for {relative!r}.")
            if split not in SPLITS:
                raise CategoryTrainingError(f"Invalid split for {relative!r}: {split!r}.")
            if not relative or not digest:
                raise CategoryTrainingError("Included category row has a blank path or hash.")
            if digest in seen_included_hashes:
                raise CategoryTrainingError(f"Duplicate canonical hash appears more than once: {relative!r}.")
            seen_included_hashes.add(digest)
            if split in {"train", "validation"}:
                included.append(CategoryImage(relative, digest, folder, split, folder))
            status = "provisional_include"
        else:
            split = ""
            status = "provisional_excluded"
            if not reason:
                reason = "not a canonical included row in the provisional family manifest"

        manifest_rows.append({
            "source_relative_path": relative,
            "sha256": digest,
            "source_folder": folder,
            "provisional_category": folder,
            "provisional_family": CATEGORY_TO_FAMILY[folder],
            "label_provenance": PROVENANCE,
            "duplicate_group_id": (row.get("duplicate_group_id") or "").strip(),
            "canonical_path": canonical,
            "group_status": status,
            "split": split,
            "exclusion_reason": reason,
        })

    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest_rows:
        if row["group_status"] == "provisional_include" and row["split"]:
            hash_splits[row["sha256"]].add(row["split"])
    if any(len(splits) != 1 for splits in hash_splits.values()):
        raise CategoryTrainingError("Exact duplicate category hashes cross train/validation/test splits.")
    _validate_class_counts(included)
    return manifest_rows, included


def _validate_class_counts(
    records: list[CategoryImage], splits: tuple[str, ...] = ("train", "validation")
) -> dict[str, dict[str, int]]:
    counts = {
        split: Counter(record.category for record in records if record.split == split)
        for split in splits
    }
    missing = [f"{split}/{category}" for split in splits for category in CLASS_NAMES if not counts[split][category]]
    if missing:
        raise CategoryTrainingError(
            "Every category must have at least one canonical image in the checked split(s); missing: "
            + ", ".join(missing)
        )
    return {split: {category: counts[split][category] for category in CLASS_NAMES} for split in splits}


def _safe_image_path(dataset_root: Path, record: CategoryImage) -> Path:
    try:
        path = family_trainer.validate_image_file(
            dataset_root,
            family_trainer.ProvisionalImage(
                record.source_relative_path,
                record.sha256,
                CATEGORY_TO_FAMILY[record.category],
                record.split,
                record.source_folder,
            ),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CategoryTrainingError(str(error)) from error
    return path


class CategoryImageDataset(Dataset):
    def __init__(self, records, paths, transform):
        self.records = records
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            with Image.open(self.paths[record.source_relative_path]) as opened:
                image = opened.convert("RGB")
                image.load()
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise CategoryTrainingError(f"Unreadable image {record.source_relative_path!r}: {error}") from error
        return self.transform(image), CLASS_NAMES.index(record.category)


def _metrics(confusion: np.ndarray) -> dict[str, Any]:
    per_class = {}
    f1s = []
    for index, category in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[category] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
    total = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    total = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += criterion(logits, targets).item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            for actual, predicted in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual, predicted] += 1
            total += targets.size(0)
    return loss_sum / max(total, 1), _metrics(confusion)


def _test_records_after_selection(rows: list[dict[str, str]]) -> list[CategoryImage]:
    """Read category labels from held-out rows only after checkpoint selection."""
    records = []
    for row in rows:
        if row.get("group_status") != "provisional_include" or row.get("split") != "test":
            continue
        category = (row.get("provisional_category") or "").strip()
        relative = (row.get("source_relative_path") or "").strip().replace("\\", "/")
        digest = (row.get("sha256") or "").strip().lower()
        if category not in CLASS_NAMES or row.get("label_provenance") != PROVENANCE or len(digest) != 64:
            raise CategoryTrainingError(f"Invalid provisional test row for {relative!r}.")
        records.append(CategoryImage(relative, digest, category, "test", row["source_folder"]))
    counts = Counter(record.category for record in records)
    missing = [category for category in CLASS_NAMES if not counts[category]]
    if missing:
        raise CategoryTrainingError("Held-out test split missing category(s): " + ", ".join(missing))
    if len({record.sha256 for record in records}) != len(records):
        raise CategoryTrainingError("Held-out test contains duplicate category hashes.")
    return records


def train_category_classifier(
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    dataset_root: Path = family_data.workflow.DATASET_DIR,
    output_dir: Path = OUTPUT_DIR,
    *, seed: int = 42, epochs: int = 8, batch_size: int = 16, learning_rate: float = 1e-3,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.resolve() != OUTPUT_DIR.resolve():
        raise CategoryTrainingError(f"Category outputs may only be written under {OUTPUT_DIR}.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")
    manifest_rows, records = build_category_manifest(source_manifest)
    split_counts = _validate_class_counts(records)
    print(f"Train/validation counts by category: {split_counts}", flush=True)
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    train_paths = {record.source_relative_path: _safe_image_path(dataset_root, record) for record in train_records}
    validation_paths = {record.source_relative_path: _safe_image_path(dataset_root, record) for record in validation_records}
    print(
        f"Validated {len(train_paths)} training and {len(validation_paths)} validation images; "
        "test images remain unopened until checkpoint selection.",
        flush=True,
    )

    family_trainer.reviewed_trainer.set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(family_trainer.reviewed_trainer.IMAGENET_MEAN, family_trainer.reviewed_trainer.IMAGENET_STD),
    ])
    train_loader = DataLoader(
        CategoryImageDataset(train_records, train_paths, transform),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    validation_loader = DataLoader(
        CategoryImageDataset(validation_records, validation_paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    weights_path = family_trainer.reviewed_trainer.resolve_cached_weights()
    model = family_trainer.reviewed_trainer.build_model(weights_path).to(device)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    class_counts = Counter(record.category for record in train_records)
    class_weights = torch.tensor(
        [len(train_records) / (len(CLASS_NAMES) * class_counts[name]) for name in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    best_key = None
    best_state = None
    best_epoch = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
        validation_loss, validation_metrics = _evaluate(model, validation_loader, device, criterion)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        })
        print(
            f"Epoch {epoch}/{epochs}: train_loss={history[-1]['train_loss']:.4f}, "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        key = (validation_metrics["macro_f1"], -validation_loss)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    # Do not parse test category labels or open test images until validation has selected a checkpoint.
    test_records = _test_records_after_selection(manifest_rows)
    split_counts["test"] = {category: sum(record.category == category for record in test_records) for category in CLASS_NAMES}
    all_included_records = records + test_records
    test_paths = {record.source_relative_path: _safe_image_path(dataset_root, record) for record in test_records}
    test_loader = DataLoader(
        CategoryImageDataset(test_records, test_paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model.load_state_dict(best_state)
    test_loss, test_metrics = _evaluate(model, test_loader, device, criterion)

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CATEGORY_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    checkpoint_path = output_dir / "best_model.pth"
    torch.save({
        "state_dict": best_state,
        "class_names": CLASS_NAMES,
        "class_to_index": {name: index for index, name in enumerate(CLASS_NAMES)},
        "category_to_family": CATEGORY_TO_FAMILY,
        "preprocessing": PREPROCESSING,
        "provisional": True,
        "label_provenance": PROVENANCE,
        "best_epoch": best_epoch,
    }, checkpoint_path)
    mapping = {
        "class_names": list(CLASS_NAMES),
        "class_to_index": {name: index for index, name in enumerate(CLASS_NAMES)},
        "category_to_family": CATEGORY_TO_FAMILY,
        "label_provenance": PROVENANCE,
        "verified_ground_truth": False,
    }
    (output_dir / "class_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "model_status": "PROVISIONAL",
        "architecture": "torchvision ResNet18",
        "pretrained_weights": str(weights_path),
        "class_names": list(CLASS_NAMES),
        "class_to_index": mapping["class_to_index"],
        "category_to_family": CATEGORY_TO_FAMILY,
        "label_provenance": PROVENANCE,
        "verified_ground_truth": False,
        "label_provenance_note": "Category and family labels were derived from source-folder names and are not verified ground truth.",
        "human_review_decisions_used_as_labels": False,
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_manifest_sha256": hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "included_unique_hash_groups": len({record.sha256 for record in all_included_records}),
        "excluded_rows_by_reason": dict(Counter(
            row["exclusion_reason"] or "unspecified" for row in manifest_rows
            if row["group_status"] != "provisional_include"
        )),
        "excluded_source_folders": ["metal", "trash", "textile"],
        "split_counts_by_category": split_counts,
        "split_counts_by_family": {
            split: dict(Counter(CATEGORY_TO_FAMILY[record.category] for record in all_included_records if record.split == split))
            for split in SPLITS
        },
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "backbone_frozen": True,
        "class_weighted_loss": True,
        "image_size": IMAGE_SIZE,
        "preprocessing": PREPROCESSING,
        "confidence_threshold": 0.70,
        "confidence_threshold_note": "Uncalibrated demonstration threshold; not a verified probability cutoff.",
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": np.__version__,
        "best_epoch": best_epoch,
        "test_set_used_only_after_validation_model_selection": True,
        "warning": "Held-out metrics measure agreement with folder-derived labels only; real-world category performance is not established.",
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "final_test_metrics.json").write_text(json.dumps({
        "evaluation_status": "PROVISIONAL folder-derived category labels",
        "test_label_provenance": PROVENANCE,
        "loss": test_loss,
        **test_metrics,
    }, indent=2), encoding="utf-8")
    return {"output_dir": output_dir, "metadata": metadata, "test_metrics": test_metrics}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=family_data.workflow.DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")
    try:
        result = train_category_classifier(
            args.manifest, args.dataset_root, args.output_dir,
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        )
    except (CategoryTrainingError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"Category training stopped safely: {error}\n")
    print(f"PROVISIONAL category model and reports saved under: {result['output_dir']}")
    print(f"Best validation-selected epoch: {result['metadata']['best_epoch']}")
    print(f"Split counts by category: {result['metadata']['split_counts_by_category']}")
    print(f"Final held-out metrics: {result['test_metrics']}")


if __name__ == "__main__":
    main()
