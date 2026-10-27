"""Train a strictly provisional binary family baseline from folder-derived labels.

This experiment is separate from the human-reviewed family trainer and the
paper/cardboard pilot. It uses only local cached pretrained weights and refuses
to overwrite its dedicated output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_waste_family as reviewed_trainer  # noqa: E402
import prepare_provisional_family as provisional_data  # noqa: E402


DEFAULT_MANIFEST = provisional_data.OUTPUT_PATH
DEFAULT_DATASET_ROOT = provisional_data.workflow.DATASET_DIR
OUTPUT_DIR = provisional_data.PROVISIONAL_OUTPUT_DIR
CLASS_NAMES = ("MSW", "E_WASTE")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = reviewed_trainer.IMAGE_SIZE
PREPROCESSING = reviewed_trainer.PREPROCESSING
REQUIRED_COLUMNS = set(provisional_data.MANIFEST_FIELDS)


class ProvisionalTrainingError(ValueError):
    """Raised for invalid provisional manifest/data or unsafe training output."""


@dataclass(frozen=True)
class ProvisionalImage:
    source_relative_path: str
    sha256: str
    family: str
    split: str
    source_folder: str


def read_provisional_manifest(
    manifest_path: Path,
) -> tuple[list[dict[str, str]], list[ProvisionalImage]]:
    """Validate provenance and split integrity; defer test labels until evaluation."""
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(f"Provisional family manifest not found: {manifest_path}")
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ProvisionalTrainingError("Manifest missing columns: " + ", ".join(sorted(missing)))
        rows = [dict(row) for row in reader]

    hash_splits: dict[str, set[str]] = defaultdict(set)
    for line, row in enumerate(rows, start=2):
        split = (row.get("split") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        if split and split not in {"train", "validation", "test"}:
            raise ProvisionalTrainingError(f"Invalid split on manifest line {line}: {split!r}.")
        if split and digest:
            hash_splits[digest].add(split)
    leaking = [digest for digest, splits in hash_splits.items() if len(splits) > 1]
    if leaking:
        raise ProvisionalTrainingError(f"Exact duplicate hash leakage across splits ({len(leaking)} group(s)).")

    train_validation: list[ProvisionalImage] = []
    seen_hashes: set[str] = set()
    for line, row in enumerate(rows, start=2):
        split = (row.get("split") or "").strip()
        if split not in {"train", "validation"}:
            continue
        relative = (row.get("source_relative_path") or "").strip().replace("\\", "/")
        family = (row.get("provisional_family") or "").strip()
        folder = (row.get("source_folder") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        if row.get("group_status") != "provisional_include" or row.get("canonical_path") != relative:
            raise ProvisionalTrainingError(f"Split row is not a canonical provisional group on line {line}.")
        if row.get("label_provenance") != provisional_data.PROVENANCE:
            raise ProvisionalTrainingError(f"Invalid label provenance for {relative!r}.")
        if family not in CLASS_TO_INDEX or row.get("source_folder_mapping") != f"{folder} -> {family}":
            raise ProvisionalTrainingError(f"Invalid provisional family mapping for {relative!r}.")
        safe_relative_path(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProvisionalTrainingError(f"Invalid SHA-256 in manifest for {relative!r}.")
        if digest in seen_hashes:
            raise ProvisionalTrainingError(f"Duplicate canonical hash appears more than once: {relative!r}.")
        seen_hashes.add(digest)
        train_validation.append(ProvisionalImage(relative, digest, family, split, folder))
    if not train_validation:
        raise ProvisionalTrainingError("No provisional train/validation rows have split assignments.")
    validate_train_validation_counts(train_validation)
    return rows, train_validation


def safe_relative_path(value: str) -> PurePosixPath:
    normalized = (value or "").strip().replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath((value or "").strip())
    if (
        not normalized or posix.is_absolute() or windows.is_absolute() or windows.drive
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ProvisionalTrainingError(f"Unsafe source_relative_path: {value!r}")
    return posix


def validate_train_validation_counts(records: list[ProvisionalImage]) -> dict[str, dict[str, int]]:
    counts = {
        split: Counter(record.family for record in records if record.split == split)
        for split in ("train", "validation")
    }
    missing = [f"{split}/{family}" for split in counts for family in CLASS_NAMES if not counts[split][family]]
    if missing:
        raise ProvisionalTrainingError("Train and validation need both families; missing: " + ", ".join(missing))
    return {split: {family: counts[split][family] for family in CLASS_NAMES} for split in counts}


def validate_test_records(rows: list[dict[str, str]]) -> list[ProvisionalImage]:
    """Inspect test class/family labels only after validation-selected checkpoint."""
    records = []
    for row in rows:
        if row.get("split") != "test" or row.get("group_status") != "provisional_include":
            continue
        relative = (row.get("source_relative_path") or "").strip().replace("\\", "/")
        family = (row.get("provisional_family") or "").strip()
        folder = (row.get("source_folder") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        if (
            row.get("label_provenance") != provisional_data.PROVENANCE
            or family not in CLASS_TO_INDEX
            or row.get("canonical_path") != relative
            or row.get("source_folder_mapping") != f"{folder} -> {family}"
            or len(digest) != 64
        ):
            raise ProvisionalTrainingError(f"Invalid provisional test record for {relative!r}.")
        safe_relative_path(relative)
        records.append(ProvisionalImage(relative, digest, family, "test", folder))
    counts = Counter(record.family for record in records)
    missing = [family for family in CLASS_NAMES if not counts[family]]
    if missing:
        raise ProvisionalTrainingError("Held-out test split lacks provisional class(es): " + ", ".join(missing))
    if len({record.sha256 for record in records}) != len(records):
        raise ProvisionalTrainingError("Held-out test contains duplicate provisional hash groups.")
    return records


def validate_image_file(dataset_root: Path, record: ProvisionalImage) -> Path:
    relative = safe_relative_path(record.source_relative_path)
    path = Path(dataset_root).resolve().joinpath(*relative.parts)
    root = Path(dataset_root).resolve()
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ProvisionalTrainingError(f"Missing or unsafe provisional image {record.source_relative_path!r}.") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ProvisionalTrainingError(f"Provisional image path traverses a symlink: {record.source_relative_path!r}")
    if not resolved.is_file() or reviewed_trainer.sha256_file(resolved) != record.sha256:
        raise ProvisionalTrainingError(f"Provisional image is missing or SHA-256 changed: {record.source_relative_path!r}")
    try:
        with Image.open(resolved) as image:
            image.verify()
        with Image.open(resolved) as image:
            image.load()
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise ProvisionalTrainingError(f"Provisional image is unreadable: {record.source_relative_path!r}: {error}") from error
    return resolved


def _split_counts(records: list[ProvisionalImage], splits: tuple[str, ...]) -> dict[str, dict[str, int]]:
    counts = {split: Counter() for split in splits}
    for record in records:
        if record.split in counts:
            counts[record.split][record.family] += 1
    return {split: {family: counts[split][family] for family in CLASS_NAMES} for split in splits}


def _source_category_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row.get("group_status") == "provisional_include":
            counts[row["source_folder"]][row["split"]] += 1
    return {folder: dict(sorted(split_counts.items())) for folder, split_counts in sorted(counts.items())}


def train_provisional_classifier(
    manifest_path: Path = DEFAULT_MANIFEST,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = OUTPUT_DIR,
    *, seed: int = 42, epochs: int = 8, batch_size: int = 16, learning_rate: float = 1e-3,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.resolve() != OUTPUT_DIR.resolve():
        raise ProvisionalTrainingError(f"Provisional outputs may only be written under {OUTPUT_DIR}.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")
    rows, train_validation = read_provisional_manifest(manifest_path)
    test_hashes = {
        row["sha256"] for row in rows
        if row.get("split") == "test" and row.get("group_status") == "provisional_include"
    }
    if test_hashes & {record.sha256 for record in train_validation}:
        raise ProvisionalTrainingError("Exact hash leakage between train/validation and test.")
    counts = validate_train_validation_counts(train_validation)
    weights_path = reviewed_trainer.resolve_cached_weights()
    train_records = [record for record in train_validation if record.split == "train"]
    validation_records = [record for record in train_validation if record.split == "validation"]
    train_paths = {record.source_relative_path: validate_image_file(dataset_root, record) for record in train_records}
    validation_paths = {record.source_relative_path: validate_image_file(dataset_root, record) for record in validation_records}

    reviewed_trainer.set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(reviewed_trainer.IMAGENET_MEAN, reviewed_trainer.IMAGENET_STD),
    ])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        reviewed_trainer.FamilyImageDataset(train_records, dataset_root, transform, train_paths),
        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        reviewed_trainer.FamilyImageDataset(validation_records, dataset_root, transform, validation_paths),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model = reviewed_trainer.build_model(weights_path).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    class_weights = reviewed_trainer.training_class_weights(train_records).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, output_dir / "split_manifest.csv")
    history = []
    best_key = None
    checkpoint_path = output_dir / "best_model.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.size(0)
            seen += targets.size(0)
        validation_loss, validation_metrics = reviewed_trainer.evaluate(
            model, validation_loader, device, criterion
        )
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        })
        key = (validation_metrics["macro_f1"], -validation_loss)
        if best_key is None or key > best_key:
            best_key = key
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "class_to_index": CLASS_TO_INDEX,
                "epoch": epoch,
                "preprocessing": PREPROCESSING,
                "label_provenance": provisional_data.PROVENANCE,
                "provisional": True,
            }, checkpoint_path)

    # Do not validate, count, or load test labels/files until after selection.
    test_records = validate_test_records(rows)
    test_paths = {record.source_relative_path: validate_image_file(dataset_root, record) for record in test_records}
    counts["test"] = {family: sum(record.family == family for record in test_records) for family in CLASS_NAMES}
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    test_loader = DataLoader(
        reviewed_trainer.FamilyImageDataset(test_records, dataset_root, transform, test_paths),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loss, test_metrics = reviewed_trainer.evaluate(model, test_loader, device, criterion)

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    source_counts = _source_category_counts(rows)
    excluded_counts = Counter(
        provisional_data.exclusion_reason_category(row)
        for row in rows if row.get("group_status") != "provisional_include"
    )
    class_mapping = {
        "class_names": list(CLASS_NAMES),
        "class_to_index": CLASS_TO_INDEX,
        "source_folder_mapping": dict(sorted({
            row["source_folder"]: row["provisional_family"]
            for row in rows if row.get("label_provenance") == provisional_data.PROVENANCE
        }.items())),
        "label_provenance": provisional_data.PROVENANCE,
        "verified_ground_truth": False,
    }
    (output_dir / "class_mapping.json").write_text(json.dumps(class_mapping, indent=2), encoding="utf-8")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_status": "PROVISIONAL",
        "architecture": "torchvision ResNet18",
        "pretrained_weights": str(weights_path.resolve()),
        **class_mapping,
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": reviewed_trainer.sha256_file(Path(manifest_path)),
        "dataset_root": str(Path(dataset_root).resolve()),
        "label_provenance_note": "All binary labels are provisional source-folder mappings, not verified ground truth.",
        "human_review_decisions_used_as_labels": False,
        "human_hold_exclude_or_conflicting_decisions_respected": True,
        "split_counts_by_family": counts,
        "split_counts_by_source_category": source_counts,
        "excluded_row_counts_by_reason": dict(excluded_counts),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "backbone_frozen": True,
        "class_weighted_loss": True,
        "image_size": IMAGE_SIZE,
        "preprocessing": PREPROCESSING,
        "confidence_threshold": 0.70,
        "confidence_threshold_note": "Provisional demonstration threshold; not calibrated.",
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": np.__version__,
        "test_set_used_only_after_validation_model_selection": True,
        "best_epoch": int(best_checkpoint["epoch"]),
        "warning": "Folder-derived evaluation is provisional and cannot establish family-label accuracy or production performance.",
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "final_test_metrics.json").write_text(json.dumps({
        "evaluation_status": "PROVISIONAL folder-derived labels",
        "test_label_provenance": provisional_data.PROVENANCE,
        "loss": test_loss,
        **test_metrics,
    }, indent=2), encoding="utf-8")
    return {"output_dir": output_dir, "metadata": metadata, "test_metrics": test_metrics}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    try:
        result = train_provisional_classifier(
            args.manifest, args.dataset_root, args.output_dir,
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Provisional family training stopped safely: {error}\n")
    print(f"PROVISIONAL model artifacts: {result['output_dir']}")
    print(f"Best validation-selected epoch: {result['metadata']['best_epoch']}")
    print("Held-out provisional metrics: " + json.dumps(result["test_metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
