"""Train an experimental ResNet18 MSW/E-waste classifier from an audited manifest.

Only canonical, human-included rows with an existing train/validation/test
assignment are eligible. The test labels are read only after validation-based
checkpoint selection. No pretrained weights are downloaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "family_split_manifest.csv"
DEFAULT_DATASET_ROOT = (
    Path.home() / "Downloads" / "Waste_classification_dataset" / "final_merged_dataset"
)
OUTPUT_DIR = PROJECT_ROOT / "models" / "waste_family_resnet18_pilot"
CLASS_NAMES = ("MSW", "E_WASTE")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "validation", "test")
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = {
    "source_relative_path", "sha256", "review_status", "reviewed_family",
    "group_status", "split",
}
PREPROCESSING = {
    "color_mode": "RGB",
    "resize": [IMAGE_SIZE, IMAGE_SIZE],
    "tensor": "ToTensor",
    "normalize_mean": list(IMAGENET_MEAN),
    "normalize_std": list(IMAGENET_STD),
}


class FamilyDatasetValidationError(ValueError):
    """Raised for unsafe, unreviewed, inconsistent, or unusable family data."""


@dataclass(frozen=True)
class FamilyImage:
    source_relative_path: str
    sha256: str
    family: str
    split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_source_path(dataset_root: Path, relative: str) -> Path:
    value = (relative or "").strip().replace("\\", "/")
    posix = PurePosixPath(value)
    windows = PureWindowsPath((relative or "").strip())
    if (
        not value or posix.is_absolute() or windows.is_absolute() or windows.drive
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise FamilyDatasetValidationError(f"Unsafe source_relative_path: {relative!r}")
    root = dataset_root.resolve()
    candidate = root.joinpath(*posix.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise FamilyDatasetValidationError(
            f"Image path is missing or resolves outside the dataset root: {relative!r}"
        ) from error
    current = root
    for part in posix.parts:
        current = current / part
        if current.is_symlink():
            raise FamilyDatasetValidationError(f"Image path traverses a symlink: {relative!r}")
    if not resolved.is_file():
        raise FamilyDatasetValidationError(f"Manifest path is not a file: {relative!r}")
    return resolved


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[FamilyImage]]:
    """Read train/validation records and defer all test labels until evaluation."""
    if not path.is_file():
        raise FileNotFoundError(f"Family split manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise FamilyDatasetValidationError(
                "Family manifest is missing columns: " + ", ".join(sorted(missing))
            )
        rows = [dict(row) for row in reader]

    hash_splits: dict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(rows, start=2):
        split = (row.get("split") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        if split and split not in SPLITS:
            raise FamilyDatasetValidationError(
                f"Invalid split {split!r} on manifest line {line_number}."
            )
        if split and digest:
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise FamilyDatasetValidationError(
                    f"Invalid SHA-256 on manifest line {line_number}."
                )
            hash_splits[digest].add(split)
    leaked = [digest for digest, splits in hash_splits.items() if len(splits) > 1]
    if leaked:
        raise FamilyDatasetValidationError(
            f"Exact-duplicate hash leakage found across splits ({len(leaked)} group(s))."
        )

    training_records: list[FamilyImage] = []
    for line_number, row in enumerate(rows, start=2):
        split = (row.get("split") or "").strip()
        if split not in {"train", "validation"}:
            continue
        if (row.get("group_status") or "").strip() != "include":
            raise FamilyDatasetValidationError(
                f"Manifest row on line {line_number} has a split but is not an included canonical group."
            )
        relative = (row.get("source_relative_path") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        status = (row.get("review_status") or "").strip()
        family = (row.get("reviewed_family") or "").strip()
        if status != "include":
            raise FamilyDatasetValidationError(
                f"Training/validation row {relative!r} is not explicitly human-included."
            )
        if family not in CLASS_TO_INDEX:
            raise FamilyDatasetValidationError(
                f"Invalid reviewed_family {family!r} for {relative!r}."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise FamilyDatasetValidationError(f"Invalid SHA-256 for {relative!r}.")
        if not relative:
            raise FamilyDatasetValidationError(f"Blank source path on manifest line {line_number}.")
        training_records.append(FamilyImage(relative, digest, family, split))

    if not training_records:
        raise FamilyDatasetValidationError(
            "No canonical, human-included train or validation rows have split assignments. "
            "Review more images from both families and prepare a new family manifest first."
        )
    seen_hashes: set[str] = set()
    for record in training_records:
        if record.sha256 in seen_hashes:
            raise FamilyDatasetValidationError(
                f"A canonical duplicate hash appears more than once: {record.source_relative_path!r}."
            )
        seen_hashes.add(record.sha256)
    return rows, training_records


def class_counts(
    records: Iterable[FamilyImage], splits: tuple[str, ...] = ("train", "validation")
) -> dict[str, dict[str, int]]:
    counts = {split: Counter({family: 0 for family in CLASS_NAMES}) for split in splits}
    for record in records:
        if record.split in counts:
            counts[record.split][record.family] += 1
    return {split: dict(counter) for split, counter in counts.items()}


def training_class_weights(records: Iterable[FamilyImage]) -> torch.Tensor:
    """Give each family equal total loss weight despite count imbalance."""
    counts = Counter(record.family for record in records)
    if any(counts[family] == 0 for family in CLASS_NAMES):
        raise FamilyDatasetValidationError("Class weights require training examples from both families.")
    total = sum(counts.values())
    return torch.tensor(
        [total / (len(CLASS_NAMES) * counts[family]) for family in CLASS_NAMES],
        dtype=torch.float32,
    )


def validate_train_validation_counts(records: list[FamilyImage]) -> dict[str, dict[str, int]]:
    counts = class_counts(records)
    missing = [f"{split}/{family}" for split in counts for family, count in counts[split].items() if count == 0]
    if missing:
        raise FamilyDatasetValidationError(
            "Train and validation each need reviewed examples from both families; missing: "
            + ", ".join(missing)
        )
    return counts


def validate_test_records(rows: list[dict[str, str]]) -> list[FamilyImage]:
    """Read and validate held-out labels only after model selection is complete."""
    records: list[FamilyImage] = []
    for line_number, row in enumerate(rows, start=2):
        if (row.get("split") or "").strip() != "test":
            continue
        if (row.get("group_status") or "").strip() != "include":
            continue
        relative = (row.get("source_relative_path") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        status = (row.get("review_status") or "").strip()
        family = (row.get("reviewed_family") or "").strip()
        if status != "include":
            raise FamilyDatasetValidationError(
                f"Final test row {relative!r} is not explicitly human-included."
            )
        if family not in CLASS_TO_INDEX:
            raise FamilyDatasetValidationError(f"Invalid final test family {family!r} for {relative!r}.")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not relative:
            raise FamilyDatasetValidationError(
                f"Final test row on manifest line {line_number} has an invalid path or SHA-256."
            )
        records.append(FamilyImage(relative, digest, family, "test"))
    if not records:
        raise FamilyDatasetValidationError(
            "After validation checkpoint selection, final evaluation requires canonical test rows."
        )
    counts = class_counts(records, ("test",))["test"]
    missing = [family for family, count in counts.items() if count == 0]
    if missing:
        raise FamilyDatasetValidationError(
            "After checkpoint selection, final test data is missing family/families: "
            + ", ".join(missing)
        )
    return records


def validate_image_file(dataset_root: Path, record: FamilyImage) -> Path:
    path = _safe_source_path(dataset_root, record.source_relative_path)
    try:
        actual_hash = sha256_file(path)
    except OSError as error:
        raise FamilyDatasetValidationError(
            f"Could not read image {record.source_relative_path!r} for hashing: {error}"
        ) from error
    if actual_hash != record.sha256:
        raise FamilyDatasetValidationError(
            f"SHA-256 mismatch for {record.source_relative_path!r}; source may have changed."
        )
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise FamilyDatasetValidationError(
            f"Image is unreadable or corrupt: {record.source_relative_path!r}: {error}"
        ) from error
    return path


def validate_images(dataset_root: Path, records: Iterable[FamilyImage]) -> dict[str, Path]:
    return {record.source_relative_path: validate_image_file(dataset_root, record) for record in records}


class FamilyImageDataset(Dataset):
    def __init__(self, records, dataset_root, transform, paths=None):
        self.records = records
        self.dataset_root = dataset_root
        self.transform = transform
        self.paths = paths or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.paths.get(record.source_relative_path) or _safe_source_path(
            self.dataset_root, record.source_relative_path
        )
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.load()
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise FamilyDatasetValidationError(
                f"Could not load {record.source_relative_path!r}: {error}"
            ) from error
        return self.transform(image), CLASS_TO_INDEX[record.family]


def resolve_cached_weights(cache_dir: Path | None = None) -> Path:
    filename = ResNet18_Weights.DEFAULT.url.rsplit("/", 1)[-1]
    cache = cache_dir or Path(torch.hub.get_dir()) / "checkpoints"
    path = cache / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"ImageNet ResNet18 weights are not cached locally: {path}. No download attempted."
        )
    return path


def build_model(weights_path: Path | None = None) -> nn.Module:
    model = models.resnet18(weights=None)
    if weights_path is not None:
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"Could not load cached ResNet18 weights: {error}") from error
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    per_class = {}
    f1_values = []
    for index, family in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[family] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
    total = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_total = 0.0
    total = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss_total += criterion(logits, targets).item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            for actual, predicted in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual, predicted] += 1
            total += targets.size(0)
    return loss_total / max(total, 1), metrics_from_confusion(confusion)


def train_family_classifier(
    manifest_path: Path = DEFAULT_MANIFEST,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path | None = None,
    *, seed: int = 42, epochs: int = 20, batch_size: int = 8, learning_rate: float = 1e-3,
) -> dict[str, Any]:
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    if output_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError(f"Training outputs must be written only to {OUTPUT_DIR}.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {output_dir}")
    rows, train_validation = read_manifest(manifest_path)
    counts = validate_train_validation_counts(train_validation)
    print(f"Train class counts: {counts['train']}")
    print(f"Validation class counts: {counts['validation']}")
    print("Family dataset warning: metrics are estimates on a human-reviewed local pilot and do not establish general performance.")
    weights_path = resolve_cached_weights()
    train_records = [record for record in train_validation if record.split == "train"]
    validation_records = [record for record in train_validation if record.split == "validation"]
    train_paths = validate_images(dataset_root, train_records)
    validation_paths = validate_images(dataset_root, validation_records)

    set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_loader = DataLoader(
        FamilyImageDataset(train_records, dataset_root, transform, train_paths),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    validation_loader = DataLoader(
        FamilyImageDataset(validation_records, dataset_root, transform, validation_paths),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    model = build_model(weights_path).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=training_class_weights(train_records).to(device))

    output_dir.mkdir(parents=True, exist_ok=False)
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
        validation_loss, validation_metrics = evaluate(model, validation_loader, device, criterion)
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
                "epoch": epoch,
                "preprocessing": PREPROCESSING,
            }, checkpoint_path)

    # Held-out labels and files are first inspected after model selection.
    test_records = validate_test_records(rows)
    test_paths = validate_images(dataset_root, test_records)
    counts["test"] = class_counts(test_records, ("test",))["test"]
    print(f"Test class counts (final evaluation only): {counts['test']}")
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    test_loader = DataLoader(
        FamilyImageDataset(test_records, dataset_root, transform, test_paths),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loss, test_metrics = evaluate(model, test_loader, device, criterion)

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "torchvision ResNet18",
        "pretrained_weights": str(weights_path.resolve()),
        "class_names": list(CLASS_NAMES),
        "class_to_index": CLASS_TO_INDEX,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_root": str(dataset_root.resolve()),
        "split_counts": counts,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "class_weighted_loss": True,
        "class_weighted_loss": True,
        "image_size": IMAGE_SIZE,
        "preprocessing": PREPROCESSING,
        "confidence_threshold": 0.70,
        "confidence_threshold_note": "Provisional demonstration threshold, not calibrated.",
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": np.__version__,
        "test_set_used_only_after_validation_model_selection": True,
        "best_epoch": int(best_checkpoint["epoch"]),
        "warning": "Experimental human-reviewed dataset; do not interpret these metrics as production performance.",
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "final_test_metrics.json").write_text(
        json.dumps({"loss": test_loss, **test_metrics}, indent=2), encoding="utf-8"
    )
    return {"output_dir": output_dir, "metadata": metadata, "test_metrics": test_metrics}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")
    try:
        result = train_family_classifier(
            args.manifest, args.dataset_root, seed=args.seed,
            epochs=args.epochs, batch_size=args.batch_size,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Family training stopped safely: {error}\n")
    print(f"Family pilot artifacts saved under: {result['output_dir']}")
    print(f"Held-out test metrics: {result['test_metrics']}")


if __name__ == "__main__":
    main()
