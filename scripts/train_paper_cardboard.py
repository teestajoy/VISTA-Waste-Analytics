"""Train a small ResNet18 paper/cardboard pilot from a reviewed manifest.

This is an experiment on a very small dataset, not a claim of reliable
real-world waste classification. It never downloads pretrained weights.
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
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "paper_cardboard_pilot_split_manifest.csv"
DEFAULT_DATASET_ROOT = (
    Path.home() / "Downloads" / "Waste_classification_dataset" / "final_merged_dataset"
)
OUTPUT_DIR = PROJECT_ROOT / "models" / "paper_cardboard_resnet18_pilot"
CLASS_NAMES = ("paper", "cardboard")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "validation", "test")
INCLUDED_STATUSES = {"approved", "approved_via_exact_duplicate"}
CANONICAL_REASON = "Included as canonical path."
REQUIRED_COLUMNS = {
    "source_relative_path",
    "sha256",
    "source_folder",
    "proposed_label",
    "reviewed_label",
    "review_status",
    "duplicate_group_id",
    "split",
    "inclusion_exclusion_reason",
}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DatasetValidationError(ValueError):
    """Raised when the reviewed manifest or an included image is unsafe/invalid."""


@dataclass(frozen=True)
class ManifestImage:
    source_relative_path: str
    sha256: str
    label: str
    split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(source_relative_path: str) -> PurePosixPath:
    """Validate manifest path syntax without accessing the referenced file."""
    value = source_relative_path.strip().replace("\\", "/")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(source_relative_path.strip())
    if (
        not value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise DatasetValidationError(
            f"Unsafe source_relative_path in manifest: {source_relative_path!r}"
        )
    return posix_path


def _safe_source_path(dataset_root: Path, source_relative_path: str) -> Path:
    """Resolve a manifest path while rejecting traversal and escaping symlinks."""
    relative_path = _validate_relative_path(source_relative_path)

    root = dataset_root.resolve()
    candidate = root.joinpath(*relative_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise DatasetValidationError(
            f"Image path is missing or resolves outside the dataset root: "
            f"{source_relative_path!r}"
        ) from error
    if not resolved.is_file():
        raise DatasetValidationError(
            f"Manifest path is not a file: {source_relative_path!r}"
        )
    return resolved


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[ManifestImage]]:
    """Read the manifest and select only rows explicitly marked canonical-included."""
    if not path.is_file():
        raise FileNotFoundError(f"Pilot manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise DatasetValidationError(
                "Pilot manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        rows = [dict(row) for row in reader]

    # Check split leakage against every hashed manifest row, including duplicate copies.
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(rows, start=2):
        split = (row.get("split") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        if split and split not in SPLITS:
            raise DatasetValidationError(
                f"Invalid split {split!r} on manifest line {line_number}."
            )
        if split and digest:
            hash_splits[digest].add(split)
    leaked = [digest for digest, splits in hash_splits.items() if len(splits) > 1]
    if leaked:
        raise DatasetValidationError(
            f"Exact-duplicate hash leakage found across splits ({len(leaked)} hash group(s))."
        )

    included: list[ManifestImage] = []
    seen_hashes: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        reason = (row.get("inclusion_exclusion_reason") or "").strip()
        if not reason.startswith(CANONICAL_REASON):
            continue
        split = (row.get("split") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        relative_path = (row.get("source_relative_path") or "").strip()
        if split not in SPLITS:
            raise DatasetValidationError(
                f"Included image {relative_path!r} has invalid or missing split {split!r}."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise DatasetValidationError(
                f"Included image {relative_path!r} has an invalid SHA-256 value."
            )
        if not relative_path:
            raise DatasetValidationError(
                f"Included row on manifest line {line_number} has a blank source path."
            )
        if digest in seen_hashes:
            raise DatasetValidationError(
                f"Duplicate canonical SHA-256 appears more than once: {relative_path!r}."
            )
        seen_hashes.add(digest)

        # Test records remain opaque until validation-based model selection ends.
        if split == "test":
            continue
        status = (row.get("review_status") or "").strip()
        label = (row.get("reviewed_label") or "").strip()
        if status not in INCLUDED_STATUSES:
            raise DatasetValidationError(
                f"Canonical included row on line {line_number} has non-approved "
                f"review_status {status!r}."
            )
        if label not in CLASS_TO_INDEX:
            raise DatasetValidationError(
                f"Invalid reviewed_label {label!r} on manifest line {line_number}."
            )
        included.append(ManifestImage(relative_path, digest, label, split))

    if not included:
        raise DatasetValidationError(
            "The pilot manifest contains no canonical training or validation images."
        )
    return rows, included


def class_counts(
    images: Iterable[ManifestImage], splits: tuple[str, ...] = ("train", "validation")
) -> dict[str, dict[str, int]]:
    counts = {split: Counter({label: 0 for label in CLASS_NAMES}) for split in splits}
    for image in images:
        if image.split in counts:
            counts[image.split][image.label] += 1
    return {split: dict(counts[split]) for split in splits}


def validate_split_class_counts(images: list[ManifestImage]) -> dict[str, dict[str, int]]:
    splits = ("train", "validation")
    counts = class_counts(images, splits)
    missing = [f"{split}/{label}" for split in splits for label, count in counts[split].items() if count == 0]
    if missing:
        raise DatasetValidationError(
            "Training and validation must each contain both classes; missing: "
            + ", ".join(missing)
        )
    return counts


def validate_test_records(rows: list[dict[str, str]]) -> list[ManifestImage]:
    """Inspect and validate held-out labels only after checkpoint selection."""
    test_records: list[ManifestImage] = []
    for line_number, row in enumerate(rows, start=2):
        reason = (row.get("inclusion_exclusion_reason") or "").strip()
        if not reason.startswith(CANONICAL_REASON):
            continue
        split = (row.get("split") or "").strip()
        if split != "test":
            continue
        relative_path = (row.get("source_relative_path") or "").strip()
        status = (row.get("review_status") or "").strip()
        label = (row.get("reviewed_label") or "").strip()
        digest = (row.get("sha256") or "").strip().lower()
        if status not in INCLUDED_STATUSES:
            raise DatasetValidationError(
                f"Canonical test row on manifest line {line_number} has non-approved "
                f"review_status {status!r}."
            )
        if label not in CLASS_TO_INDEX:
            raise DatasetValidationError(
                f"Invalid test reviewed_label {label!r} on manifest line {line_number}."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise DatasetValidationError(
                f"Test image {relative_path!r} has an invalid SHA-256 value."
            )
        if not relative_path:
            raise DatasetValidationError(
                f"Canonical test row on manifest line {line_number} has a blank source path."
            )
        test_records.append(ManifestImage(relative_path, digest, label, split))

    if not test_records:
        raise DatasetValidationError(
            "After checkpoint selection, final evaluation cannot proceed: "
            "the test split has no canonical included images."
        )
    counts = class_counts(test_records, ("test",))["test"]
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        raise DatasetValidationError(
            "After checkpoint selection, final evaluation requires both classes "
            "in the test split; missing: " + ", ".join(missing)
        )
    return test_records


def validate_image_file(dataset_root: Path, image: ManifestImage) -> Path:
    """Check an included image's safe path, current hash, and Pillow readability."""
    path = _safe_source_path(dataset_root, image.source_relative_path)
    try:
        actual_hash = sha256_file(path)
    except OSError as error:
        raise DatasetValidationError(
            f"Could not read image {image.source_relative_path!r} for hashing: {error}"
        ) from error
    if actual_hash != image.sha256:
        raise DatasetValidationError(
            f"SHA-256 mismatch for {image.source_relative_path!r}; the source may have changed."
        )
    try:
        with Image.open(path) as image_file:
            image_file.verify()
        with Image.open(path) as image_file:
            image_file.load()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise DatasetValidationError(
            f"Image is unreadable or corrupt: {image.source_relative_path!r}: {error}"
        ) from error
    return path


def validate_images(
    dataset_root: Path, images: Iterable[ManifestImage]
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for image in images:
        paths[image.source_relative_path] = validate_image_file(dataset_root, image)
    return paths


class PilotImageDataset(Dataset):
    def __init__(
        self,
        images: list[ManifestImage],
        dataset_root: Path,
        transform: Any,
        validated_paths: dict[str, Path] | None = None,
    ) -> None:
        self.images = images
        self.dataset_root = dataset_root
        self.transform = transform
        self.validated_paths = validated_paths or {}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.images[index]
        path = self.validated_paths.get(record.source_relative_path)
        if path is None:
            path = _safe_source_path(self.dataset_root, record.source_relative_path)
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.load()
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise DatasetValidationError(
                f"Could not load image {record.source_relative_path!r}: {error}"
            ) from error
        return self.transform(image), CLASS_TO_INDEX[record.label]


def resolve_cached_weights(cache_dir: Path | None = None) -> Path:
    """Return the official ResNet18 weights only when already present locally."""
    filename = ResNet18_Weights.DEFAULT.url.rsplit("/", 1)[-1]
    resolved_cache = cache_dir or Path(torch.hub.get_dir()) / "checkpoints"
    weights_path = resolved_cache / filename
    if not weights_path.is_file():
        raise FileNotFoundError(
            "ImageNet-pretrained ResNet18 weights are not cached locally. "
            f"Expected: {weights_path}. No download was attempted."
        )
    return weights_path


def build_model(weights_path: Path | None = None) -> nn.Module:
    """Build a two-class ResNet18; production calls provide a cached weights file."""
    model = models.resnet18(weights=None)
    if weights_path is not None:
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
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


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            for actual, predicted in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual, predicted] += 1
            total += targets.size(0)
    metrics = _metrics_from_confusion(confusion)
    return total_loss / max(total, 1), metrics


def _metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, label in enumerate(CLASS_NAMES):
        true_positive = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
    total = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def train_pilot(
    manifest_path: Path = DEFAULT_MANIFEST,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path | None = None,
    *,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    if output_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError(f"Training outputs must be written only to {OUTPUT_DIR}.")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite it: {output_dir}"
        )
    manifest_rows, included = read_manifest(manifest_path)
    counts = validate_split_class_counts(included)
    for split in counts:
        print(f"{split.capitalize()} class counts: {counts[split]}")
    print(
        "Small pilot warning: 36 reviewed canonical images are not enough to "
        "establish reliable real-world classification performance."
    )
    weights_path = resolve_cached_weights()

    # Validate train and validation before optimization. Test paths are checked for
    # split leakage from manifest hashes, but test labels, paths, and bytes are
    # deliberately not examined until validation-based model selection is finished.
    train_records = [record for record in included if record.split == "train"]
    validation_records = [record for record in included if record.split == "validation"]
    train_paths = validate_images(dataset_root, train_records)
    validation_paths = validate_images(dataset_root, validation_records)

    set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_loader = DataLoader(
        PilotImageDataset(train_records, dataset_root, transform, train_paths),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_loader = DataLoader(
        PilotImageDataset(validation_records, dataset_root, transform, validation_paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = build_model(weights_path).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create the fixed run directory only after all configuration and preflight
    # checks pass; exist_ok=False protects previous results.
    output_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, float | int]] = []
    best_key: tuple[float, float] | None = None
    best_checkpoint = output_dir / "best_model.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * targets.size(0)
            train_total += targets.size(0)
        validation_loss, validation_metrics = _evaluate(
            model, validation_loader, device, criterion
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_total, 1),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(epoch_row)
        key = (validation_metrics["macro_f1"], -validation_loss)
        if best_key is None or key > best_key:
            best_key = key
            torch.save(
                {"state_dict": model.state_dict(), "class_names": CLASS_NAMES, "epoch": epoch},
                best_checkpoint,
            )

    # Model selection is complete. Only now inspect test labels and validate/open
    # the held-out test files.
    test_records = validate_test_records(manifest_rows)
    test_paths = validate_images(dataset_root, test_records)
    counts.update(class_counts(test_records, ("test",)))
    print(f"Test class counts (final evaluation only): {counts['test']}")
    best_state = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best_state["state_dict"])
    test_loader = DataLoader(
        PilotImageDataset(test_records, dataset_root, transform, test_paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loss, test_metrics = _evaluate(model, test_loader, device, criterion)

    with (output_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "torchvision ResNet18",
        "pretrained_weights": str(weights_path),
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
        "image_size": IMAGE_SIZE,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": np.__version__,
        "small_dataset_warning": "36 reviewed canonical images; metrics are highly uncertain.",
        "test_set_used_only_after_validation_model_selection": True,
        "best_epoch": int(best_state["epoch"]),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
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
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")
    try:
        result = train_pilot(
            args.manifest,
            args.dataset_root,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except (DatasetValidationError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"Training stopped safely: {error}\n")
    print(f"Pilot model and reports saved under: {result['output_dir']}")
    print(f"Final held-out test metrics: {result['test_metrics']}")


if __name__ == "__main__":
    main()
