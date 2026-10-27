"""Build a read-only, duplicate-aware paper/cardboard manifest.

Without a review CSV, every image is conservatively marked unreviewed and
excluded. This script never copies or edits source images.
"""

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "Waste_classification_dataset"
    / "final_merged_dataset"
)
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "msw_split_manifest.csv"
CATEGORIES = ("paper", "cardboard")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = frozenset(CATEGORIES)
REVIEW_FIELDS = ("source_relative_path", "decision", "reviewed_label", "reason")
MANIFEST_FIELDS = (
    "source_relative_path",
    "sha256",
    "source_folder",
    "proposed_label",
    "reviewed_label",
    "review_status",
    "duplicate_group_id",
    "split",
    "inclusion_exclusion_reason",
)
SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class ManifestRecord:
    source_relative_path: str
    sha256: str
    source_folder: str
    proposed_label: str
    duplicate_group_id: str
    review_status: str = "unreviewed"
    reviewed_label: str = ""
    review_reason: str = ""
    split: str = ""
    inclusion_exclusion_reason: str = ""


@dataclass(frozen=True)
class GroupDecision:
    status: str
    label: str = ""
    reason: str = ""


def sha256_file(path: Path) -> str:
    """Hash the exact file bytes in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def validate_output_path(dataset_dir: Path, output_path: Path) -> Path:
    """Resolve the manifest path and reject any location inside the source tree."""
    resolved_dataset = dataset_dir.resolve()
    resolved_output = output_path.resolve()
    try:
        resolved_output.relative_to(resolved_dataset)
    except ValueError:
        return resolved_output
    raise ValueError(
        f"Manifest output must be outside the original dataset tree: "
        f"{resolved_output}"
    )


def discover_images(dataset_dir: Path) -> list[ManifestRecord]:
    """Read and hash supported images from the paper and cardboard folders only."""
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    records: list[ManifestRecord] = []
    for category in CATEGORIES:
        category_dir = dataset_dir / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Required category folder not found: {category_dir}")

        try:
            paths = sorted(
                (
                    path
                    for path in category_dir.rglob("*")
                    if path.is_file()
                    and not path.is_symlink()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda path: _relative_posix(path, dataset_dir),
            )
        except OSError as error:
            raise OSError(f"Could not list category folder {category_dir}: {error}") from error

        for path in paths:
            relative_path = _relative_posix(path, dataset_dir)
            record = ManifestRecord(
                source_relative_path=relative_path,
                sha256="",
                source_folder=category,
                proposed_label=category,
                duplicate_group_id="",
            )
            try:
                record.sha256 = sha256_file(path)
                record.duplicate_group_id = f"sha256:{record.sha256}"
            except OSError as error:
                record.review_status = "unreadable"
                record.inclusion_exclusion_reason = (
                    f"Excluded: could not read file to calculate SHA-256: {error}"
                )
            records.append(record)

    return records


def _normalize_review_path(raw_path: str) -> str:
    value = raw_path.strip().replace("\\", "/")
    relative_path = PurePosixPath(value)
    if (
        not value
        or relative_path.is_absolute()
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"Invalid source_relative_path in review CSV: {raw_path!r}")
    return relative_path.as_posix()


def load_review_csv(
    review_path: Path, records: Iterable[ManifestRecord]
) -> dict[str, dict[str, str]]:
    """Read explicit human decisions, rejecting malformed or unknown paths."""
    if not review_path.is_file():
        raise FileNotFoundError(f"Review CSV not found: {review_path}")

    known_paths = {record.source_relative_path for record in records}
    reviews: dict[str, dict[str, str]] = {}
    with review_path.open("r", encoding="utf-8-sig", newline="") as review_file:
        reader = csv.DictReader(review_file)
        missing_fields = set(REVIEW_FIELDS) - set(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Review CSV is missing required columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            # DictReader skips physically blank lines. Also ignore a fully empty
            # delimited row, but never silently accept a partially filled row
            # without its source path.
            if not any(value and value.strip() for value in row.values() if isinstance(value, str)):
                continue
            raw_path = row.get("source_relative_path") or ""
            if not raw_path.strip():
                raise ValueError(
                    f"Review CSV line {line_number} has a blank source_relative_path."
                )
            relative_path = _normalize_review_path(raw_path)
            if relative_path not in known_paths:
                raise ValueError(
                    f"Review CSV line {line_number} references an unknown image path: "
                    f"{relative_path}"
                )
            if relative_path in reviews:
                raise ValueError(
                    f"Review CSV contains duplicate path on line {line_number}: "
                    f"{relative_path}"
                )

            decision = (row.get("decision") or "").strip().lower()
            if decision not in {"include", "hold", "exclude"}:
                raise ValueError(
                    f"Review CSV line {line_number} has invalid decision {decision!r}; "
                    "use include, hold, or exclude."
                )

            reason = (row.get("reason") or "").strip()
            if decision in {"hold", "exclude"} and not reason:
                raise ValueError(
                    f"Review CSV line {line_number} for {relative_path} requires a "
                    f"non-empty reason when decision is {decision!r}."
                )

            reviewed_label = (row.get("reviewed_label") or "").strip()
            if decision == "include" and reviewed_label not in LABELS:
                raise ValueError(
                    f"Review CSV line {line_number} must give an included image "
                    f"the reviewed label paper or cardboard."
                )
            if reviewed_label and reviewed_label not in LABELS:
                raise ValueError(
                    f"Review CSV line {line_number} has unsupported reviewed label "
                    f"{reviewed_label!r}; use paper or cardboard."
                )

            reviews[relative_path] = {
                "decision": decision,
                "reviewed_label": reviewed_label,
                "reason": reason,
            }

    return reviews


def apply_review_records(
    records: list[ManifestRecord], reviews: dict[str, dict[str, str]]
) -> None:
    for record in records:
        review = reviews.get(record.source_relative_path)
        if review is None:
            record.review_status = "unreviewed"
            record.reviewed_label = ""
            record.review_reason = ""
            continue
        record.review_status = {
            "include": "approved",
            "hold": "hold",
            "exclude": "excluded",
        }[review["decision"]]
        record.reviewed_label = review["reviewed_label"]
        record.review_reason = review["reason"]


def decide_hash_groups(
    records: list[ManifestRecord], review_csv_supplied: bool
) -> tuple[dict[str, list[ManifestRecord]], dict[str, GroupDecision]]:
    """Decide each exact-byte group without silently resolving label conflicts."""
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.sha256:
            groups[record.duplicate_group_id].append(record)

    decisions: dict[str, GroupDecision] = {}
    for group_id, members in groups.items():
        members.sort(key=lambda item: item.source_relative_path)
        source_labels = {member.proposed_label for member in members}
        statuses = {member.review_status for member in members}

        if not review_csv_supplied:
            reason = "No human review CSV supplied; excluded conservatively as unreviewed."
            if len(source_labels) > 1:
                reason += " The exact-byte group also has conflicting source folders."
            decisions[group_id] = GroupDecision("unreviewed", reason=reason)
        elif "excluded" in statuses:
            excluded_members = [member for member in members if member.review_status == "excluded"]
            details = "; ".join(
                f"{member.source_relative_path}: {member.review_reason or 'no reason provided'}"
                for member in excluded_members
            )
            decisions[group_id] = GroupDecision(
                "excluded",
                reason=(
                    "At least one exact-duplicate path was explicitly excluded. "
                    f"Excluded path(s): {details}"
                ),
            )
        elif "hold" in statuses:
            held_members = [member for member in members if member.review_status == "hold"]
            details = "; ".join(
                f"{member.source_relative_path}: {member.review_reason or 'no reason provided'}"
                for member in held_members
            )
            decisions[group_id] = GroupDecision(
                "hold",
                reason=(
                    "At least one exact-duplicate path is held for manual review. "
                    f"Held path(s): {details}"
                ),
            )
        elif len(source_labels) > 1:
            all_paths_approved = all(member.review_status == "approved" for member in members)
            reviewed_labels = {member.reviewed_label for member in members}
            if all_paths_approved and len(reviewed_labels) == 1:
                decisions[group_id] = GroupDecision(
                    "approved",
                    label=next(iter(reviewed_labels)),
                    reason="All paths in the cross-folder exact-byte group were reviewed "
                    "and assigned one consistent label.",
                )
            else:
                decisions[group_id] = GroupDecision(
                    "hold",
                    reason="Exact bytes occur in both source folders; every path must be "
                    "explicitly reviewed with one consistent label.",
                )
        else:
            approved_labels = {
                member.reviewed_label
                for member in members
                if member.review_status == "approved"
            }
            if len(approved_labels) > 1:
                decisions[group_id] = GroupDecision(
                    "hold",
                    reason="Exact-duplicate review records assign conflicting labels.",
                )
            elif approved_labels:
                decisions[group_id] = GroupDecision(
                    "approved",
                    label=next(iter(approved_labels)),
                    reason="At least one exact-byte-identical path was explicitly reviewed.",
                )
            else:
                decisions[group_id] = GroupDecision(
                    "unreviewed",
                    reason="No approved human review record exists for this exact-byte group.",
                )

    return groups, decisions


def _split_counts(group_count: int) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("At least three unique approved groups per class are required.")

    targets = (0.70 * group_count, 0.15 * group_count, 0.15 * group_count)
    counts = [1, 1, 1]
    for _ in range(group_count - 3):
        next_split = max(
            range(3),
            key=lambda index: (targets[index] - counts[index], -index),
        )
        counts[next_split] += 1
    return counts[0], counts[1], counts[2]


def assign_splits(
    groups: dict[str, list[ManifestRecord]],
    decisions: dict[str, GroupDecision],
    seed: int,
) -> None:
    """Assign approved hash groups by label using a seeded 70/15/15 split."""
    groups_by_label: dict[str, list[str]] = {label: [] for label in CATEGORIES}
    for group_id, decision in decisions.items():
        if decision.status == "approved":
            groups_by_label[decision.label].append(group_id)

    counts_by_label = {
        label: len(group_ids) for label, group_ids in groups_by_label.items()
    }
    insufficient = {
        label: count for label, count in counts_by_label.items() if count < 3
    }
    if insufficient:
        details = ", ".join(
            f"{label}: {count}" for label, count in insufficient.items()
        )
        raise ValueError(
            "Cannot create train/validation/test splits; need at least three "
            f"unique approved hash groups per class ({details})."
        )

    rng = random.Random(seed)
    for label in CATEGORIES:
        group_ids = sorted(
            groups_by_label[label],
            key=lambda group_id: groups[group_id][0].source_relative_path,
        )
        rng.shuffle(group_ids)
        train_count, validation_count, _ = _split_counts(len(group_ids))
        for index, group_id in enumerate(group_ids):
            split = (
                "train"
                if index < train_count
                else "validation"
                if index < train_count + validation_count
                else "test"
            )
            for member in groups[group_id]:
                member.split = split


def finalize_manifest_rows(
    records: list[ManifestRecord],
    groups: dict[str, list[ManifestRecord]],
    decisions: dict[str, GroupDecision],
    review_csv_supplied: bool,
) -> None:
    for record in records:
        if not record.sha256:
            if not record.inclusion_exclusion_reason:
                record.inclusion_exclusion_reason = "Excluded: image bytes were unreadable."
            continue

        group = groups[record.duplicate_group_id]
        decision = decisions[record.duplicate_group_id]
        canonical = group[0]

        if decision.status == "approved":
            if record.review_status == "unreviewed":
                record.review_status = "approved_via_exact_duplicate"
            if not record.reviewed_label:
                record.reviewed_label = decision.label
            if record is canonical:
                record.inclusion_exclusion_reason = (
                    f"Included as canonical path. {decision.reason}"
                )
            else:
                record.inclusion_exclusion_reason = (
                    "Excluded duplicate copy; canonical path is "
                    f"{canonical.source_relative_path}. All exact-byte copies retain "
                    "the group's split to prevent leakage."
                )
        elif decision.status == "hold":
            record.inclusion_exclusion_reason = f"Held: {decision.reason}"
            if record.review_status == "unreviewed" and review_csv_supplied:
                record.review_status = "held_by_duplicate_group"
        elif decision.status == "excluded":
            record.inclusion_exclusion_reason = f"Excluded: {decision.reason}"
        else:
            record.inclusion_exclusion_reason = f"Excluded conservatively: {decision.reason}"


def write_manifest(output_path: Path, records: list[ManifestRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.source_relative_path):
            writer.writerow(
                {
                    "source_relative_path": record.source_relative_path,
                    "sha256": record.sha256,
                    "source_folder": record.source_folder,
                    "proposed_label": record.proposed_label,
                    "reviewed_label": record.reviewed_label,
                    "review_status": record.review_status,
                    "duplicate_group_id": record.duplicate_group_id,
                    "split": record.split,
                    "inclusion_exclusion_reason": record.inclusion_exclusion_reason,
                }
            )


def prepare_manifest(
    dataset_dir: Path,
    output_path: Path,
    review_csv: Path | None = None,
    seed: int = 42,
) -> list[ManifestRecord]:
    """Create an audit/split CSV while leaving the source tree unchanged."""
    output_path = validate_output_path(dataset_dir, output_path)
    if review_csv is not None and output_path == review_csv.resolve():
        raise ValueError(
            f"Manifest output and review CSV refer to the same file: {output_path}"
        )
    records = discover_images(dataset_dir)

    if review_csv is not None:
        reviews = load_review_csv(review_csv, records)
        apply_review_records(records, reviews)

    groups, decisions = decide_hash_groups(records, review_csv_supplied=review_csv is not None)
    if review_csv is not None:
        assign_splits(groups, decisions, seed)
    finalize_manifest_rows(
        records,
        groups,
        decisions,
        review_csv_supplied=review_csv is not None,
    )
    write_manifest(output_path, records)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a read-only, exact-duplicate-aware paper/cardboard manifest. "
            "Images are never copied or modified."
        )
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        help=(
            "optional CSV with source_relative_path, decision, reviewed_label, reason; "
            "only explicitly approved items are eligible for splits"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"manifest path outside the MWCD source tree (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducible group-level splits (default: 42)",
    )
    args = parser.parse_args(argv)

    try:
        output_path = validate_output_path(DATASET_DIR, args.output)
        records = prepare_manifest(
            dataset_dir=DATASET_DIR,
            output_path=output_path,
            review_csv=args.review_csv,
            seed=args.seed,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))

    if args.review_csv is None:
        print(
            "No review CSV supplied: all readable images are marked unreviewed "
            "and excluded; no split is assigned."
        )
    included = sum(
        record.inclusion_exclusion_reason.startswith("Included as canonical path")
        for record in records
    )
    print(f"Manifest written to: {output_path}")
    print(f"Source images inventoried: {len(records)}")
    print(f"Approved canonical images included: {included}")
    if args.review_csv is not None:
        for split in SPLIT_NAMES:
            print(
                f"{split.capitalize()} included images: "
                f"{sum(record.split == split and record.inclusion_exclusion_reason.startswith('Included as canonical path') for record in records)}"
            )


if __name__ == "__main__":
    main()
