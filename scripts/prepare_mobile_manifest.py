"""Validate reviewed mobile-phone image records and create a fixed pilot split.

This workflow never infers labels from folders or model proposals. Only rows
with complete, explicit human category and family decisions are eligible.
Images are read-only; outputs use exclusive creation and a new path.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import zlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import family_dataset_workflow as workflow  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "mobile_phone_review_batch_v1.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "mobile_phone_split_manifest_v1.csv"
DEFAULT_PROTECTED_MANIFESTS = (
    PROJECT_ROOT / "data" / "processed" / "family_provisional_split_manifest.csv",
    PROJECT_ROOT / "models" / "waste_category_provisional" / "split_manifest.csv",
    PROJECT_ROOT / "data" / "processed" / "msw_split_manifest.csv",
    PROJECT_ROOT / "data" / "processed" / "paper_cardboard_pilot_split_manifest.csv",
)
DEFAULT_UNRESOLVED_REVIEW = PROJECT_ROOT / "docs" / "category_disagreement_re_review_v1.csv"

SPLITS = ("train", "validation", "test")
SPLIT_DEVICE_COUNTS = {"train": 30, "validation": 10, "test": 20}
REQUIRED_FIELDS = (
    "source_relative_path", "sha256", "proposed_category", "proposed_family",
    "reviewed_category", "reviewed_family", "category_decision", "family_decision",
    "review_status", "reviewer_id", "reviewed_utc", "physical_device_group_id",
    "capture_session_id", "image_provenance", "duplicate_group_id",
    "near_duplicate_group_id", "split", "exclusion_reason",
)
DECISIONS = frozenset({"include", "hold", "exclude"})
STATUSES = frozenset({"pending", "complete", "held", "excluded"})
VALID_REVIEWED_CATEGORIES = frozenset({
    "paper", "cardboard", "plastic", "glass", "organic", "battery", "mobile",
})
VALID_REVIEWED_FAMILIES = frozenset({"MSW", "E_WASTE"})


class MobileManifestError(ValueError):
    """Raised when review data or a proposed split is unsafe."""


class InsufficientDeviceGroupsError(MobileManifestError):
    """Raised when the fixed pilot split cannot be populated without guessing."""


class ExistingSplitRecords:
    """Path/hash references and split assignments read from protected manifests."""

    def __init__(self) -> None:
        self.path_splits: dict[str, set[str]] = defaultdict(set)
        self.hash_splits: dict[str, set[str]] = defaultdict(set)
        self.path_hashes: dict[str, set[str]] = defaultdict(set)


def merge_protected_manifests(
    additional_paths: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Always retain defaults, append custom sources, and deduplicate resolved paths."""
    merged: list[Path] = []
    seen: set[str] = set()
    for path in (*DEFAULT_PROTECTED_MANIFESTS, *additional_paths):
        resolved = Path(path).resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            merged.append(resolved)
    return tuple(merged)


def read_csv(path: Path, required_fields: tuple[str, ...] = REQUIRED_FIELDS) -> list[dict[str, str]]:
    """Read a CSV while requiring the complete versioned schema, even if empty."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Mobile review CSV not found: {path}")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required_fields) - set(reader.fieldnames or ())
        if missing:
            raise MobileManifestError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        return [{field: (row.get(field) or "").strip() for field in required_fields} for row in reader]


def load_existing_split_records(paths: tuple[Path, ...]) -> ExistingSplitRecords:
    """Index all known split records and fail on ambiguous split declarations."""
    records = ExistingSplitRecords()
    for manifest_path in paths:
        if not Path(manifest_path).is_file():
            raise FileNotFoundError(f"Protected split manifest not found: {manifest_path}")
        with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            if not {"source_relative_path", "split"}.issubset(fields):
                raise MobileManifestError(
                    f"Protected manifest lacks source_relative_path or split: {manifest_path}"
                )
            for line, row in enumerate(reader, start=2):
                relative = (row.get("source_relative_path") or "").strip()
                split = (row.get("split") or "").strip().lower()
                if not relative:
                    if split:
                        raise MobileManifestError(
                            f"Protected manifest has a split without an image path at {manifest_path}:{line}."
                        )
                    continue
                normalized = workflow.normalize_relative_path(relative).casefold()
                if split and split not in SPLITS:
                    raise MobileManifestError(
                        f"Cannot establish existing split {split!r} in {manifest_path}:{line}."
                    )
                group_status = (row.get("group_status") or "").strip().lower()
                if not split and group_status in {"include", "included", "provisional_include"}:
                    raise MobileManifestError(
                        f"Included record has no split in {manifest_path}:{line}; cannot establish split safely."
                    )
                digest = (row.get("sha256") or "").strip().lower()
                if digest:
                    if not _valid_sha256(digest):
                        raise MobileManifestError(
                            f"Invalid SHA-256 in protected test manifest {manifest_path}: {relative}"
                        )
                    records.path_hashes[normalized].add(digest)
                    if split:
                        records.hash_splits[digest].add(split)
                if split:
                    records.path_splits[normalized].add(split)
    return records


def load_unresolved_records(path: Path) -> tuple[set[str], set[str]]:
    """Load paths and hashes that must remain untouched while their review is pending."""
    rows = read_csv(path, ("source_relative_path", "sha256", "review_status"))
    paths: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        if row["review_status"].lower() != "pending":
            continue
        relative = workflow.normalize_relative_path(row["source_relative_path"])
        digest = row["sha256"].lower()
        if not _valid_sha256(digest):
            raise MobileManifestError(f"Invalid SHA-256 in unresolved review record: {relative}")
        paths.add(relative)
        hashes.add(digest)
    return paths, hashes


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _validate_review_state(row: dict[str, str], line: int) -> bool:
    """Validate review fields; return whether this row is explicitly eligible."""
    status = row["review_status"].lower()
    category_decision = row["category_decision"].lower()
    family_decision = row["family_decision"].lower()
    reviewed_category = row["reviewed_category"]
    reviewed_family = row["reviewed_family"]
    reviewer = row["reviewer_id"]
    reviewed_utc = row["reviewed_utc"]

    if row["proposed_category"] != "mobile" or row["proposed_family"] != "E_WASTE":
        raise MobileManifestError(f"Row {line}: proposed labels must be mobile / E_WASTE.")
    if status not in STATUSES:
        raise MobileManifestError(f"Row {line}: invalid review_status {status!r}.")
    if status == "pending":
        if any((category_decision, family_decision, reviewed_category, reviewed_family, reviewer, reviewed_utc)):
            raise MobileManifestError(
                f"Row {line}: pending rows cannot contain partial reviewer decisions or labels."
            )
        return False

    if reviewed_category and reviewed_category not in VALID_REVIEWED_CATEGORIES:
        raise MobileManifestError(f"Row {line}: invalid reviewed category {reviewed_category!r}.")
    if reviewed_family and reviewed_family not in VALID_REVIEWED_FAMILIES:
        raise MobileManifestError(f"Row {line}: invalid reviewed family {reviewed_family!r}.")
    if category_decision not in DECISIONS or family_decision not in DECISIONS:
        raise MobileManifestError(f"Row {line}: complete review decisions are required for both tasks.")
    if not reviewer or not reviewed_utc:
        raise MobileManifestError(f"Row {line}: reviewer_id and reviewed_utc are required for reviewed rows.")
    try:
        review_time = datetime.fromisoformat(reviewed_utc.replace("Z", "+00:00"))
    except ValueError as error:
        raise MobileManifestError(f"Row {line}: reviewed_utc must be an ISO-8601 timestamp.") from error
    if review_time.tzinfo is None:
        raise MobileManifestError(f"Row {line}: reviewed_utc must include a timezone.")

    if status == "complete":
        if (category_decision, family_decision) != ("include", "include"):
            raise MobileManifestError(
                f"Row {line}: complete rows require include decisions for category and family."
            )
        if reviewed_category != "mobile" or reviewed_family != "E_WASTE":
            raise MobileManifestError(f"Row {line}: reviewed labels must be mobile / E_WASTE.")
        if not row["physical_device_group_id"]:
            raise MobileManifestError(f"Row {line}: physical_device_group_id is required for included data.")
        if not row["image_provenance"]:
            raise MobileManifestError(f"Row {line}: image_provenance is required for included data.")
        return True

    if status == "held":
        if "hold" not in {category_decision, family_decision}:
            raise MobileManifestError(f"Row {line}: held rows require a hold decision.")
        if row["split"]:
            raise MobileManifestError(f"Row {line}: held rows cannot have a split assignment.")
        return False

    if "exclude" not in {category_decision, family_decision}:
        raise MobileManifestError(f"Row {line}: excluded rows require at least one exclude decision.")
    if row["split"]:
        raise MobileManifestError(f"Row {line}: excluded rows cannot have a split assignment.")
    return False


def validate_intake_rows(
    rows: list[dict[str, str]],
    dataset_root: Path,
    *,
    existing_splits: ExistingSplitRecords,
    unresolved_paths: set[str],
    unresolved_hashes: set[str],
) -> list[dict[str, str]]:
    """Validate the intake, verify images/hashes, and keep non-included rows unsplit."""
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    result = []
    for line, original in enumerate(rows, start=2):
        row = {field: (original.get(field) or "").strip() for field in REQUIRED_FIELDS}
        try:
            relative = workflow.normalize_relative_path(row["source_relative_path"])
        except ValueError as error:
            raise MobileManifestError(f"Row {line}: {error}") from error
        digest = row["sha256"].lower()
        if not _valid_sha256(digest):
            raise MobileManifestError(f"Row {line}: invalid SHA-256 for {relative}.")
        if relative in seen_paths:
            raise MobileManifestError(f"Row {line}: duplicate image path {relative}.")
        if digest in seen_hashes:
            raise MobileManifestError(f"Row {line}: duplicate image hash {digest}.")
        seen_paths.add(relative)
        seen_hashes.add(digest)
        row["source_relative_path"] = relative
        row["sha256"] = digest
        if row["split"]:
            raise MobileManifestError(
                f"Row {line}: intake split must be blank; splits are assigned only after complete review."
            )
        if row["exclusion_reason"]:
            raise MobileManifestError(f"Row {line}: exclusion_reason is generated by this workflow and must be blank in intake.")
        if relative in unresolved_paths or digest in unresolved_hashes:
            raise MobileManifestError(
                f"Row {line}: unresolved disagreement records cannot be reused as mobile data: {relative}."
            )

        eligible = _validate_review_state(row, line)
        path_key = relative.casefold()
        path_hashes = existing_splits.path_hashes.get(path_key, set())
        if path_hashes and digest not in path_hashes:
            raise MobileManifestError(
                f"Row {line}: existing manifest path has a different image hash; cannot establish split safely: {relative}."
            )
        known_splits = (
            existing_splits.path_splits.get(path_key, set())
            | existing_splits.hash_splits.get(digest, set())
        )
        if len(known_splits) > 1:
            raise MobileManifestError(
                f"Row {line}: existing manifests assign this path/hash to conflicting splits: {relative}."
            )
        prior_split = next(iter(known_splits), "")
        if eligible and prior_split in {"train", "validation"}:
            raise MobileManifestError(
                f"Row {line}: image already belongs to an existing {prior_split} split and cannot be reused: {relative}."
            )
        if eligible and prior_split == "test":
            row["_existing_test_only"] = "true"
        else:
            row["_existing_test_only"] = "false"

        try:
            image_path = workflow.safe_source_path(dataset_root, relative)
            if not image_path.is_file():
                raise MobileManifestError(f"Row {line}: image does not exist: {relative}.")
            if workflow.sha256_file(image_path) != digest:
                raise MobileManifestError(f"Row {line}: SHA-256 does not match the source image: {relative}.")
            workflow._verify_image(image_path)
        except MobileManifestError:
            raise
        except (OSError, ValueError, RuntimeError) as error:
            raise MobileManifestError(f"Row {line}: unsafe or unreadable image {relative}: {error}") from error

        if row["duplicate_group_id"] and row["duplicate_group_id"] != f"sha256:{digest}":
            raise MobileManifestError(f"Row {line}: duplicate_group_id must be sha256:<image hash> when provided.")
        if not row["duplicate_group_id"]:
            row["duplicate_group_id"] = f"sha256:{digest}"
        result.append(row)
    return result


def assign_fixed_splits(rows: list[dict[str, str]], *, seed: int = 42) -> dict[str, object]:
    """Assign exactly 30/10/20 physical-device groups; never split pending rows."""
    included = [row for row in rows if row["review_status"] == "complete"]
    records_by_device: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in included:
        records_by_device[row["physical_device_group_id"]].append(row)
    device_ids = sorted(records_by_device)
    required = sum(SPLIT_DEVICE_COUNTS.values())
    if len(device_ids) < required:
        shortfall = required - len(device_ids)
        raise InsufficientDeviceGroupsError(
            f"Insufficient reviewed phone data: {len(device_ids)} distinct eligible physical devices; "
            f"the fixed 30/10/20 pilot split requires {required} (short by {shortfall}). "
            "No split manifest was written. Collect and review more distinct devices; no rows were fabricated."
        )

    protected_devices = {
        device for device, members in records_by_device.items()
        if any(member.get("_existing_test_only", "false") == "true" for member in members)
    }
    test_limit = SPLIT_DEVICE_COUNTS["test"]
    if len(protected_devices) > test_limit:
        raise InsufficientDeviceGroupsError(
            f"{len(protected_devices)} physical-device groups overlap existing held-out test data, "
            f"exceeding the fixed {test_limit}-device test allocation. No split manifest was written."
        )

    stable_seed = (seed + zlib.crc32(b"mobile-phone")) & 0xFFFFFFFF
    shuffled = device_ids[:]
    random.Random(stable_seed).shuffle(shuffled)
    test_devices = set(protected_devices)
    for device in shuffled:
        if len(test_devices) == test_limit:
            break
        test_devices.add(device)
    remaining = [device for device in shuffled if device not in test_devices]
    assigned = {
        device: "test" for device in test_devices
    }
    assigned.update({device: "train" for device in remaining[:SPLIT_DEVICE_COUNTS["train"]]})
    assigned.update({
        device: "validation"
        for device in remaining[
            SPLIT_DEVICE_COUNTS["train"]:
            SPLIT_DEVICE_COUNTS["train"] + SPLIT_DEVICE_COUNTS["validation"]
        ]
    })

    for row in rows:
        row.pop("_existing_test_only", None)
        row["split"] = ""
        row["exclusion_reason"] = ""
        if row["review_status"] == "complete":
            device = row["physical_device_group_id"]
            row["split"] = assigned.get(device, "")
            if not row["split"]:
                row["exclusion_reason"] = "outside the fixed 30/10/20 device pilot allocation"
        elif row["review_status"] == "pending":
            row["exclusion_reason"] = "pending review; no split assigned"
        elif row["review_status"] == "held":
            row["exclusion_reason"] = "held by human review; no split assigned"
        else:
            row["exclusion_reason"] = "excluded by human review; no split assigned"

    validate_prepared_manifest(rows)
    return {
        "eligible_devices": len(device_ids),
        "selected_devices": len(assigned),
        "split_device_counts": {
            split: sum(value == split for value in assigned.values()) for split in SPLITS
        },
        "unselected_devices": len(device_ids) - len(assigned),
        "reviewed_images": sum(bool(row["split"]) for row in rows),
    }


def validate_prepared_manifest(rows: list[dict[str, str]]) -> None:
    """Reject split leakage by device or near-duplicate group."""
    device_splits: dict[str, set[str]] = defaultdict(set)
    near_duplicate_splits: dict[str, set[str]] = defaultdict(set)
    hashes: set[str] = set()
    paths: set[str] = set()
    for line, row in enumerate(rows, start=2):
        relative = workflow.normalize_relative_path(row["source_relative_path"])
        digest = row["sha256"].lower()
        if not _valid_sha256(digest):
            raise MobileManifestError(f"Row {line}: invalid SHA-256 for {relative}.")
        if relative in paths:
            raise MobileManifestError(f"Row {line}: duplicate image path {relative}.")
        if digest in hashes:
            raise MobileManifestError(f"Row {line}: duplicate image hash {digest}.")
        paths.add(relative)
        hashes.add(digest)
        split = row["split"]
        if split and split not in SPLITS:
            raise MobileManifestError(f"Row {line}: invalid split {split!r}.")
        if split:
            if row["review_status"] != "complete":
                raise MobileManifestError(f"Row {line}: only fully reviewed records may have a split.")
            device = row["physical_device_group_id"]
            if not device:
                raise MobileManifestError(f"Row {line}: split record has no physical-device group ID.")
            device_splits[device].add(split)
            near_group = row["near_duplicate_group_id"]
            if near_group:
                near_duplicate_splits[near_group].add(split)
        elif row["review_status"] != "complete" and not row["exclusion_reason"]:
            raise MobileManifestError(f"Row {line}: unsplit intake record needs an exclusion reason.")

    leaked_devices = sorted(group for group, splits in device_splits.items() if len(splits) > 1)
    if leaked_devices:
        raise MobileManifestError(
            "Physical-device groups cross splits: " + ", ".join(leaked_devices[:5])
        )
    leaked_near = sorted(group for group, splits in near_duplicate_splits.items() if len(splits) > 1)
    if leaked_near:
        raise MobileManifestError(
            "Near-duplicate groups cross splits: " + ", ".join(leaked_near[:5])
        )


def write_manifest_new(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write a manifest with exclusive creation; refuse every existing path."""
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {destination}. "
            "Choose a new versioned output path, such as mobile_phone_split_manifest_v2.csv."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REQUIRED_FIELDS})
    return destination


def prepare_manifest(
    input_path: Path,
    dataset_root: Path,
    output_path: Path,
    *,
    protected_manifests: tuple[Path, ...] = (),
    unresolved_review_path: Path = DEFAULT_UNRESOLVED_REVIEW,
    seed: int = 42,
) -> dict[str, object]:
    """Validate inputs, enforce test protection, allocate groups, then write once."""
    input_resolved = Path(input_path).resolve()
    output_resolved = Path(output_path).resolve()
    if output_resolved == input_resolved or output_resolved == Path(unresolved_review_path).resolve():
        raise MobileManifestError("Output path must differ from every input/review file.")
    if output_resolved.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_resolved}. "
            "Choose a new versioned output path, such as mobile_phone_split_manifest_v2.csv."
        )
    root_resolved = Path(dataset_root).resolve()
    if output_resolved.is_relative_to(root_resolved):
        raise MobileManifestError(f"Manifest output must be outside the source dataset: {output_resolved}")

    rows = read_csv(input_path)
    all_protected_manifests = merge_protected_manifests(protected_manifests)
    existing_splits = load_existing_split_records(all_protected_manifests)
    unresolved_paths, unresolved_hashes = load_unresolved_records(unresolved_review_path)
    rows = validate_intake_rows(
        rows, dataset_root,
        existing_splits=existing_splits,
        unresolved_paths=unresolved_paths,
        unresolved_hashes=unresolved_hashes,
    )
    summary = assign_fixed_splits(rows, seed=seed)
    split_paths = {row["source_relative_path"] for row in rows if row["split"] in {"train", "validation"}}
    split_hashes = {row["sha256"] for row in rows if row["split"] in {"train", "validation"}}
    test_paths = {
        path for path, splits in existing_splits.path_splits.items() if "test" in splits
    }
    test_hashes = {
        digest for digest, splits in existing_splits.hash_splits.items() if "test" in splits
    }
    overlap_paths = split_paths & test_paths
    overlap_hashes = split_hashes & test_hashes
    if overlap_paths or overlap_hashes:
        raise MobileManifestError("Existing held-out test paths or hashes were assigned to training/validation.")
    write_manifest_new(output_path, rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset-root", type=Path, default=workflow.DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protected-manifest", type=Path, action="append", dest="protected_manifests")
    parser.add_argument("--unresolved-review", type=Path, default=DEFAULT_UNRESOLVED_REVIEW)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    protected = tuple(args.protected_manifests or ())
    try:
        summary = prepare_manifest(
            args.input, args.dataset_root, args.output,
            protected_manifests=protected,
            unresolved_review_path=args.unresolved_review,
            seed=args.seed,
        )
    except InsufficientDeviceGroupsError as error:
        print(f"DATA SHORTAGE: {error}", file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, MobileManifestError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Prepared mobile manifest: " + str(Path(args.output).resolve()))
    print(f"Reviewed images assigned: {summary['reviewed_images']}")
    print(f"Physical-device groups: {summary['split_device_counts']}")
    print(f"Eligible but outside fixed pilot allocation: {summary['unselected_devices']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
