"""Prepare a separate folder-derived provisional MSW/E-waste experiment.

These labels are weak folder proxies, not verified ground truth. The script
reads the existing inventory, excludes unsafe or conflicting hash groups,
rechecks candidate source files, and never edits review decisions or images.
"""

from __future__ import annotations

import argparse
import csv
import random
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import family_dataset_workflow as workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = PROJECT_ROOT / "data" / "processed" / "family_dataset_inventory.csv"
REVIEW_QUEUE_PATH = PROJECT_ROOT / "data" / "processed" / "family_review_queue.csv"
REVIEW_DECISIONS_PATH = PROJECT_ROOT / "data" / "processed" / "family_review_decisions.csv"
BATCH_DECISIONS_PATH = PROJECT_ROOT / "data" / "processed" / "family_msw_review_batch_decisions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "family_provisional_split_manifest.csv"
PROVISIONAL_OUTPUT_DIR = PROJECT_ROOT / "models" / "waste_family_provisional"

DEFAULT_MSW_FOLDERS = ("paper", "cardboard", "plastic", "glass", "organic")
DEFAULT_E_WASTE_FOLDERS = ("battery",)
ALLOWED_MSW_FOLDERS = frozenset(DEFAULT_MSW_FOLDERS)
ALLOWED_E_WASTE_FOLDERS = frozenset(DEFAULT_E_WASTE_FOLDERS)
PROVENANCE = "folder_derived_provisional"
SPLITS = ("train", "validation", "test")
MANIFEST_FIELDS = (
    "source_relative_path", "sha256", "source_folder", "provisional_family",
    "label_provenance", "source_folder_mapping", "human_review_status",
    "human_reviewed_family", "human_review_reason", "duplicate_group_id",
    "canonical_path", "group_status", "split", "exclusion_reason",
)


class ProvisionalDataError(ValueError):
    """Raised when provisional source data cannot be split safely."""


def parse_folder_mapping(
    msw_folders: Iterable[str] = DEFAULT_MSW_FOLDERS,
    e_waste_folders: Iterable[str] = DEFAULT_E_WASTE_FOLDERS,
) -> dict[str, str]:
    """Validate the explicitly allowed provisional source-folder mapping."""
    msw = tuple(folder.strip().lower() for folder in msw_folders if folder.strip())
    e_waste = tuple(folder.strip().lower() for folder in e_waste_folders if folder.strip())
    if len(msw) != len(set(msw)) or len(e_waste) != len(set(e_waste)):
        raise ProvisionalDataError("A source folder appears more than once in the mapping.")
    if not msw or not e_waste or set(msw) & set(e_waste):
        raise ProvisionalDataError("Mapping must contain disjoint, non-empty MSW and E_WASTE folders.")
    invalid_msw = set(msw) - ALLOWED_MSW_FOLDERS
    invalid_e_waste = set(e_waste) - ALLOWED_E_WASTE_FOLDERS
    if invalid_msw or invalid_e_waste:
        invalid = sorted(invalid_msw | invalid_e_waste)
        raise ProvisionalDataError(
            "Unsupported provisional mapping folder(s): " + ", ".join(invalid)
            + ". metal, trash, and textile remain excluded under the current policy."
        )
    return {**{folder: "MSW" for folder in msw}, **{folder: "E_WASTE" for folder in e_waste}}


def _load_explicit_decisions(
    inventory: list[dict[str, str]], queue: list[dict[str, str]], paths: Iterable[Path]
) -> dict[str, dict[str, str]]:
    """Load human review records only as exclusion/contradiction safeguards."""
    merged: dict[str, dict[str, str]] = {}
    for path in paths:
        if not Path(path).is_file():
            continue
        decisions = workflow._load_decisions(Path(path), inventory, queue)
        merged.update({
            relative: record for relative, record in decisions.items()
            if record["review_status"] != "pending"
        })
    return merged


def _split_counts(count: int) -> tuple[int, int, int]:
    if count < 3:
        raise ProvisionalDataError(
            "At least three distinct eligible hashes per family are required to populate "
            "train, validation, and test."
        )
    counts = [1, 1, 1]
    targets = [count * 0.70, count * 0.15, count * 0.15]
    for _ in range(count - 3):
        index = max(range(3), key=lambda i: (targets[i] - counts[i], -i))
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def _validate_source(row: dict[str, str], dataset_root: Path) -> str:
    """Return a clear error string for a missing, changed, or unreadable source."""
    relative = row["source_relative_path"]
    try:
        path = workflow.safe_source_path(dataset_root, relative)
        if not path.is_file():
            return f"source file missing: {relative}"
        actual = workflow.sha256_file(path)
        if actual != row["sha256"]:
            return f"current SHA-256 differs from inventory: {relative}"
        workflow._verify_image(path)
    except (OSError, ValueError, RuntimeError) as error:
        return f"source is unsafe, unreadable, or unavailable ({relative}): {error}"
    return ""


def _read_decisions_for_candidates(
    inventory: list[dict[str, str]],
    mapping: dict[str, str],
    dataset_root: Path,
    review_queue_path: Path,
    decisions_paths: Iterable[Path],
) -> dict[str, dict[str, str]]:
    folders = set(mapping)
    # The normal queue contains all source routes. When present, use it so the
    # existing family decision loader can validate status/reason/schema exactly.
    if Path(review_queue_path).is_file():
        full_queue = workflow._load_queue(review_queue_path, inventory)
    else:
        full_queue = [
            {field: row.get(field, "") or "" for field in workflow.QUEUE_FIELDS}
            for row in inventory
        ]
    try:
        return _load_explicit_decisions(inventory, full_queue, decisions_paths)
    except (OSError, ValueError) as error:
        raise ProvisionalDataError(f"Could not validate existing family-review decisions: {error}") from error


def build_provisional_manifest(
    inventory_rows: list[dict[str, str]],
    dataset_root: Path,
    *,
    msw_folders: Iterable[str] = DEFAULT_MSW_FOLDERS,
    e_waste_folders: Iterable[str] = DEFAULT_E_WASTE_FOLDERS,
    decisions: dict[str, dict[str, str]] | None = None,
    seed: int = 42,
    max_per_folder: int | None = 1000,
) -> list[dict[str, str]]:
    """Build deterministic hash-group splits with explicitly provisional labels."""
    if max_per_folder is not None and max_per_folder < 3:
        raise ProvisionalDataError("--max-per-folder must be at least 3 or omitted.")
    mapping = parse_folder_mapping(msw_folders, e_waste_folders)
    decisions = decisions or {}
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inventory_rows:
        digest = (row.get("sha256") or "").strip().lower()
        if digest:
            by_hash[digest].append(row)

    result: dict[str, dict[str, str]] = {}
    eligible_by_folder: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in inventory_rows:
        relative = row["source_relative_path"]
        folder = row["source_folder"]
        family = mapping.get(folder, "")
        decision = decisions.get(relative, {
            "review_status": "pending", "reviewed_family": "", "reason": "",
        })
        digest = (row.get("sha256") or "").strip().lower()
        result[relative] = {
            "source_relative_path": relative,
            "sha256": digest,
            "source_folder": folder,
            "provisional_family": family,
            "label_provenance": PROVENANCE if family else "not_mapped",
            "source_folder_mapping": f"{folder} -> {family}" if family else "",
            "human_review_status": decision["review_status"],
            "human_reviewed_family": decision["reviewed_family"],
            "human_review_reason": decision["reason"],
            "duplicate_group_id": f"sha256:{digest}" if digest else "",
            "canonical_path": "",
            "group_status": "excluded",
            "split": "",
            "exclusion_reason": "",
        }
        if not family:
            result[relative]["exclusion_reason"] = "source folder is excluded by the configured family mapping"
        elif not digest:
            result[relative]["exclusion_reason"] = row.get("inventory_error") or "no usable SHA-256 in inventory"
        elif row.get("inventory_error"):
            result[relative]["exclusion_reason"] = row["inventory_error"]
        else:
            by_hash[digest]  # make the required group explicit for this path

    for digest, members in by_hash.items():
        mapped_members = [member for member in members if member["source_folder"] in mapping]
        if not mapped_members:
            continue
        folders = {member["source_folder"] for member in members}
        paths = sorted(member["source_relative_path"] for member in members)
        canonical = paths[0]
        reason = ""
        if len(folders) > 1:
            reason = "exact-hash group spans source folders; excluded as a label-conflict risk"
        elif any(member.get("inventory_error") for member in mapped_members):
            reason = "an exact-hash group member has an inventory readability/hash error"
        else:
            records = [decisions.get(member["source_relative_path"], {
                "review_status": "pending", "reviewed_family": "", "reason": "",
            }) for member in mapped_members]
            if any(record["review_status"] in {"hold", "exclude"} for record in records):
                reason = "existing human HOLD/EXCLUDE decision blocks this provisional hash group"
            else:
                expected_family = mapping[mapped_members[0]["source_folder"]]
                conflicting = [
                    record for record in records
                    if record["review_status"] == "include"
                    and record["reviewed_family"] != expected_family
                ]
                if conflicting:
                    reason = "existing human family decision conflicts with provisional folder mapping"

        if not reason:
            validation_errors = [
                error for member in mapped_members
                if (error := _validate_source(member, dataset_root))
            ]
            if validation_errors:
                reason = validation_errors[0]

        if reason:
            for member in members:
                relative = member["source_relative_path"]
                result[relative]["canonical_path"] = canonical
                if result[relative]["provisional_family"]:
                    result[relative]["exclusion_reason"] = reason
            continue

        family = mapping[mapped_members[0]["source_folder"]]
        for member in members:
            relative = member["source_relative_path"]
            result[relative]["canonical_path"] = canonical
            result[relative]["group_status"] = "duplicate_copy"
            result[relative]["exclusion_reason"] = f"exact duplicate copy; canonical path is {canonical}"
        canonical_row = next(member for member in members if member["source_relative_path"] == canonical)
        eligible_by_folder[canonical_row["source_folder"]].append((digest, canonical))

    selected_by_folder: dict[str, list[tuple[str, str]]] = {}
    for folder in mapping:
        candidates = sorted(eligible_by_folder.get(folder, []))
        stable_seed = (seed + zlib.crc32(folder.encode("utf-8"))) & 0xFFFFFFFF
        random.Random(stable_seed).shuffle(candidates)
        selected = candidates if max_per_folder is None else candidates[:max_per_folder]
        selected_by_folder[folder] = selected
        selected_hashes = {digest for digest, _ in selected}
        for digest, canonical in candidates:
            if digest not in selected_hashes:
                result[canonical]["group_status"] = "excluded"
                result[canonical]["exclusion_reason"] = (
                    f"outside deterministic per-folder sample cap ({max_per_folder})"
                )

    for family in ("MSW", "E_WASTE"):
        family_groups = [
            (digest, canonical)
            for folder, groups in selected_by_folder.items()
            if mapping[folder] == family
            for digest, canonical in groups
        ]
        if len(family_groups) < 3:
            raise ProvisionalDataError(
                f"Insufficient distinct eligible hashes for {family}: {len(family_groups)}. "
                "No manifest was written; at least 3 are required for train/validation/test."
            )
        shuffled = sorted(family_groups)
        stable_seed = (seed + zlib.crc32(family.encode("utf-8"))) & 0xFFFFFFFF
        random.Random(stable_seed).shuffle(shuffled)
        train_count, validation_count, _ = _split_counts(len(shuffled))
        for index, (_, canonical) in enumerate(shuffled):
            row = result[canonical]
            row["group_status"] = "provisional_include"
            row["exclusion_reason"] = ""
            row["split"] = (
                "train" if index < train_count
                else "validation" if index < train_count + validation_count
                else "test"
            )

    manifest = sorted(result.values(), key=lambda row: row["source_relative_path"].casefold())
    validate_manifest(manifest)
    return manifest


def validate_manifest(rows: list[dict[str, str]]) -> None:
    """Check provenance, class presence, canonical groups, and split leakage."""
    seen_paths: set[str] = set()
    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for row in rows:
        relative = workflow.normalize_relative_path(row.get("source_relative_path", ""))
        if relative in seen_paths:
            raise ProvisionalDataError(f"Manifest repeats source path: {relative}")
        seen_paths.add(relative)
        if row["group_status"] == "provisional_include":
            if row["label_provenance"] != PROVENANCE or row["provisional_family"] not in {"MSW", "E_WASTE"}:
                raise ProvisionalDataError(f"Included provisional row has invalid provenance: {relative}")
            if row["split"] not in SPLITS or row["canonical_path"] != relative:
                raise ProvisionalDataError(f"Included provisional row is not canonical or split: {relative}")
            if not row["sha256"]:
                raise ProvisionalDataError(f"Included provisional row has no SHA-256: {relative}")
            counts[(row["provisional_family"], row["split"])] += 1
            splits_by_hash[row["sha256"]].add(row["split"])
        elif row["split"]:
            raise ProvisionalDataError(f"Excluded provisional row has a split: {relative}")
    leaking = [digest for digest, splits in splits_by_hash.items() if len(splits) > 1]
    if leaking:
        raise ProvisionalDataError(f"Exact duplicate leakage across splits: {len(leaking)} hash group(s).")
    missing = [(family, split) for family in ("MSW", "E_WASTE") for split in SPLITS if not counts[(family, split)]]
    if missing:
        raise ProvisionalDataError("Missing required family/split groups: " + ", ".join(f"{a}/{b}" for a, b in missing))


def write_manifest(path: Path, rows: list[dict[str, str]], dataset_root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(dataset_root).resolve())
    except ValueError:
        pass
    else:
        raise ProvisionalDataError(f"Manifest output must be outside the dataset: {resolved}")
    protected = {
        workflow.INVENTORY_PATH.resolve(), workflow.QUEUE_PATH.resolve(),
        workflow.DECISIONS_PATH.resolve(), workflow.MANIFEST_PATH.resolve(),
        workflow.MERGED_DECISIONS_PATH.resolve(),
        BATCH_DECISIONS_PATH.resolve(),
        (PROJECT_ROOT / "data" / "processed" / "family_msw_review_batch_queue.csv").resolve(),
        (PROJECT_ROOT / "data" / "processed" / "family_candidate_audit.csv").resolve(),
        *workflow.PILOT_PROTECTED_PATHS,
    }
    if resolved in protected or resolved.name in {
        "paper_cardboard_pilot_review.csv", "paper_cardboard_pilot_split_manifest.csv",
    }:
        raise ProvisionalDataError(f"Refusing to overwrite a protected VISTA manifest: {resolved}")
    protected_directories = (
        PROJECT_ROOT / "models" / "paper_cardboard_resnet18_pilot",
        PROVISIONAL_OUTPUT_DIR,
    )
    for directory in protected_directories:
        try:
            resolved.relative_to(directory.resolve())
        except ValueError:
            continue
        raise ProvisionalDataError(f"Manifest output cannot be inside a model artifact directory: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing provisional manifest: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def summarize(rows: list[dict[str, str]]) -> tuple[dict, dict, Counter]:
    split_counts = Counter(
        (row["provisional_family"], row["split"])
        for row in rows if row["group_status"] == "provisional_include"
    )
    category_counts = Counter(
        (row["source_folder"], row["split"])
        for row in rows if row["group_status"] == "provisional_include"
    )
    excluded = Counter(
        exclusion_reason_category(row)
        for row in rows if row["group_status"] != "provisional_include"
    )
    return split_counts, category_counts, excluded


def exclusion_reason_category(row: dict[str, str]) -> str:
    """Group detailed per-path exclusion reasons into concise report categories."""
    reason = (row.get("exclusion_reason") or "").lower()
    if "source folder is excluded" in reason:
        return "source folder not mapped"
    if "per-folder sample cap" in reason:
        return "outside per-folder sample cap"
    if "spans source folders" in reason:
        return "cross-folder exact-hash conflict"
    if "exact duplicate copy" in reason:
        return "same-folder exact duplicate copy"
    if "human hold/exclude" in reason:
        return "blocked by existing human hold/exclude"
    if "human family decision conflicts" in reason:
        return "conflicts with existing human family decision"
    if "inventory readability/hash error" in reason or "inventory" in reason:
        return "inventory error"
    if "sha-256" in reason or "hash" in reason:
        return "source hash mismatch or missing hash"
    if "unreadable" in reason or "missing" in reason or "unsafe" in reason:
        return "missing, unsafe, or unreadable source"
    return "other exclusion"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--dataset-root", type=Path, default=workflow.DATASET_DIR)
    parser.add_argument("--review-queue", type=Path, default=REVIEW_QUEUE_PATH)
    parser.add_argument("--review-decisions", type=Path, default=REVIEW_DECISIONS_PATH)
    parser.add_argument("--batch-decisions", type=Path, default=BATCH_DECISIONS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--msw-folders", nargs="+", default=list(DEFAULT_MSW_FOLDERS))
    parser.add_argument("--e-waste-folders", nargs="+", default=list(DEFAULT_E_WASTE_FOLDERS))
    parser.add_argument("--max-per-folder", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        mapping = parse_folder_mapping(args.msw_folders, args.e_waste_folders)
        inventory = workflow._load_inventory(args.inventory)
        decisions = _read_decisions_for_candidates(
            inventory, mapping, args.dataset_root, args.review_queue,
            (args.review_decisions, args.batch_decisions),
        )
        rows = build_provisional_manifest(
            inventory, args.dataset_root,
            msw_folders=args.msw_folders,
            e_waste_folders=args.e_waste_folders,
            decisions=decisions,
            seed=args.seed,
            max_per_folder=args.max_per_folder,
        )
        output = write_manifest(args.output, rows, args.dataset_root)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Provisional preparation stopped safely: {error}\n")

    family_counts, category_counts, excluded = summarize(rows)
    print("LABEL PROVENANCE: folder-derived provisional; not human-verified ground truth")
    print("Mapping: " + ", ".join(f"{folder} -> {family}" for folder, family in mapping.items()))
    print(f"Manifest: {output}")
    print("Included distinct hash groups by family and split:")
    for family in ("MSW", "E_WASTE"):
        print(f"  {family}: " + ", ".join(f"{split}={family_counts[(family, split)]}" for split in SPLITS))
    print("Included canonical groups by source category and split:")
    for folder in mapping:
        print(f"  {folder}: " + ", ".join(f"{split}={category_counts[(folder, split)]}" for split in SPLITS))
    print(f"Excluded manifest rows: {sum(excluded.values())}")
    for reason, count in excluded.most_common():
        print(f"  {count}: {reason}")
    print("No source images, human decisions, queues, or paper/cardboard pilot files were modified.")


if __name__ == "__main__":
    main()
