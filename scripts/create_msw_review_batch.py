"""Create a compact, review-only MSW candidate batch from existing CSVs.

Folder routes are candidate suggestions, not reviewed labels. This script only
selects source paths and renders contact sheets; every new decision starts
pending and requires an explicit human action in the review interface.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import family_dataset_workflow as workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = workflow.INVENTORY_PATH
CANDIDATE_AUDIT_PATH = PROJECT_ROOT / "data" / "processed" / "family_candidate_audit.csv"
QUEUE_PATH = workflow.QUEUE_PATH
DECISIONS_PATH = workflow.DECISIONS_PATH
DEFAULT_BATCH_QUEUE = PROJECT_ROOT / "data" / "processed" / "family_msw_review_batch_queue.csv"
DEFAULT_BATCH_DECISIONS = PROJECT_ROOT / "data" / "processed" / "family_msw_review_batch_decisions.csv"
DEFAULT_SHEETS_DIR = PROJECT_ROOT / "test-data" / "family_msw_review_batch"
MSW_FOLDERS = ("paper", "cardboard", "plastic")
CANDIDATE_FIELDS = (
    "source_relative_path", "sha256", "source_folder", "proposed_review_route",
    "suggested_family", "candidate_status", "duplicate_group_id",
)


def _load_candidate_audit(path: Path, inventory: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = workflow._read_csv(path, CANDIDATE_FIELDS)
    inventory_by_path = {row["source_relative_path"]: row for row in inventory}
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        relative = workflow.normalize_relative_path(row.get("source_relative_path", ""))
        if relative in seen:
            raise ValueError(f"Candidate audit repeats source path on line {line}: {relative}")
        seen.add(relative)
        source = inventory_by_path.get(relative)
        if source is None:
            raise ValueError(f"Candidate audit contains a path absent from inventory: {relative}")
        for field in ("sha256", "source_folder", "proposed_review_route", "duplicate_group_id"):
            if (row.get(field) or "") != (source.get(field) or ""):
                raise ValueError(f"Candidate audit is stale for {relative}: {field} differs from inventory")
    if seen != set(inventory_by_path):
        raise ValueError("Candidate audit does not cover the current inventory exactly")
    return rows


def select_review_batch(
    inventory: list[dict[str, str]],
    candidate_audit: list[dict[str, str]],
    existing_queue: list[dict[str, str]],
    existing_decisions: dict[str, dict[str, str]],
    dataset_root: Path,
    *,
    per_folder: int = 20,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Sample pending canonical candidates already present in the original queue."""
    if per_folder < 1:
        raise ValueError("per_folder must be at least 1")
    inventory_by_path = {row["source_relative_path"]: row for row in inventory}
    queue_by_path = {row["source_relative_path"]: row for row in existing_queue}
    candidate_by_path = {row["source_relative_path"]: row for row in candidate_audit}
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        if row.get("sha256"):
            by_hash[row["sha256"]].append(row)

    cross_folder_hashes = {
        digest for digest, members in by_hash.items()
        if len({row["source_folder"] for row in members}) > 1
    }
    canonical_by_hash = {
        digest: min(row["source_relative_path"] for row in members)
        for digest, members in by_hash.items()
    }
    selected: list[dict[str, str]] = []
    for folder_index, folder in enumerate(MSW_FOLDERS):
        candidates = []
        for path, queue_row in queue_by_path.items():
            audit = candidate_by_path[path]
            source = inventory_by_path[path]
            if (
                source["source_folder"] != folder
                or not source.get("sha256")
                or source.get("inventory_error")
                or audit.get("suggested_family") != "MSW"
                or audit.get("candidate_status") != "pending_human_review"
                or source["sha256"] in cross_folder_hashes
                or canonical_by_hash[source["sha256"]] != path
            ):
                continue
            group_members = by_hash[source["sha256"]]
            if any(
                existing_decisions.get(member["source_relative_path"], {}).get("review_status", "pending")
                != "pending"
                for member in group_members
            ):
                continue
            candidates.append(queue_row)

        candidates.sort(key=lambda row: row["source_relative_path"])
        random.Random(seed + folder_index).shuffle(candidates)
        for row in candidates[:per_folder]:
            source_path = workflow.safe_source_path(dataset_root, row["source_relative_path"])
            if not source_path.is_file():
                raise FileNotFoundError(f"Review candidate is missing: {row['source_relative_path']}")
            current_hash = workflow.sha256_file(source_path)
            if current_hash != row["sha256"]:
                raise ValueError(f"SHA-256 changed since inventory: {row['source_relative_path']}")
            try:
                workflow._verify_image(source_path)
            except (OSError, ValueError) as error:
                raise ValueError(f"Review candidate is unreadable: {row['source_relative_path']}: {error}") from error
            selected.append({field: row.get(field, "") or "" for field in workflow.QUEUE_FIELDS})

    return sorted(selected, key=lambda row: row["source_relative_path"].casefold())


def create_batch(
    *,
    dataset_root: Path = workflow.DATASET_DIR,
    inventory_path: Path = INVENTORY_PATH,
    candidate_audit_path: Path = CANDIDATE_AUDIT_PATH,
    existing_queue_path: Path = QUEUE_PATH,
    existing_decisions_path: Path = DECISIONS_PATH,
    output_queue_path: Path = DEFAULT_BATCH_QUEUE,
    output_decisions_path: Path = DEFAULT_BATCH_DECISIONS,
    sheets_dir: Path = DEFAULT_SHEETS_DIR,
    per_folder: int = 20,
    seed: int = 42,
) -> list[dict[str, str]]:
    inventory = workflow._load_inventory(inventory_path)
    candidates = _load_candidate_audit(candidate_audit_path, inventory)
    existing_queue = workflow._load_queue(existing_queue_path, inventory)
    existing_decisions = workflow._load_decisions(
        existing_decisions_path, inventory, existing_queue
    )

    outputs = tuple(
        workflow._validate_new_output(path, dataset_root)
        for path in (output_queue_path, output_decisions_path, sheets_dir)
    )
    if len(set(outputs)) != 3:
        raise ValueError("Batch queue, decisions, and contact-sheet paths must be different")
    for index, first in enumerate(outputs):
        for second in outputs[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError("Batch output paths cannot contain one another")

    queue_rows = select_review_batch(
        inventory, candidates, existing_queue, existing_decisions, dataset_root,
        per_folder=per_folder, seed=seed,
    )
    if not queue_rows:
        raise ValueError("No pending, readable, canonical MSW candidates are available in the existing queue")

    pages = {
        folder: workflow._render_pdf_pages(
            [row for row in queue_rows if row["source_folder"] == folder], dataset_root
        )
        for folder in MSW_FOLDERS
        if any(row["source_folder"] == folder for row in queue_rows)
    }
    output_queue, output_decisions, output_sheets = outputs
    output_sheets.mkdir(parents=True, exist_ok=False)
    for folder, folder_pages in pages.items():
        with (output_sheets / f"{folder}.pdf").open("xb") as target:
            folder_pages[0].save(
                target, format="PDF", save_all=True, append_images=folder_pages[1:], resolution=100
            )
        for page in folder_pages:
            page.close()

    workflow._write_csv_new(output_queue, workflow.QUEUE_FIELDS, queue_rows)
    pending = [
        {"source_relative_path": row["source_relative_path"], "review_status": "pending",
         "reviewed_family": "", "reason": ""}
        for row in queue_rows
    ]
    workflow._write_csv_new(output_decisions, workflow.DECISION_FIELDS, pending)
    return queue_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=workflow.DATASET_DIR)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--candidate-audit", type=Path, default=CANDIDATE_AUDIT_PATH)
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    parser.add_argument("--decisions", type=Path, default=DECISIONS_PATH)
    parser.add_argument("--per-folder", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-queue", type=Path, default=DEFAULT_BATCH_QUEUE)
    parser.add_argument("--output-decisions", type=Path, default=DEFAULT_BATCH_DECISIONS)
    parser.add_argument("--sheets-dir", type=Path, default=DEFAULT_SHEETS_DIR)
    args = parser.parse_args(argv)
    try:
        rows = create_batch(
            dataset_root=args.dataset_root,
            inventory_path=args.inventory,
            candidate_audit_path=args.candidate_audit,
            existing_queue_path=args.queue,
            existing_decisions_path=args.decisions,
            output_queue_path=args.output_queue,
            output_decisions_path=args.output_decisions,
            sheets_dir=args.sheets_dir,
            per_folder=args.per_folder,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Created review-only batch with {len(rows)} images; all decisions remain pending.")
    for folder in MSW_FOLDERS:
        print(f"  {folder}: {sum(row['source_folder'] == folder for row in rows)}")
    print(f"Queue: {args.output_queue}")
    print(f"Blank decisions: {args.output_decisions}")
    print(f"Contact-sheet PDFs: {args.sheets_dir}")
    print("Policy: these are MSW review candidates only; approve each image from visible content after inspecting the contact sheets.")


if __name__ == "__main__":
    main()
