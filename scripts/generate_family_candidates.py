"""Create an auditable, metadata-only family candidate report.

Folder policy is used only to organize human review. This script never writes
human decisions, split assignments, or model predictions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import family_dataset_workflow as workflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from vista.classification.family_data_policy import (
    MWCD_FAMILY_SOURCE_POLICY,
    SourceDisposition,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "family_candidate_audit.csv"
FIELDS = (
    "source_relative_path",
    "sha256",
    "source_folder",
    "proposed_review_route",
    "suggested_family",
    "candidate_status",
    "review_requirement",
    "duplicate_group_id",
    "duplicate_group_paths",
    "human_review_status",
    "human_reviewed_family",
    "human_review_reason",
)


def _decision_rows(
    inventory: list[dict[str, str]],
    queue: list[dict[str, str]],
    decisions_path: Path,
) -> dict[str, dict[str, str]]:
    if not decisions_path.exists():
        return {}
    validated = workflow._load_decisions(decisions_path, inventory, queue)
    return validated


def build_candidate_rows(
    inventory: list[dict[str, str]],
    queue: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Return policy suggestions for every inventory row, never approvals."""
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        digest = (row.get("sha256") or "").strip().lower()
        if digest:
            by_hash[digest].append(row)

    queue_paths = {row["source_relative_path"] for row in queue}
    decisions_by_path = decisions
    output: list[dict[str, str]] = []
    for row in inventory:
        relative = row["source_relative_path"]
        folder = row["source_folder"]
        digest = (row.get("sha256") or "").strip().lower()
        policy = MWCD_FAMILY_SOURCE_POLICY[folder]
        group = by_hash.get(digest, []) if digest else []
        group_paths = sorted(item["source_relative_path"] for item in group)
        cross_folder = len({item["source_folder"] for item in group}) > 1
        same_folder_canonical = group_paths[0] if group_paths else ""
        suggestion = policy.candidate_family.value if policy.candidate_family else ""
        status = "pending_human_review"
        requirement = policy.rationale

        if row.get("inventory_error") or not digest:
            suggestion = ""
            status = "unreadable_or_unhashed"
            requirement = row.get("inventory_error") or "A valid SHA-256 is required before review."
        elif cross_folder:
            suggestion = ""
            status = "cross_folder_duplicate_hold"
            requirement = (
                "Exact bytes occur in multiple source folders; inspect the complete group "
                "and resolve the family consistently before any inclusion."
            )
        elif same_folder_canonical and relative != same_folder_canonical:
            status = "same_folder_duplicate_copy"
            requirement = f"Review canonical path only: {same_folder_canonical}"
        elif policy.disposition is SourceDisposition.HOLD_FOR_POLICY:
            suggestion = ""
            status = "hold_for_manual_review"
        elif policy.disposition is SourceDisposition.UNSUPPORTED:
            suggestion = ""
            status = "unsupported"
        elif folder == "organic":
            requirement = "MSW candidate only after a human confirms visible food waste."
        elif folder == "battery":
            requirement = "E_WASTE candidate only after a human confirms visible battery/e-waste content."

        decision = decisions_by_path.get(relative, {}) if relative in queue_paths else {}
        output.append({
            "source_relative_path": relative,
            "sha256": digest,
            "source_folder": folder,
            "proposed_review_route": row["proposed_review_route"],
            "suggested_family": suggestion,
            "candidate_status": status,
            "review_requirement": requirement,
            "duplicate_group_id": row.get("duplicate_group_id", "") or "",
            "duplicate_group_paths": " | ".join(group_paths),
            "human_review_status": decision.get("review_status", "pending"),
            "human_reviewed_family": decision.get("reviewed_family", ""),
            "human_review_reason": decision.get("reason", ""),
        })
    return sorted(output, key=lambda item: item["source_relative_path"].casefold())


def summarize(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    """Summarize review candidates without converting suggestions to labels."""
    by_folder: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_human_status: Counter[str] = Counter()
    for row in rows:
        by_folder[row["source_folder"]] += 1
        by_family[row["suggested_family"] or "no_family_suggestion"] += 1
        by_status[row["candidate_status"]] += 1
        status = row["candidate_status"]
        folder = row["source_folder"]
        if status == "cross_folder_duplicate_hold":
            reason_group = "Cross-folder exact duplicate requires group review"
        elif status == "same_folder_duplicate_copy":
            reason_group = "Same-folder duplicate copy; use the canonical path"
        elif status == "unreadable_or_unhashed":
            reason_group = "Unreadable or missing SHA-256"
        elif status == "unsupported":
            reason_group = "Unsupported source category"
        elif folder == "organic":
            reason_group = "Confirm visible food waste"
        elif folder == "battery":
            reason_group = "Confirm visible battery/e-waste content"
        elif status == "hold_for_manual_review":
            reason_group = "Ambiguous source category; manual review"
        else:
            reason_group = "Human image review required"
        by_reason[reason_group] += 1
        by_human_status[row["human_review_status"]] += 1
    return {
        "source_folder": by_folder,
        "suggested_family": by_family,
        "candidate_status": by_status,
        "review_requirement": by_reason,
        "human_review_status": by_human_status,
    }


def generate_report(
    inventory_path: Path,
    queue_path: Path,
    decisions_path: Path,
    output_path: Path,
    dataset_root: Path,
) -> tuple[list[dict[str, str]], dict[str, Counter[str]]]:
    """Read existing metadata and write a new report without scanning images."""
    output_path = workflow._validate_new_output(output_path, dataset_root)
    inventory = workflow._load_inventory(inventory_path)
    queue = workflow._load_queue(queue_path, inventory)
    decisions = _decision_rows(inventory, queue, decisions_path)
    rows = build_candidate_rows(inventory, queue, decisions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, summarize(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=workflow.DATASET_DIR)
    parser.add_argument("--inventory", type=Path, default=workflow.INVENTORY_PATH)
    parser.add_argument("--queue", type=Path, default=workflow.QUEUE_PATH)
    parser.add_argument("--decisions", type=Path, default=workflow.DECISIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        rows, summary = generate_report(
            args.inventory, args.queue, args.decisions, args.output, args.dataset_root
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Candidate audit stopped safely: {error}\n")
    print(f"Candidate audit rows: {len(rows)}")
    for dimension, counts in summary.items():
        print(f"{dimension.replace('_', ' ').title()}: {dict(sorted(counts.items()))}")
    print("Suggestions are metadata-only and are not human approvals or model predictions.")
    print(f"Wrote new report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
