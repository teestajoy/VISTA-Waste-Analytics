"""Small, side-effect-safe helpers for gallery review snapshots."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Mapping

REVIEW_STATES = ("pending", "same_device", "different_device", "uncertain")


def group_ids(row: Mapping[str, str]) -> set[str]:
    return {value.strip() for value in row.get("candidate_group_id", "").split(";") if value.strip()}


def apply_group_review(
    rows: Iterable[Mapping[str, str]],
    candidate_group_id: str,
    decision: str,
    additional_note: str = "",
) -> tuple[list[dict[str, str]], int]:
    """Return a changed copy; only review state and notes for group members can change.

    A decision applies to all image rows belonging to the selected candidate group.
    Existing notes are retained and new notes are appended.
    """
    if decision not in REVIEW_STATES:
        raise ValueError(f"Unsupported review state: {decision}")
    if not candidate_group_id.strip():
        raise ValueError("Candidate group ID must not be empty")

    note = additional_note.strip()
    updated: list[dict[str, str]] = []
    matched = 0
    for source in rows:
        row = dict(source)
        if candidate_group_id in group_ids(row):
            matched += 1
            row["human_decision"] = decision
            previous = row.get("notes", "").strip()
            if note and note not in previous:
                row["notes"] = f"{previous}\n{note}" if previous else note
        updated.append(row)

    if matched == 0:
        raise ValueError(f"No worksheet rows belong to candidate group {candidate_group_id!r}")
    return updated, matched


def serialize_rows(rows: Iterable[Mapping[str, str]]) -> str:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("Cannot export an empty review worksheet")
    fields = list(materialized[0])
    if any(list(row) != fields for row in materialized):
        raise ValueError("Review rows do not share the same CSV columns")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(materialized)
    return stream.getvalue()


def write_review_csv_exclusive(
    destination: Path,
    source_csv: Path,
    rows: Iterable[Mapping[str, str]],
) -> Path:
    """Write a review snapshot once; never overwrite a source or existing output."""
    destination = Path(destination)
    source_csv = Path(source_csv)
    if destination.resolve() == source_csv.resolve():
        raise ValueError("Output path must not be the source review worksheet")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {destination.parent}")
    content = serialize_rows(rows)
    try:
        with destination.open("x", encoding="utf-8-sig", newline="") as target:
            target.write(content)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing review file: {destination}. "
            "Choose a new versioned CSV path."
        ) from exc
    return destination
