"""Exclusive CSV persistence for the focused strong-bridge review gallery."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Mapping


REVIEW_STATES = ("pending", "same_device", "different_device", "uncertain")
BRIDGE_IDS = {"#3", "#4"}
FIELDS = ("bridge_id", "image_path_a", "image_path_b", "decision", "notes")


def serialize_reviews(reviews: Iterable[Mapping[str, str]]) -> str:
    rows = [dict(row) for row in reviews]
    if len(rows) != 2 or {row.get("bridge_id", "") for row in rows} != BRIDGE_IDS:
        raise ValueError("Review CSV must contain exactly one row each for bridges #3 and #4")
    for row in rows:
        if any(not str(row.get(field, "")).strip() for field in FIELDS if field != "notes"):
            raise ValueError("Bridge ID, both image paths, and decision are required")
        if row["decision"] not in REVIEW_STATES:
            raise ValueError(f"Unsupported review decision: {row['decision']}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)
    return stream.getvalue()


def save_reviews_exclusive(destination: Path, reviews: Iterable[Mapping[str, str]]) -> Path:
    """Create a new CSV with exclusive mode; never replace an existing review file."""
    destination = Path(destination)
    if destination.suffix.lower() != ".csv":
        raise ValueError("Review output must have a .csv extension")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Review output directory does not exist: {destination.parent}")
    content = serialize_reviews(reviews)
    try:
        with destination.open("x", encoding="utf-8-sig", newline="") as target:
            target.write(content)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing review file: {destination}. Choose a new versioned filename."
        ) from exc
    return destination
