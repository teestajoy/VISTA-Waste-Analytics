"""Exclusive CSV persistence for bridge #21 review decisions."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Mapping

REVIEW_STATES = {"pending", "same_device", "different_device", "uncertain"}
FIELDS = ("bridge_id", "image_path_a", "image_path_b", "decision", "notes")


def serialize_review(review: Mapping[str, str]) -> str:
    row = {field: review.get(field, "") for field in FIELDS}
    if row["bridge_id"] != "#21":
        raise ValueError("This review file is reserved for bridge #21")
    if not row["image_path_a"].strip() or not row["image_path_b"].strip():
        raise ValueError("Both bridge image paths are required")
    if row["decision"] not in REVIEW_STATES:
        raise ValueError(f"Unsupported review decision: {row['decision']}")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue()


def save_review_exclusive(destination: Path, review: Mapping[str, str]) -> Path:
    """Create one review CSV without replacing any existing file."""
    destination = Path(destination)
    if destination.suffix.lower() != ".csv":
        raise ValueError("Review output must have a .csv extension")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Review output directory does not exist: {destination.parent}")
    content = serialize_review(review)
    try:
        with destination.open("x", encoding="utf-8-sig", newline="") as output:
            output.write(content)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing review file: {destination}. Choose a new versioned filename."
        ) from exc
    return destination
