from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


GALLERY_DIR = Path(__file__).resolve().parents[1] / "outputs" / "mobile_battery_strong_bridge_review_v1"
sys.path.insert(0, str(GALLERY_DIR))
import review_store  # noqa: E402


def sample_reviews():
    return [
        {
            "bridge_id": "#3",
            "image_path_a": "test/images/a.jpg",
            "image_path_b": "train/images/b.jpg",
            "decision": "uncertain",
            "notes": "Visually similar; device identity not confirmed.",
        },
        {
            "bridge_id": "#4",
            "image_path_a": "test/images/a.jpg",
            "image_path_b": "valid/images/c.jpg",
            "decision": "pending",
            "notes": "",
        },
    ]


def test_saves_both_bridge_decisions_and_notes(tmp_path):
    destination = tmp_path / "bridge_reviews_v1.csv"

    saved = review_store.save_reviews_exclusive(destination, sample_reviews())

    assert saved == destination
    with destination.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows == sample_reviews()
    assert tuple(rows[0]) == review_store.FIELDS


def test_existing_review_file_is_never_overwritten(tmp_path):
    destination = tmp_path / "bridge_reviews_v1.csv"
    original = b"preserve this user file\n"
    destination.write_bytes(original)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        review_store.save_reviews_exclusive(destination, sample_reviews())

    assert destination.read_bytes() == original


def test_rejects_missing_or_invalid_bridge_decisions(tmp_path):
    invalid = sample_reviews()
    invalid[0]["decision"] = "maybe_same_device"

    with pytest.raises(ValueError, match="Unsupported review decision"):
        review_store.save_reviews_exclusive(tmp_path / "review.csv", invalid)
