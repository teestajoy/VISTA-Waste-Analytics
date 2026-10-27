from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "mobile_battery_strong_bridge_21_review_v1"
    / "review_store.py"
)
SPEC = importlib.util.spec_from_file_location("mobile_battery_bridge21_review_store", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
review_store = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_store)


def sample_review():
    return {
        "bridge_id": "#21",
        "image_path_a": "train/images/a.jpg",
        "image_path_b": "train/images/b.jpg",
        "decision": "uncertain",
        "notes": "Similar appearance; physical identity not confirmed.",
    }


def test_saves_bridge21_decision_and_notes(tmp_path):
    destination = tmp_path / "bridge21_review_v1.csv"

    saved = review_store.save_review_exclusive(destination, sample_review())

    assert saved == destination
    with destination.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows == [sample_review()]


def test_existing_review_file_is_never_overwritten(tmp_path):
    destination = tmp_path / "bridge21_review_v1.csv"
    original = b"existing human review\n"
    destination.write_bytes(original)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        review_store.save_review_exclusive(destination, sample_review())

    assert destination.read_bytes() == original


def test_rejects_wrong_bridge_id_and_invalid_decision(tmp_path):
    review = sample_review()
    review["bridge_id"] = "#20"
    with pytest.raises(ValueError, match="reserved for bridge #21"):
        review_store.save_review_exclusive(tmp_path / "wrong.csv", review)

    review = sample_review()
    review["decision"] = "maybe"
    with pytest.raises(ValueError, match="Unsupported review decision"):
        review_store.save_review_exclusive(tmp_path / "invalid.csv", review)
