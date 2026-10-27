from __future__ import annotations

import importlib.util
import csv
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "mobile_battery_related_image_gallery_v2"
    / "review_store.py"
)
SPEC = importlib.util.spec_from_file_location("mobile_battery_review_store_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
review_store = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_store)


def example_rows() -> list[dict[str, str]]:
    return [
        {
            "candidate_group_id": "ND-01;TS-B-01",
            "image_path": "test/images/a.jpg",
            "original_class": "Battery_Waste",
            "original_split": "test",
            "similarity_evidence": "evidence a",
            "human_decision": "uncertain",
            "notes": "Earlier visual note.",
            "reviewer": "",
            "review_date": "",
        },
        {
            "candidate_group_id": "ND-01",
            "image_path": "valid/images/b.jpg",
            "original_class": "Battery_Waste",
            "original_split": "valid",
            "similarity_evidence": "evidence b",
            "human_decision": "pending",
            "notes": "",
            "reviewer": "",
            "review_date": "",
        },
        {
            "candidate_group_id": "ND-02",
            "image_path": "train/images/c.jpg",
            "original_class": "Mobile",
            "original_split": "train",
            "similarity_evidence": "evidence c",
            "human_decision": "same_device",
            "notes": "Keep this previous decision.",
            "reviewer": "",
            "review_date": "",
        },
    ]


def test_group_decision_and_notes_apply_to_members_only_and_save(tmp_path: Path) -> None:
    rows = example_rows()
    source = tmp_path / "review_v2.csv"
    source.write_text("source", encoding="utf-8")

    updated, count = review_store.apply_group_review(
        rows,
        "ND-01",
        "uncertain",
        "Appears to be the same source image or an augmented copy; physical identity unconfirmed.",
    )

    assert count == 2
    assert updated[0]["human_decision"] == "uncertain"
    assert "Earlier visual note." in updated[0]["notes"]
    assert "same source image or an augmented copy" in updated[0]["notes"]
    assert updated[1]["human_decision"] == "uncertain"
    assert updated[2] == rows[2]
    assert [updated[i]["image_path"] for i in range(2)] == [rows[i]["image_path"] for i in range(2)]
    assert [updated[i]["original_split"] for i in range(2)] == ["test", "valid"]
    assert [updated[i]["original_class"] for i in range(2)] == ["Battery_Waste", "Battery_Waste"]
    assert [updated[i]["similarity_evidence"] for i in range(2)] == ["evidence a", "evidence b"]

    output = tmp_path / "review_v3.csv"
    review_store.write_review_csv_exclusive(output, source, updated)
    assert output.is_file()
    assert source.read_text(encoding="utf-8") == "source"
    with output.open(encoding="utf-8-sig", newline="") as saved_file:
        saved = {row["image_path"]: row for row in csv.DictReader(saved_file)}
    assert saved["test/images/a.jpg"]["human_decision"] == "uncertain"
    assert "same source image or an augmented copy" in saved["test/images/a.jpg"]["notes"]
    assert saved["train/images/c.jpg"]["human_decision"] == "same_device"


def test_prior_decisions_and_notes_are_preserved_for_other_groups(tmp_path: Path) -> None:
    rows = example_rows()
    prior = dict(rows[2])
    updated, _ = review_store.apply_group_review(rows, "ND-01", "different_device", "Reviewed this pair.")

    assert updated[2] == prior
    assert updated[0]["notes"].startswith("Earlier visual note.")
    assert updated[0]["human_decision"] == "different_device"
    unchanged_note, _ = review_store.apply_group_review(rows, "ND-01", "uncertain", "")
    assert unchanged_note[0]["notes"] == rows[0]["notes"]

    source = tmp_path / "review_v2.csv"
    source.write_text("source", encoding="utf-8")
    output = tmp_path / "review_v3.csv"
    review_store.write_review_csv_exclusive(output, source, updated)
    with output.open(encoding="utf-8-sig", newline="") as saved_file:
        saved = {row["image_path"]: row for row in csv.DictReader(saved_file)}
    assert saved["train/images/c.jpg"]["human_decision"] == prior["human_decision"]
    assert saved["train/images/c.jpg"]["notes"] == prior["notes"]


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "review_v2.csv"
    source.write_text("source", encoding="utf-8")
    output = tmp_path / "review_v3.csv"
    output.write_bytes(b"existing user file")

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing review file"):
        review_store.write_review_csv_exclusive(output, source, example_rows())

    assert output.read_bytes() == b"existing user file"
    assert source.read_text(encoding="utf-8") == "source"
