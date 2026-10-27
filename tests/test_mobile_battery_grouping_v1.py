from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "group_mobile_battery_dataset_v1.py"
SPEC = importlib.util.spec_from_file_location("mobile_battery_grouping_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
grouping = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grouping)


def record(path: str, sha: str, split: str = "train", label: str = "Mobile", timestamp: str = "") -> dict:
    return {
        "image_path": path,
        "sha256": sha,
        "original_split": split,
        "class_labels": {label},
        "timestamp": timestamp,
    }


def test_exact_hash_duplicates_form_one_strong_group() -> None:
    first = record("train/images/a.jpg", "same-hash", "train")
    second = record("test/images/b.jpg", "same-hash", "test")
    evidence = grouping.classify_pair(first, second, phash_distance=0, orb_inliers=40, orb_inlier_fraction=1.0)
    assert evidence["match_level"] == "exact_duplicate"
    edge = {"image_path_a": first["image_path"], "image_path_b": second["image_path"], **evidence}
    ids, _ = grouping.assign_strong_groups([first, second], [edge])
    assert ids[first["image_path"]] == ids[second["image_path"]]


def test_uncertain_visual_match_stays_a_weak_bridge_not_a_strong_group() -> None:
    first = record("train/images/a.jpg", "hash-a", "train")
    second = record("valid/images/b.jpg", "hash-b", "valid")
    evidence = grouping.classify_pair(first, second, phash_distance=8, orb_inliers=5, orb_inlier_fraction=0.2)
    assert evidence["match_level"] == "weak_match"
    edge = {"image_path_a": first["image_path"], "image_path_b": second["image_path"], **evidence}
    ids, weak = grouping.assign_strong_groups([first, second], [edge])
    assert ids[first["image_path"]] != ids[second["image_path"]]
    assert ids[second["image_path"]] in weak[ids[first["image_path"]]]


def test_conflicting_timestamp_and_visual_evidence_is_flagged() -> None:
    first = record("train/images/a.jpg", "hash-a", "train", timestamp="20250803_120000")
    second = record("test/images/b.jpg", "hash-b", "test", timestamp="20250803_120000")
    evidence = grouping.classify_pair(first, second, phash_distance=24, orb_inliers=1, orb_inlier_fraction=0.1)
    assert evidence["match_level"] == "weak_match"
    assert evidence["evidence_conflict"] is True
    assert "timestamp_match_but_visual_features_disagree" in evidence["conflict_reasons"]


def test_strong_group_leakage_is_detected_without_assigning_new_splits() -> None:
    first = record("train/images/a.jpg", "shared", "train")
    second = record("test/images/b.jpg", "shared", "test")
    evidence = grouping.classify_pair(first, second, phash_distance=0, orb_inliers=40, orb_inlier_fraction=1.0)
    edge = {"image_path_a": first["image_path"], "image_path_b": second["image_path"], **evidence}
    ids, _ = grouping.assign_strong_groups([first, second], [edge])
    assert ids[first["image_path"]] == ids[second["image_path"]]
    leakage = grouping.detect_split_leakage([first, second], ids)
    assert leakage == {ids[first["image_path"]]: {"train", "test"}}
    manifest_rows = [
        {"image_path": item["image_path"], "new_split_assignment": ""}
        for item in (first, second)
    ]
    assert all(row["new_split_assignment"] == "" for row in manifest_rows)


def test_conflicting_original_labels_are_flagged_even_for_exact_hash() -> None:
    first = record("train/images/a.jpg", "shared", "train", "Mobile")
    second = record("valid/images/b.jpg", "shared", "valid", "Battery_Waste")
    evidence = grouping.classify_pair(first, second, phash_distance=0, orb_inliers=30, orb_inlier_fraction=0.9)
    assert evidence["match_level"] == "exact_duplicate"
    assert evidence["class_conflict"] is True
    assert evidence["evidence_conflict"] is True


def test_outputs_refuse_to_overwrite_any_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.csv"
    destination.write_bytes(b"user-owned content")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        grouping.write_outputs_exclusive({destination: "new content"})
    assert destination.read_bytes() == b"user-owned content"
