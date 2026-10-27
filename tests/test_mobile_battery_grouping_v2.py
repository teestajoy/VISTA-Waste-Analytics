from __future__ import annotations

import importlib.util
import csv
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "group_mobile_battery_dataset_v2.py"
SPEC = importlib.util.spec_from_file_location("mobile_battery_grouping_v2", MODULE_PATH)
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


def test_weak_bridge_chains_are_reported_without_merging_strong_groups() -> None:
    records = [
        record("train/images/a.jpg", "ha", "train"),
        record("train/images/b.jpg", "hb", "train"),
        record("valid/images/c.jpg", "hc", "valid"),
        record("test/images/d.jpg", "hd", "test"),
    ]
    strong = grouping.classify_pair(records[0], records[1], phash_distance=2, orb_inliers=25, orb_inlier_fraction=0.8)
    other_strong = grouping.classify_pair(records[2], records[3], phash_distance=3, orb_inliers=30, orb_inlier_fraction=0.9)
    weak = grouping.classify_pair(records[1], records[2], phash_distance=8, orb_inliers=3, orb_inlier_fraction=0.1)
    edges = [
        {"image_path_a": records[0]["image_path"], "image_path_b": records[1]["image_path"], **strong},
        {"image_path_a": records[2]["image_path"], "image_path_b": records[3]["image_path"], **other_strong},
        {"image_path_a": records[1]["image_path"], "image_path_b": records[2]["image_path"], **weak},
    ]
    strong_ids, _ = grouping.assign_strong_groups(records, edges)
    all_ids, sizes = grouping.candidate_component_ids(
        records, edges, {"exact_duplicate", "strong_candidate", "weak_match"}, "CAND"
    )
    assert strong_ids[records[0]["image_path"]] == strong_ids[records[1]["image_path"]]
    assert strong_ids[records[2]["image_path"]] == strong_ids[records[3]["image_path"]]
    assert strong_ids[records[0]["image_path"]] != strong_ids[records[2]["image_path"]]
    assert len(set(all_ids.values())) == 1
    assert sizes[all_ids[records[0]["image_path"]]] == 4


def test_oversized_candidate_components_are_separate_from_physical_identity() -> None:
    records = [record(f"train/images/{i}.jpg", f"hash-{i}") for i in range(13)]
    edges = []
    for left, right in zip(records, records[1:]):
        evidence = grouping.classify_pair(left, right, phash_distance=9, orb_inliers=1, orb_inlier_fraction=0.1)
        edges.append({"image_path_a": left["image_path"], "image_path_b": right["image_path"], **evidence})
    candidate_ids, sizes = grouping.candidate_component_ids(
        records, edges, {"exact_duplicate", "strong_candidate", "weak_match"}, "CAND"
    )
    strong_ids, _ = grouping.assign_strong_groups(records, edges)
    candidate_id = candidate_ids[records[0]["image_path"]]
    assert sizes[candidate_id] == 13
    assert len(set(strong_ids.values())) == 13


def bridge_edge(left: str, right: str) -> dict:
    return {
        "image_path_a": left,
        "image_path_b": right,
        "match_level": "strong_candidate",
        "strong_graph_bridge": "True",
    }


def write_bridge_reviews(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["bridge_id", "image_path_a", "image_path_b", "decision", "notes"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_bridge_edge_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["image_path_a", "image_path_b", "strong_graph_bridge", "match_level"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_bridge_review_ids_map_to_expected_edge_image_pairs(tmp_path: Path) -> None:
    paths = {
        "a": "test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg",
        "b": "train/images/IMG_20250803_220022_jpg.rf.dc0fdee22f504920709e6321bf117c22.jpg",
        "c": "valid/images/IMG_20250803_220024_jpg.rf.2fc166ace133b8664434156a034ba786.jpg",
        "d": "train/images/IMG_20250805_203922_jpg.rf.fbc486f138d29a49b1f3de783c184c12.jpg",
        "e": "train/images/IMG_20250805_203923_jpg.rf.5c6fc3c85b160812cf0a3d80250b24d1.jpg",
    }
    edges = [bridge_edge(f"train/images/dummy-{i}-a.jpg", f"train/images/dummy-{i}-b.jpg") for i in range(1, 22)]
    edges[2] = bridge_edge(paths["a"], paths["b"])
    edges[3] = bridge_edge(paths["a"], paths["c"])
    edges[20] = bridge_edge(paths["d"], paths["e"])
    edge_table = tmp_path / "edges.csv"
    write_bridge_edge_table(edge_table, edges)
    review_a = tmp_path / "bridge_reviews_a.csv"
    review_b = tmp_path / "bridge_reviews_b.csv"
    write_bridge_reviews(
        review_a,
        [
            {"bridge_id": "#3", "image_path_a": paths["a"], "image_path_b": paths["b"], "decision": "same_device", "notes": "same object"},
            {"bridge_id": "#4", "image_path_a": paths["a"], "image_path_b": paths["c"], "decision": "same_device", "notes": "same object"},
        ],
    )

    write_bridge_reviews(
        review_b,
        [{"bridge_id": "#21", "image_path_a": paths["d"], "image_path_b": paths["e"], "decision": "same_device", "notes": "same phone; different images"}],
    )
    decisions = grouping.load_bridge_review_decisions([review_a, review_b], edge_table)

    assert decisions["#3"]["image_path_a"] == paths["a"]
    assert decisions["#3"]["image_path_b"] == paths["b"]
    assert decisions["#4"]["image_path_a"] == paths["a"]
    assert decisions["#4"]["image_path_b"] == paths["c"]
    assert decisions["#3"]["decision"] == decisions["#4"]["decision"] == "same_device"
    assert decisions["#21"]["image_path_a"] == paths["d"]
    assert decisions["#21"]["image_path_b"] == paths["e"]
    assert decisions["#21"]["decision"] == "same_device"


def test_bridge_review_loader_rejects_id_with_wrong_paths(tmp_path: Path) -> None:
    edges = [bridge_edge("test/images/a.jpg", "train/images/b.jpg")]
    edge_table = tmp_path / "edges.csv"
    write_bridge_edge_table(edge_table, edges)
    review_path = tmp_path / "bridge_review.csv"
    write_bridge_reviews(
        review_path,
        [{"bridge_id": "#1", "image_path_a": "test/images/a.jpg", "image_path_b": "valid/images/wrong.jpg", "decision": "same_device", "notes": ""}],
    )

    with pytest.raises(ValueError, match="do not match the exact pair"):
        grouping.load_bridge_review_decisions(review_path, edge_table)


def test_conflicting_bridge_decisions_across_files_are_rejected(tmp_path: Path) -> None:
    edges = [bridge_edge("test/images/a.jpg", "train/images/b.jpg")]
    edge_table = tmp_path / "edges.csv"
    write_bridge_edge_table(edge_table, edges)
    first = tmp_path / "review_same.csv"
    second = tmp_path / "review_uncertain.csv"
    write_bridge_reviews(first, [{"bridge_id": "#1", "image_path_a": "test/images/a.jpg", "image_path_b": "train/images/b.jpg", "decision": "same_device", "notes": "same object"}])
    write_bridge_reviews(second, [{"bridge_id": "#1", "image_path_a": "test/images/a.jpg", "image_path_b": "train/images/b.jpg", "decision": "uncertain", "notes": "conflicting assessment"}])

    with pytest.raises(ValueError, match="Conflicting decisions for bridge #1"):
        grouping.load_bridge_review_decisions([first, second], edge_table)


@pytest.mark.parametrize("non_same_decision", ["pending", "different_device", "uncertain"])
def test_same_device_review_is_a_split_constraint_but_other_decisions_are_not(tmp_path: Path, non_same_decision: str) -> None:
    a = record("test/images/a.jpg", "ha", "test")
    b = record("train/images/b.jpg", "hb", "train")
    c = record("valid/images/c.jpg", "hc", "valid")
    edges = [bridge_edge(a["image_path"], b["image_path"]), bridge_edge(a["image_path"], c["image_path"])]
    edge_table = tmp_path / "edges.csv"
    write_bridge_edge_table(edge_table, edges)
    review_path = tmp_path / "bridge_review.csv"
    write_bridge_reviews(
        review_path,
        [
            {"bridge_id": "#1", "image_path_a": a["image_path"], "image_path_b": b["image_path"], "decision": "same_device", "notes": "reviewed"},
            {"bridge_id": "#2", "image_path_a": a["image_path"], "image_path_b": c["image_path"], "decision": non_same_decision, "notes": "not enough evidence"},
        ],
    )
    decisions = grouping.load_bridge_review_decisions(review_path, edge_table)

    groups = grouping.assign_split_constraint_groups([a, b, c], [], decisions)

    assert groups[a["image_path"]] == groups[b["image_path"]]
    assert groups[a["image_path"]] != groups[c["image_path"]]
