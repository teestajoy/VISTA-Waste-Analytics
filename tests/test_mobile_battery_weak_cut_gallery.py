from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "mobile_battery_weak_cut_review_v1"
    / "review_store.py"
)
SPEC = importlib.util.spec_from_file_location("weak_cut_review_store", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
review_store = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_store)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_reviews():
    return [
        {
            "edge_id": edge_id,
            "image_path_a": paths[0],
            "image_path_b": paths[1],
            "decision": "pending",
            "notes": "",
        }
        for edge_id, paths in review_store.EDGE_PATHS.items()
    ]


def test_loads_fixed_edge_ids_with_exact_paths_and_cand_membership(tmp_path):
    edge_rows = []
    manifest_rows = []
    for edge_id, (path_a, path_b) in review_store.EDGE_PATHS.items():
        edge_rows.append({
            "image_path_a": path_a,
            "image_path_b": path_b,
            "match_level": "weak_match",
            "class_conflict": "True",
        })
        for path, group, label in ((path_a, f"G-{edge_id}-A", "Battery_Waste"), (path_b, f"G-{edge_id}-B", "Mobile")):
            manifest_rows.append({
                "image_path": path,
                "candidate_graph_component_id": "CAND-0002",
                "candidate_graph_component_size": "214",
                "strong_group_id": group,
                "original_class": label,
                "original_split": "train",
            })
    edge_csv = tmp_path / "edges.csv"
    manifest_csv = tmp_path / "manifest.csv"
    write_csv(edge_csv, edge_rows, ["image_path_a", "image_path_b", "match_level", "class_conflict"])
    write_csv(manifest_csv, manifest_rows, ["image_path", "candidate_graph_component_id", "candidate_graph_component_size", "strong_group_id", "original_class", "original_split"])

    loaded = review_store.load_verified_edges(edge_csv, manifest_csv)

    assert set(loaded) == {"W-CUT-01", "W-CUT-03"}
    for edge_id, paths in review_store.EDGE_PATHS.items():
        assert (loaded[edge_id]["image_path_a"], loaded[edge_id]["image_path_b"]) == paths
        assert loaded[edge_id]["candidate_graph_component_id"] == "CAND-0002"


def test_review_save_persists_both_edge_decisions(tmp_path):
    destination = tmp_path / "weak_cut_review_v1.csv"
    reviews = sample_reviews()
    reviews[0]["decision"] = "uncertain"
    reviews[0]["notes"] = "Cross-class visual match; identity unconfirmed."

    saved = review_store.save_reviews_exclusive(destination, reviews)

    assert saved == destination
    with destination.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows == reviews


def test_existing_review_csv_is_never_overwritten(tmp_path):
    destination = tmp_path / "weak_cut_review_v1.csv"
    original = b"preserve this review\n"
    destination.write_bytes(original)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        review_store.save_reviews_exclusive(destination, sample_reviews())

    assert destination.read_bytes() == original


def test_save_rejects_mismatched_edge_paths(tmp_path):
    reviews = sample_reviews()
    reviews[0]["image_path_b"] = "train/images/unrelated.jpg"

    with pytest.raises(ValueError, match="do not match the registered pair"):
        review_store.save_reviews_exclusive(tmp_path / "weak_cut_review.csv", reviews)
