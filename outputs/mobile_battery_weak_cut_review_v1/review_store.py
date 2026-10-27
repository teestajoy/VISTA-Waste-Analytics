"""Validation and exclusive persistence for the W-CUT-01/W-CUT-03 gallery."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, Mapping


EDGE_PATHS = {
    "W-CUT-01": (
        "train/images/IMG_20250803_215704_jpg.rf.0d941aee39572d80f929eb96b4640aaa.jpg",
        "train/images/IMG_20250805_203846_jpg.rf.9e943104abcc04412073581826afec4d.jpg",
    ),
    "W-CUT-03": (
        "train/images/IMG_20250803_220046_jpg.rf.e75fd5549c943e12ae1d5c9f747e2c80.jpg",
        "train/images/IMG_20250822_231220_jpg.rf.9c02b05d7b7b49ba3c228d2dc9bf9b7e.jpg",
    ),
}
REVIEW_STATES = ("pending", "same_device", "different_device", "uncertain")
REVIEW_FIELDS = ("edge_id", "image_path_a", "image_path_b", "decision", "notes")


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_verified_edges(edge_csv: Path, manifest_csv: Path) -> dict[str, dict[str, str]]:
    """Resolve fixed review IDs against the saved edge and manifest tables."""
    edge_rows = read_csv(edge_csv)
    manifest = {row["image_path"]: row for row in read_csv(manifest_csv)}
    resolved: dict[str, dict[str, str]] = {}
    for edge_id, expected_paths in EDGE_PATHS.items():
        matches = [
            row for row in edge_rows
            if (row.get("image_path_a"), row.get("image_path_b")) == expected_paths
            or (row.get("image_path_b"), row.get("image_path_a")) == expected_paths
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one v2 edge for {edge_id}; found {len(matches)}")
        edge = matches[0]
        if edge.get("match_level") != "weak_match":
            raise ValueError(f"{edge_id} is not classified as a weak_match in the v2 edge table")
        if edge.get("class_conflict", "").strip().lower() not in {"true", "1"}:
            raise ValueError(f"{edge_id} no longer has the expected cross-class evidence")
        a, b = edge["image_path_a"], edge["image_path_b"]
        if a not in manifest or b not in manifest:
            raise ValueError(f"{edge_id} endpoint is missing from the v2 manifest")
        manifest_a, manifest_b = manifest[a], manifest[b]
        if (
            manifest_a.get("candidate_graph_component_id") != "CAND-0002"
            or manifest_b.get("candidate_graph_component_id") != "CAND-0002"
        ):
            raise ValueError(f"{edge_id} endpoints are not both in CAND-0002")
        if manifest_a.get("strong_group_id") == manifest_b.get("strong_group_id"):
            raise ValueError(f"{edge_id} endpoints are expected to join distinct strong groups")
        resolved[edge_id] = {
            **edge,
            "edge_id": edge_id,
            "candidate_graph_component_id": "CAND-0002",
            "candidate_graph_component_size": manifest_a.get("candidate_graph_component_size", ""),
            "strong_group_id_a": manifest_a.get("strong_group_id", ""),
            "strong_group_id_b": manifest_b.get("strong_group_id", ""),
            "cut_component_sizes": "209 + 5" if edge_id == "W-CUT-01" else "207 + 7",
        }
    return resolved


def serialize_reviews(reviews: Iterable[Mapping[str, str]]) -> str:
    rows = [dict(row) for row in reviews]
    if len(rows) != len(EDGE_PATHS) or {row.get("edge_id", "") for row in rows} != set(EDGE_PATHS):
        raise ValueError("Review export must contain exactly one row each for W-CUT-01 and W-CUT-03")
    for row in rows:
        edge_id = row["edge_id"]
        actual = tuple(sorted(((row.get("image_path_a") or "").strip(), (row.get("image_path_b") or "").strip())))
        expected = tuple(sorted(EDGE_PATHS[edge_id]))
        if not all(actual) or actual != expected:
            raise ValueError(f"Image paths do not match the registered pair for {edge_id}")
        if row.get("decision") not in REVIEW_STATES:
            raise ValueError(f"Unsupported decision for {edge_id}: {row.get('decision')!r}")
        row["notes"] = row.get("notes", "")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in REVIEW_FIELDS} for row in rows)
    return output.getvalue()


def save_reviews_exclusive(destination: Path, reviews: Iterable[Mapping[str, str]]) -> Path:
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
            f"Refusing to overwrite existing review file: {destination}. Choose a new versioned CSV."
        ) from exc
    return destination
