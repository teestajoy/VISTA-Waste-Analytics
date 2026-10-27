"""Build read-only candidate groups for the archived Mobile/Battery_Waste images.

Images are read directly from the source ZIP. Output CSVs and the report are new,
versioned artifacts; existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "VISTA_mobile_dataset_raw.zip"
DEFAULT_REVIEW = PROJECT_ROOT / "data/processed/mobile_battery_related_image_review_v2.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/mobile_battery_auto_group_manifest_v1.csv"
DEFAULT_EDGES = PROJECT_ROOT / "data/processed/mobile_battery_auto_group_edges_v1.csv"
DEFAULT_REPORT = PROJECT_ROOT / "docs/mobile_battery_automated_grouping_v1.md"
TARGET_CLASSES = {"Battery_Waste", "Mobile"}
PHASH_CANDIDATE_MAX_DISTANCE = 10
STRONG_ORB_MIN_INLIERS = 20
STRONG_ORB_MIN_INLIER_FRACTION = 0.50
WEAK_ORB_MIN_INLIERS = 8
WEAK_ORB_MIN_INLIER_FRACTION = 0.35
LARGE_GROUP_REVIEW_SIZE = 12
TIMESTAMP_RE = re.compile(r"(20\d{6}_\d{6})")


def phash_bits(image: Image.Image) -> int:
    """Match the audit's 32x32 DCT / 8x8 low-frequency perceptual hash."""
    dct_matrix = np.cos(
        np.pi * (2 * np.arange(32)[:, None] + 1) * np.arange(32)[None, :] / 64
    )
    dct_matrix[:, 0] *= 1 / math.sqrt(2)
    gray = np.asarray(
        image.resize((32, 32), Image.Resampling.LANCZOS).convert("L"),
        dtype=np.float64,
    )
    transformed = (2 / 32) * dct_matrix.T @ gray @ dct_matrix
    low = transformed[:8, :8].flatten()
    median = float(np.median(low[1:]))
    return sum(int(value) << index for index, value in enumerate(low > median))


def classify_pair(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    phash_distance: int | None,
    orb_inliers: int,
    orb_inlier_fraction: float,
) -> dict[str, Any]:
    """Classify pair evidence without equating visual similarity with identity."""
    same_hash = bool(a["sha256"] == b["sha256"])
    same_timestamp = bool(a.get("timestamp") and a.get("timestamp") == b.get("timestamp"))
    labels_a = set(a.get("class_labels", ()))
    labels_b = set(b.get("class_labels", ()))
    class_conflict = bool(labels_a and labels_b and labels_a.isdisjoint(labels_b))
    strong_visual = bool(
        phash_distance is not None
        and phash_distance <= PHASH_CANDIDATE_MAX_DISTANCE
        and orb_inliers >= STRONG_ORB_MIN_INLIERS
        and orb_inlier_fraction >= STRONG_ORB_MIN_INLIER_FRACTION
    )
    weak_visual = bool(
        (phash_distance is not None and phash_distance <= PHASH_CANDIDATE_MAX_DISTANCE)
        or (
            orb_inliers >= WEAK_ORB_MIN_INLIERS
            and orb_inlier_fraction >= WEAK_ORB_MIN_INLIER_FRACTION
        )
    )
    if same_hash:
        level = "exact_duplicate"
    elif strong_visual:
        level = "strong_candidate"
    elif same_timestamp or weak_visual:
        level = "weak_match"
    else:
        level = "unrelated"

    reasons = []
    if same_hash:
        reasons.append("identical_sha256")
    if same_timestamp:
        reasons.append("same_filename_timestamp")
    if phash_distance is not None and phash_distance <= PHASH_CANDIDATE_MAX_DISTANCE:
        reasons.append("phash_within_candidate_radius")
    if orb_inliers >= WEAK_ORB_MIN_INLIERS and orb_inlier_fraction >= WEAK_ORB_MIN_INLIER_FRACTION:
        reasons.append("orb_ransac_geometric_support")
    if strong_visual:
        reasons.append("strong_phash_and_orb_ransac")

    timestamp_visual_conflict = bool(
        same_timestamp
        and not strong_visual
        and phash_distance is not None
        and phash_distance > PHASH_CANDIDATE_MAX_DISTANCE
        and orb_inliers < WEAK_ORB_MIN_INLIERS
    )
    conflict_reasons = []
    if class_conflict:
        conflict_reasons.append("different_original_class_labels")
    if timestamp_visual_conflict:
        conflict_reasons.append("timestamp_match_but_visual_features_disagree")

    if level == "unrelated":
        reasons.append("no_candidate_evidence")
    return {
        "match_level": level,
        "sha256_equal": same_hash,
        "same_timestamp": same_timestamp,
        "phash_distance": phash_distance,
        "orb_inliers": orb_inliers,
        "orb_inlier_fraction": orb_inlier_fraction,
        "class_conflict": class_conflict,
        "evidence_conflict": bool(conflict_reasons),
        "conflict_reasons": ";".join(conflict_reasons),
        "evidence_reasons": ";".join(reasons),
    }


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def assign_strong_groups(
    records: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Union exact and strong links only; weak links remain explicit bridges."""
    paths = [record["image_path"] for record in records]
    uf = UnionFind(paths)
    for edge in edges:
        if edge["match_level"] in {"exact_duplicate", "strong_candidate"}:
            uf.union(edge["image_path_a"], edge["image_path_b"])

    components: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        components[uf.find(path)].append(path)
    ordered_components = sorted(components.values(), key=lambda members: min(members))
    group_by_path: dict[str, str] = {}
    for index, members in enumerate(ordered_components, start=1):
        group_id = f"AUTO-G{index:04d}"
        for path in members:
            group_by_path[path] = group_id

    weak_neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge["match_level"] != "weak_match":
            continue
        group_a = group_by_path[edge["image_path_a"]]
        group_b = group_by_path[edge["image_path_b"]]
        if group_a != group_b:
            weak_neighbors[group_a].add(group_b)
            weak_neighbors[group_b].add(group_a)
    return group_by_path, weak_neighbors


def detect_split_leakage(
    records: list[dict[str, Any]], group_by_path: Mapping[str, str]
) -> dict[str, set[str]]:
    """Return provisional groups whose original records span multiple source splits."""
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        splits_by_group[group_by_path[record["image_path"]]].add(record["original_split"])
    return {group_id: splits for group_id, splits in splits_by_group.items() if len(splits) > 1}


def find_strong_graph_bridges(
    records: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> set[tuple[str, str]]:
    """Return articulation edges in the strong-link graph for manual boundary review."""
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge["match_level"] in {"exact_duplicate", "strong_candidate"}:
            a, b = edge["image_path_a"], edge["image_path_b"]
            adjacency[a].append(b)
            adjacency[b].append(a)

    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    bridges: set[tuple[str, str]] = set()
    counter = 0

    def visit(node: str, parent: str | None) -> None:
        nonlocal counter
        counter += 1
        discovery[node] = low[node] = counter
        for neighbor in adjacency[node]:
            if neighbor == parent:
                continue
            if neighbor not in discovery:
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(tuple(sorted((node, neighbor))))
            else:
                low[node] = min(low[node], discovery[neighbor])

    for record in records:
        path = record["image_path"]
        if path not in discovery and adjacency[path]:
            visit(path, None)
    return bridges


def load_review_rows(review_path: Path) -> dict[str, dict[str, str]]:
    if not review_path.is_file():
        return {}
    with review_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    return {row["image_path"]: row for row in rows if row.get("image_path")}


def collect_records(archive_path: Path) -> list[dict[str, Any]]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Mobile/E-waste archive not found: {archive_path}")
    records_by_path: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        if "data.yaml" not in names:
            raise ValueError("Archive is missing data.yaml; cannot validate class IDs")
        dataset_config = yaml.safe_load(archive.read("data.yaml"))
        class_names = dataset_config.get("names", [])
        if isinstance(class_names, dict):
            class_names = [class_names[index] for index in sorted(class_names)]
        class_id_by_name = {str(name): index for index, name in enumerate(class_names)}
        missing = TARGET_CLASSES.difference(class_id_by_name)
        if missing:
            raise ValueError(f"Archive data.yaml lacks required classes: {sorted(missing)}")

        images_by_split_stem: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name in names:
            parts = name.split("/")
            if len(parts) == 3 and parts[0] in {"train", "valid", "test"} and parts[1] == "images":
                images_by_split_stem[(parts[0], Path(parts[2]).stem)].append(name)

        target_ids = {class_id_by_name[name]: name for name in TARGET_CLASSES}
        target_ids_by_split: dict[str, Counter[str]] = defaultdict(Counter)
        for label_path in sorted(name for name in names if "/labels/" in name and name.endswith(".txt")):
            split = label_path.split("/", 1)[0]
            if split not in {"train", "valid", "test"}:
                continue
            label_ids: set[int] = set()
            for line in archive.read(label_path).decode("utf-8", errors="replace").splitlines():
                columns = line.split()
                if columns and columns[0].isdigit():
                    label_ids.add(int(columns[0]))
            found_classes = sorted(target_ids[value] for value in label_ids if value in target_ids)
            if not found_classes:
                continue
            target_ids_by_split[split].update(found_classes)
            stem = Path(label_path).stem
            matching_images = sorted(images_by_split_stem.get((split, stem), []))
            if len(matching_images) != 1:
                raise ValueError(
                    f"Expected one image for target annotation {label_path}; found {matching_images}"
                )
            image_path = matching_images[0]
            image_bytes = archive.read(image_path)
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image.load()
                gray_bytes = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if gray_bytes is None:
                    raise ValueError("OpenCV could not decode image bytes")
            except Exception as exc:
                raise ValueError(f"Could not read image {image_path}: {exc}") from exc

            timestamp_match = TIMESTAMP_RE.search(Path(image_path).name)
            item = records_by_path.setdefault(
                image_path,
                {
                    "image_path": image_path,
                    "original_split": split,
                    "class_labels": set(),
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "timestamp": timestamp_match.group(1) if timestamp_match else "",
                    "phash": phash_bits(image),
                    "gray": cv2.resize(gray_bytes, (384, 384)),
                    "image_size_bytes": len(image_bytes),
                },
            )
            item["class_labels"].update(found_classes)

    orb = cv2.ORB_create(nfeatures=500, fastThreshold=10)
    for record in records_by_path.values():
        record["keypoints"], record["descriptors"] = orb.detectAndCompute(record["gray"], None)
    records = list(records_by_path.values())
    records.sort(key=lambda record: record["image_path"])
    return records


def compare_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    edges: list[dict[str, Any]] = []
    total_pairs = len(records) * (len(records) - 1) // 2
    for index, a in enumerate(records):
        for b in records[index + 1 :]:
            phash_distance = (a["phash"] ^ b["phash"]).bit_count()
            same_timestamp = bool(a["timestamp"] and a["timestamp"] == b["timestamp"])
            sha_equal = a["sha256"] == b["sha256"]
            # Existing audit's pHash radius limits ORB work. Exact hashes and filename
            # timestamp pairs are also checked even where pHash is outside the radius.
            if not sha_equal and phash_distance > PHASH_CANDIDATE_MAX_DISTANCE and not same_timestamp:
                continue
            good_matches = []
            if a["descriptors"] is not None and b["descriptors"] is not None:
                for pair in matcher.knnMatch(a["descriptors"], b["descriptors"], k=2):
                    if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
                        good_matches.append(pair[0])
            inliers = 0
            fraction = 0.0
            if len(good_matches) >= 4:
                points_a = np.float32(
                    [a["keypoints"][match.queryIdx].pt for match in good_matches]
                ).reshape(-1, 1, 2)
                points_b = np.float32(
                    [b["keypoints"][match.trainIdx].pt for match in good_matches]
                ).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 4.0)
                if mask is not None:
                    inliers = int(mask.sum())
                    fraction = inliers / len(good_matches)
            evidence = classify_pair(
                a,
                b,
                phash_distance=phash_distance,
                orb_inliers=inliers,
                orb_inlier_fraction=fraction,
            )
            if evidence["match_level"] == "unrelated":
                continue
            edges.append(
                {
                    "image_path_a": a["image_path"],
                    "image_path_b": b["image_path"],
                    "original_class_a": ";".join(sorted(a["class_labels"])),
                    "original_class_b": ";".join(sorted(b["class_labels"])),
                    "original_split_a": a["original_split"],
                    "original_split_b": b["original_split"],
                    "sha256_a": a["sha256"],
                    "sha256_b": b["sha256"],
                    "phash_distance": phash_distance,
                    "orb_good_matches": len(good_matches),
                    **evidence,
                }
            )
    return edges, total_pairs


def bridge_pairs(edges: list[dict[str, Any]], bridges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {
        tuple(sorted((edge["image_path_a"], edge["image_path_b"])))
        for edge in edges
        if tuple(sorted((edge["image_path_a"], edge["image_path_b"]))) in bridges
    }


def build_artifacts(
    archive_path: Path,
    review_path: Path,
) -> tuple[str, str, str, dict[str, Any]]:
    records = collect_records(archive_path)
    edges, total_pairs = compare_records(records)
    group_by_path, weak_neighbors = assign_strong_groups(records, edges)
    split_leakage = detect_split_leakage(records, group_by_path)
    strong_bridges = find_strong_graph_bridges(records, edges)
    strong_bridge_set = bridge_pairs(edges, strong_bridges)
    review_by_path = load_review_rows(review_path)

    records_by_path = {record["image_path"]: record for record in records}
    members_by_group: dict[str, list[str]] = defaultdict(list)
    for path, group_id in group_by_path.items():
        members_by_group[group_id].append(path)
    edges_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        edges_by_path[edge["image_path_a"]].append(edge)
        edges_by_path[edge["image_path_b"]].append(edge)
    bridge_members_by_path: dict[str, set[str]] = defaultdict(set)
    for left, right in strong_bridge_set:
        bridge_members_by_path[left].add(right)
        bridge_members_by_path[right].add(left)

    manifest_fields = [
        "image_path", "original_class", "original_split", "sha256", "timestamp_token",
        "strong_group_id", "strong_group_size", "strong_group_splits", "strong_group_status",
        "weak_bridge_group_ids", "weak_bridge_paths", "split_blocked_by_weak_link",
        "strong_bridge_paths", "strong_bridge_review_required", "exact_duplicate_group_id",
        "candidate_match_levels", "candidate_evidence_summary", "evidence_conflict",
        "human_decision_from_v2", "review_notes_from_v2", "new_split_assignment",
    ]
    exact_by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        exact_by_hash[record["sha256"]].append(record["image_path"])
    exact_group_for_path: dict[str, str] = {}
    exact_groups = sorted(
        (paths for paths in exact_by_hash.values() if len(paths) > 1),
        key=lambda paths: min(paths),
    )
    for number, paths in enumerate(exact_groups, start=1):
        for path in paths:
            exact_group_for_path[path] = f"EXACT-{number:04d}"

    manifest_rows = []
    for path in sorted(records_by_path):
        record = records_by_path[path]
        group_id = group_by_path[path]
        group_members = members_by_group[group_id]
        group_splits = sorted({records_by_path[member]["original_split"] for member in group_members})
        local_edges = edges_by_path[path]
        match_levels = sorted({edge["match_level"] for edge in local_edges})
        evidence_summaries = []
        conflict_summaries = []
        for edge in local_edges:
            other = edge["image_path_b"] if edge["image_path_a"] == path else edge["image_path_a"]
            evidence_summaries.append(
                f"{edge['match_level']}:{other} (pHash={edge['phash_distance']}, "
                f"ORB inliers={edge['orb_inliers']}/{edge['orb_good_matches']}, "
                f"fraction={edge['orb_inlier_fraction']:.3f}, timestamp_match={str(edge['same_timestamp']).lower()})"
            )
            if edge["evidence_conflict"]:
                conflict_summaries.append(f"{other}:{edge['conflict_reasons']}")
        review = review_by_path.get(path, {})
        status = "singleton" if len(group_members) == 1 else "strong_candidate_component"
        if len(group_members) > LARGE_GROUP_REVIEW_SIZE:
            status = "large_component_review_required"
        manifest_rows.append(
            {
                "image_path": path,
                "original_class": ";".join(sorted(record["class_labels"])),
                "original_split": record["original_split"],
                "sha256": record["sha256"],
                "timestamp_token": record["timestamp"],
                "strong_group_id": group_id,
                "strong_group_size": len(group_members),
                "strong_group_splits": ";".join(group_splits),
                "strong_group_status": status,
                "weak_bridge_group_ids": ";".join(sorted(weak_neighbors.get(group_id, set()))),
                "weak_bridge_paths": ";".join(
                    sorted(
                        other
                        for edge in local_edges
                        if edge["match_level"] == "weak_match"
                        for other in [edge["image_path_b"] if edge["image_path_a"] == path else edge["image_path_a"]]
                    )
                ),
                "split_blocked_by_weak_link": bool(weak_neighbors.get(group_id)),
                "strong_bridge_paths": ";".join(sorted(bridge_members_by_path.get(path, set()))),
                "strong_bridge_review_required": bool(bridge_members_by_path.get(path)),
                "exact_duplicate_group_id": exact_group_for_path.get(path, ""),
                "candidate_match_levels": ";".join(match_levels),
                "candidate_evidence_summary": " | ".join(sorted(set(evidence_summaries))),
                "evidence_conflict": " | ".join(sorted(set(conflict_summaries))),
                "human_decision_from_v2": review.get("human_decision", ""),
                "review_notes_from_v2": review.get("notes", ""),
                "new_split_assignment": "",
            }
        )

    edge_fields = [
        "image_path_a", "image_path_b", "original_class_a", "original_class_b",
        "original_split_a", "original_split_b", "sha256_a", "sha256_b",
        "sha256_equal", "phash_distance", "orb_good_matches", "orb_inliers",
        "orb_inlier_fraction", "same_timestamp", "class_conflict", "evidence_conflict",
        "conflict_reasons", "match_level", "evidence_reasons", "strong_graph_bridge",
        "split_crossing",
    ]
    edge_rows = []
    for edge in edges:
        edge_copy = dict(edge)
        pair = tuple(sorted((edge["image_path_a"], edge["image_path_b"])))
        edge_copy["strong_graph_bridge"] = pair in strong_bridge_set
        edge_copy["split_crossing"] = edge["original_split_a"] != edge["original_split_b"]
        edge_rows.append(edge_copy)

    component_summaries = []
    for group_id, paths in sorted(members_by_group.items()):
        component_summaries.append(
            {
                "group_id": group_id,
                "members": sorted(paths),
                "size": len(paths),
                "classes": sorted({label for path in paths for label in records_by_path[path]["class_labels"]}),
                "splits": sorted({records_by_path[path]["original_split"] for path in paths}),
                "has_bridge": any(path in bridge_members_by_path for path in paths),
            }
        )
    summary = {
        "records": records,
        "manifest_rows": manifest_rows,
        "manifest_fields": manifest_fields,
        "edge_rows": edge_rows,
        "edge_fields": edge_fields,
        "components": component_summaries,
        "split_leakage": split_leakage,
        "candidate_edges": edges,
        "total_pairs": total_pairs,
        "strong_bridges": strong_bridge_set,
        "weak_links": [edge for edge in edges if edge["match_level"] == "weak_match"],
        "conflict_edges": [edge for edge in edges if edge["evidence_conflict"]],
        "exact_groups": exact_groups,
        "reviewed_row_count": len(review_by_path),
    }
    manifest_csv = csv_text(manifest_fields, manifest_rows)
    edge_csv = csv_text(edge_fields, edge_rows)
    report = report_text(archive_path, summary)
    return manifest_csv, edge_csv, report, summary


def csv_text(fields: list[str], rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def report_text(archive_path: Path, summary: dict[str, Any]) -> str:
    records = summary["records"]
    components = summary["components"]
    edges = summary["candidate_edges"]
    level_counts = Counter(edge["match_level"] for edge in edges)
    split_leaking_components = [
        group for group in components if group["group_id"] in summary["split_leakage"]
    ]
    weak_crossing = [
        edge for edge in summary["weak_links"] if edge["original_split_a"] != edge["original_split_b"]
    ]
    hard_crossing = [
        edge
        for edge in edges
        if edge["match_level"] in {"exact_duplicate", "strong_candidate"}
        and edge["original_split_a"] != edge["original_split_b"]
    ]
    sizes = Counter(group["size"] for group in components)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for label in record["class_labels"]:
            split_counts[label][record["original_split"]] += 1
    lines = [
        "# Mobile and Battery_Waste automated grouping report (v1)",
        "",
        f"- Source archive read in place: `{archive_path}`",
        "- No archive members were extracted, moved, renamed, or modified.",
        f"- Target image records: **{len(records)}**; all output split assignments are intentionally blank.",
        f"- Pairwise combinations: {summary['total_pairs']:,}; candidate evidence edges saved: {len(edges):,}; no-evidence pairs omitted from the edge table: {summary['total_pairs'] - len(edges):,}.",
        f"- Starting review worksheet rows carried forward: {summary['reviewed_row_count']}.",
        "",
        "## Original split counts",
        "",
        "| Class | train | valid | test | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("Mobile", "Battery_Waste"):
        counts = split_counts[label]
        lines.append(
            f"| {label} | {counts['train']} | {counts['valid']} | {counts['test']} | "
            f"{sum(counts.values())} |"
        )
    lines += [
        "",
        "## Evidence rules",
        "",
        "- **Exact duplicate:** original ZIP-member SHA-256 values are identical. This is a file-level match, not physical-device identity.",
        f"- **Strong candidate:** pHash Hamming distance ≤{PHASH_CANDIDATE_MAX_DISTANCE} AND ORB/RANSAC has at least {STRONG_ORB_MIN_INLIERS} inliers with inlier fraction ≥{STRONG_ORB_MIN_INLIER_FRACTION:.2f}. These thresholds reuse the prior audit. Strong links form provisional candidate groups.",
        "- **Weak match:** same parsed filename timestamp, or perceptual/keypoint evidence that does not meet the strong threshold. Weak matches do not merge provisional strong groups; they are recorded as split-blocking bridges pending review.",
        "- **Unrelated:** no exact hash, same-timestamp, pHash-shortlist, or qualifying ORB/RANSAC evidence. These pairs are counted but omitted from the edge CSV to keep the artifact compact.",
        "- Exact timestamps, pHash, ORB/RANSAC matches, and image similarity do not prove images depict the same physical device. A timestamp/visual disagreement and cross-class label conflict are explicitly flagged.",
        "",
        "## Group and match counts",
        "",
        f"- Exact duplicate groups: **{len(summary['exact_groups'])}**; duplicate member images: **{sum(len(group) for group in summary['exact_groups'])}**.",
        f"- Strong candidate groups (including singleton groups): **{len(components)}**; non-singleton strong groups: **{sum(group['size'] > 1 for group in components)}**; members in non-singleton strong groups: **{sum(group['size'] for group in components if group['size'] > 1)}**.",
        f"- Strong candidate pair edges: **{level_counts['strong_candidate']}**; weak match/bridge edges: **{level_counts['weak_match']}**; exact-match edges: **{level_counts['exact_duplicate']}**.",
        f"- Strong graph bridge edges requiring boundary review: **{len(summary['strong_bridges'])}**.",
        f"- Evidence conflict edges: **{len(summary['conflict_edges'])}**; cross-class conflict edges: **{sum(edge['class_conflict'] for edge in summary['conflict_edges'])}**.",
        f"- Strong components larger than {LARGE_GROUP_REVIEW_SIZE} images (review flagged): **{sum(group['size'] > LARGE_GROUP_REVIEW_SIZE for group in components)}**.",
        "",
        "### Strong-group size distribution",
        "",
        "| Images per group | Number of groups |",
        "|---:|---:|",
    ]
    for size, count in sorted(sizes.items()):
        lines.append(f"| {size} | {count} |")
    lines += [
        "",
        "## Original split leakage indicators",
        "",
        f"- Strong provisional components spanning more than one original split: **{len(split_leaking_components)}**.",
        f"- Strong/exact candidate edges directly crossing original splits: **{len(hard_crossing)}**.",
        f"- Weak candidate bridges crossing original splits: **{len(weak_crossing)}**.",
        "- Original split assignments are retained only as source metadata. No new train/validation/test membership is assigned.",
        "- A future split creator must keep each strong group together and must also avoid splitting across unresolved weak bridges; otherwise resolve or document the bridge before assigning splits.",
        "",
        "## Strong components and bridge boundaries",
        "",
    ]
    for group in components:
        if group["size"] <= 1 and not group["has_bridge"]:
            continue
        lines.append(
            f"### {group['group_id']} — {group['size']} images; classes={','.join(group['classes'])}; "
            f"original splits={','.join(group['splits'])}; status="
            f"{'large_component_review_required' if group['size'] > LARGE_GROUP_REVIEW_SIZE else 'provisional_strong_component'}"
        )
        for member in group["members"]:
            lines.append(f"- `{member}`")
        for edge in summary["candidate_edges"]:
            if edge["match_level"] == "weak_match" and edge["image_path_a"] in group["members"]:
                lines.append(
                    f"- Weak bridge to `{edge['image_path_b']}` (pHash={edge['phash_distance']}, "
                    f"ORB inliers={edge['orb_inliers']}/{edge['orb_good_matches']}, "
                    f"same timestamp={edge['same_timestamp']}, conflict={edge['conflict_reasons'] or 'none'})."
                )
            elif edge["match_level"] == "weak_match" and edge["image_path_b"] in group["members"]:
                lines.append(
                    f"- Weak bridge to `{edge['image_path_a']}` (pHash={edge['phash_distance']}, "
                    f"ORB inliers={edge['orb_inliers']}/{edge['orb_good_matches']}, "
                    f"same timestamp={edge['same_timestamp']}, conflict={edge['conflict_reasons'] or 'none'})."
                )
        for left, right in sorted(summary["strong_bridges"]):
            if left in group["members"] and right in group["members"]:
                lines.append(f"- **Strong graph bridge to review:** `{left}` ↔ `{right}`.")
        lines.append("")
    if not any(group["size"] > 1 or group["has_bridge"] for group in components):
        lines.append("No non-singleton strong components or weak bridges were found.\n")
    lines += [
        "## Manifest and use",
        "",
        "`data/processed/mobile_battery_auto_group_manifest_v1.csv` contains every target image, original class and split, exact SHA-256, timestamp token, provisional strong-group ID, group size/status, weak-bridge constraints, strong graph bridge indicators, and any existing v2 review decision/notes. `new_split_assignment` is blank for every record.",
        "`data/processed/mobile_battery_auto_group_edges_v1.csv` contains exact, strong, and weak pair evidence with metrics and conflict flags.",
        "",
        "## Review cautions and limitations",
        "",
        "- Strong connected components are candidate image-family groupings, not physical-device IDs. Bridge edges can join otherwise distinct subgroups; inspect every flagged bridge before treating a component as one indivisible physical object.",
        "- Weak bridges are retained as co-assignment constraints for future split planning but are not merged into strong groups. This avoids silently turning weak transitive links into large identity groups while still flagging potential leakage.",
        "- Filename timestamps are weak evidence only. Same timestamps can be shared by augmented exports, but do not establish device identity or capture session identity.",
        "- Existing human decisions and notes are copied from review v2 where an image path is present. No new reviewer decisions, labels, device IDs, or capture-session IDs are created.",
        "- This is a grouping audit, not a model evaluation. Review labels and all flagged bridges before using the manifest to create new splits.",
        "",
    ]
    return "\n".join(lines)


def write_outputs_exclusive(outputs: Mapping[Path, str]) -> None:
    paths = [Path(path) for path in outputs]
    exists = [path for path in paths if path.exists()]
    if exists:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(str(path) for path in exists)
        )
    created: list[Path] = []
    try:
        for path, content in outputs.items():
            path = Path(path)
            if not path.parent.is_dir():
                raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
            with path.open("x", encoding="utf-8-sig" if path.suffix.lower() == ".csv" else "utf-8", newline="") as stream:
                created.append(path)
                stream.write(content)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = [args.manifest, args.edges, args.report]
    existing = [Path(path) for path in outputs if Path(path).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(str(path) for path in existing)
        )
    manifest, edges, report, summary = build_artifacts(args.archive, args.review_csv)
    write_outputs_exclusive({args.manifest: manifest, args.edges: edges, args.report: report})
    print(f"Images audited: {len(summary['records'])}")
    print(f"Strong groups: {len(summary['components'])}; weak bridges: {len(summary['weak_links'])}")
    print(f"Created manifest: {args.manifest}")
    print(f"Created evidence edges: {args.edges}")
    print(f"Created report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
