"""Focused visual review for weak edges W-CUT-01 and W-CUT-03."""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from review_store import REVIEW_STATES, load_verified_edges, save_reviews_exclusive

EDGE_CSV = PROJECT_ROOT / "data/processed/mobile_battery_auto_group_edges_v2.csv"
MANIFEST_CSV = PROJECT_ROOT / "data/processed/mobile_battery_auto_group_manifest_v2.csv"
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "VISTA_mobile_dataset_raw.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REVIEW_NAME = "mobile_battery_cand0002_weak_cut_review_v1.csv"


def safe_output_path(filename: str) -> Path:
    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise ValueError("Enter a CSV filename only (no folder path)")
    return PROCESSED_DIR / filename


st.set_page_config(page_title="CAND-0002 weak edge review", layout="wide")
st.title("CAND-0002 weak-edge review")
st.caption("Review only W-CUT-01 and W-CUT-03. Images are read directly from the ZIP in memory; no extraction occurs.")
st.warning(
    "Both pairs cross source classes. Cross-class visual similarity does not prove the images show the same device. "
    "No review decision is preselected beyond pending."
)

try:
    edges = load_verified_edges(EDGE_CSV, MANIFEST_CSV)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not verify the proposed edge pairs against the v2 artifacts: {exc}")
    st.stop()

archive_text = st.text_input("Original ZIP path", value=str(DEFAULT_ARCHIVE))
archive_path = Path(archive_text).expanduser()
if not archive_path.is_file():
    st.error(f"Original ZIP unavailable: {archive_path}. No extraction was attempted.")
    st.stop()

decisions: dict[str, str] = {}
notes: dict[str, str] = {}
try:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for edge_id in ("W-CUT-01", "W-CUT-03"):
            edge = edges[edge_id]
            st.header(edge_id)
            st.info(
                f"Both images belong to {edge['candidate_graph_component_id']} "
                f"({edge['candidate_graph_component_size']} images), but to distinct strong groups "
                f"{edge['strong_group_id_a']} and {edge['strong_group_id_b']}. This weak edge is the only graph "
                f"connection between the cut sides shown by the saved analysis; removing it counterfactually "
                f"would yield {edge['cut_component_sizes']} images. This does not establish that removal is safe."
            )
            columns = st.columns(2)
            endpoints = ((edge["image_path_a"], "a"), (edge["image_path_b"], "b"))
            for column, (path, side) in zip(columns, endpoints):
                with column:
                    st.markdown(f"**{path}**")
                    st.write(
                        f"Source class: `{edge[f'original_class_{side}']}` · "
                        f"Original split: `{edge[f'original_split_{side}']}`"
                    )
                    try:
                        image_bytes = archive.read(path)
                        actual_sha = hashlib.sha256(image_bytes).hexdigest()
                        if actual_sha != edge[f"sha256_{side}"]:
                            st.error("ZIP image SHA-256 does not match the v2 edge record")
                        st.caption(f"SHA-256: `{actual_sha}`")
                        st.image(Image.open(io.BytesIO(image_bytes)), caption=path, use_container_width=True)
                    except (KeyError, OSError, ValueError) as exc:
                        st.error(f"Image unavailable or unreadable in ZIP: {path} ({exc})")

            st.subheader("Pair evidence from v2")
            st.write(f"SHA-256 comparison: **{'match' if edge['sha256_equal'].lower() in {'true', '1'} else 'different'}**")
            st.write(f"pHash Hamming distance: **{edge['phash_distance']}**")
            st.write(
                f"ORB/RANSAC: **{edge['orb_inliers']} inliers / {edge['orb_good_matches']} good matches "
                f"({float(edge['orb_inlier_fraction']):.3f} inlier fraction)**"
            )
            st.write(f"Filename timestamp match: **{'yes' if edge['same_timestamp'].lower() in {'true', '1'} else 'no'}**")
            st.write(f"Class conflict: **{edge['class_conflict']}** · Evidence conflict: **{edge['evidence_conflict']}**")
            st.caption(f"Evidence flags: {edge['evidence_reasons'] or '(none)'}; conflict reason: {edge['conflict_reasons'] or '(none)'}")
            st.warning(
                "This is `weak_match`: pHash is within the candidate radius, but the strong rule also requires "
                "at least 20 ORB/RANSAC inliers and a 0.50 inlier fraction. This edge does not meet the inlier minimum."
            )
            decisions[edge_id] = st.selectbox(
                f"Decision for {edge_id}", REVIEW_STATES, index=0, key=f"decision_{edge_id}"
            )
            notes[edge_id] = st.text_area(
                f"Notes for {edge_id}", key=f"notes_{edge_id}", height=90,
                placeholder="Record observations; choose uncertain if physical identity is not established.",
            )
            st.divider()
except (OSError, zipfile.BadZipFile) as exc:
    st.error(f"Could not read the original ZIP: {exc}")

st.subheader("Save both reviews")
filename = st.text_input("New versioned CSV filename", value=DEFAULT_REVIEW_NAME)
try:
    destination = safe_output_path(filename)
except ValueError as exc:
    destination = None
    st.error(str(exc))

if destination is not None and destination.exists():
    st.error(f"This file already exists and will not be overwritten: {destination}. Choose a new versioned filename.")
elif destination is not None:
    st.caption(f"Output path: `{destination}`")
    if st.button("Save reviews", type="primary"):
        rows = [
            {
                "edge_id": edge_id,
                "image_path_a": edges[edge_id]["image_path_a"],
                "image_path_b": edges[edge_id]["image_path_b"],
                "decision": decisions.get(edge_id, "pending"),
                "notes": notes.get(edge_id, ""),
            }
            for edge_id in ("W-CUT-01", "W-CUT-03")
        ]
        try:
            saved = save_reviews_exclusive(destination, rows)
            st.success(f"Saved both edge reviews to {saved}")
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
            st.error(str(exc))
