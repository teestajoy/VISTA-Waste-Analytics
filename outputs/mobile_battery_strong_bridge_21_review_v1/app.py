"""Focused, read-only visual review of strong bridge #21."""

from __future__ import annotations

import csv
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

EDGE_CSV = PROJECT_ROOT / "data/processed/mobile_battery_auto_group_edges_v2.csv"
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "VISTA_mobile_dataset_raw.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REVIEW_NAME = "mobile_battery_strong_bridge_21_review_v1.csv"
REVIEW_STATES = ("pending", "same_device", "different_device", "uncertain")
TARGET_PATHS = (
    "train/images/IMG_20250805_203922_jpg.rf.fbc486f138d29a49b1f3de783c184c12.jpg",
    "train/images/IMG_20250805_203923_jpg.rf.5c6fc3c85b160812cf0a3d80250b24d1.jpg",
)


def read_edge() -> dict[str, str]:
    with EDGE_CSV.open("r", encoding="utf-8-sig", newline="") as source:
        edges = list(csv.DictReader(source))
    bridges = [
        row for row in edges
        if row.get("strong_graph_bridge", "").strip().lower() in {"true", "1"}
    ]
    if len(bridges) < 21:
        raise ValueError("The v2 edge table does not contain bridge #21")
    edge = bridges[20]
    actual = (edge["image_path_a"], edge["image_path_b"])
    if actual != TARGET_PATHS:
        raise ValueError(f"Bridge #21 endpoints differ from the reviewed report: {actual}")
    return edge


def safe_review_path(filename: str) -> Path:
    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise ValueError("Enter a CSV filename only (no folder path)")
    return PROCESSED_DIR / filename


st.set_page_config(page_title="Strong bridge #21 review", layout="wide")
st.title("Mobile/Battery strong bridge #21")
st.caption("Read-only image comparison. The ZIP is read in memory; no extraction or archive changes occur.")
st.warning(
    "Bridge #21 is a graph boundary with strong visual similarity evidence. Similarity does not prove "
    "physical-device identity. No decision is preselected beyond pending."
)

try:
    edge = read_edge()
except (OSError, csv.Error, KeyError, ValueError) as exc:
    st.error(f"Could not verify bridge #21 in the v2 edge table: {exc}")
    st.stop()

archive_text = st.text_input("Original ZIP path", value=str(DEFAULT_ARCHIVE))
archive_path = Path(archive_text).expanduser()
if not archive_path.is_file():
    st.error(f"Original archive unavailable: {archive_path}. No extraction was attempted.")
    st.stop()

st.info("Removing this bridge would split its 2-image strong group (1 + 1); the prior report shows a 211 + 3 cut in the full 214-image candidate component.")
columns = st.columns(2)
try:
    with zipfile.ZipFile(archive_path, "r") as archive:
        for column, path, side in zip(columns, TARGET_PATHS, ("a", "b")):
            with column:
                st.markdown(f"**{path}**")
                st.write(
                    f"Original class: `{edge[f'original_class_{side}']}` · "
                    f"Original split: `{edge[f'original_split_{side}']}`"
                )
                try:
                    image_bytes = archive.read(path)
                    actual_sha = hashlib.sha256(image_bytes).hexdigest()
                    recorded_sha = edge[f"sha256_{side}"]
                    if actual_sha != recorded_sha:
                        st.error("ZIP image SHA-256 does not match the v2 edge record")
                    st.caption(f"SHA-256: `{actual_sha}`")
                    st.image(Image.open(io.BytesIO(image_bytes)), caption=path, use_container_width=True)
                except (KeyError, OSError, ValueError) as exc:
                    st.error(f"Image unavailable or unreadable in ZIP: {path} ({exc})")
except (OSError, zipfile.BadZipFile) as exc:
    st.error(f"Could not open the original ZIP: {exc}")

st.subheader("Saved v2 pair evidence")
st.write(f"SHA-256 comparison: **{'match' if edge['sha256_equal'].lower() in {'true', '1'} else 'different'}**")
st.write(f"pHash Hamming distance: **{edge['phash_distance']}**")
st.write(
    f"ORB/RANSAC: **{edge['orb_inliers']} inliers / {edge['orb_good_matches']} good matches "
    f"({float(edge['orb_inlier_fraction']):.3f} inlier fraction)**"
)
st.write(f"Filename timestamp match: **{'yes' if edge['same_timestamp'].lower() in {'true', '1'} else 'no'}**")
st.write(f"Match type: **{edge['match_level']}** · cross-class conflict: **{edge['class_conflict']}** · original splits cross: **{edge['split_crossing']}**")

decision = st.selectbox("Review decision", REVIEW_STATES, index=0, key="bridge21_decision")
notes = st.text_area("Review notes", key="bridge21_notes", height=110, placeholder="Describe what you observe; keep identity uncertain if it cannot be established.")

st.subheader("Save review")
filename = st.text_input("New versioned CSV filename", value=DEFAULT_REVIEW_NAME)
try:
    destination = safe_review_path(filename)
except ValueError as exc:
    destination = None
    st.error(str(exc))

if destination is not None and destination.exists():
    st.error(f"This file already exists and will not be overwritten: {destination}. Choose a new versioned filename.")
elif destination is not None:
    st.caption(f"Output: `{destination}`")
    if st.button("Save review", type="primary"):
        from review_store import save_review_exclusive

        try:
            saved = save_review_exclusive(
                destination,
                {
                    "bridge_id": "#21",
                    "image_path_a": edge["image_path_a"],
                    "image_path_b": edge["image_path_b"],
                    "decision": decision,
                    "notes": notes,
                },
            )
            st.success(f"Saved bridge #21 review to {saved}")
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
            st.error(str(exc))
