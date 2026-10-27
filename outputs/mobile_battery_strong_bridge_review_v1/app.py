"""Focused, read-only visual review of strong bridges #3 and #4."""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
EDGE_CSV = PROJECT_ROOT / "data/processed/mobile_battery_auto_group_edges_v2.csv"
MANIFEST_CSV = PROJECT_ROOT / "data/processed/mobile_battery_auto_group_manifest_v2.csv"
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "VISTA_mobile_dataset_raw.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REVIEW_NAME = "mobile_battery_strong_bridge_review_v1.csv"
REVIEW_STATES = ("pending", "same_device", "different_device", "uncertain")

TARGETS = {
    3: (
        "test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg",
        "train/images/IMG_20250803_220022_jpg.rf.dc0fdee22f504920709e6321bf117c22.jpg",
    ),
    4: (
        "test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg",
        "valid/images/IMG_20250803_220024_jpg.rf.2fc166ace133b8664434156a034ba786.jpg",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_bridge_rows() -> dict[int, dict[str, str]]:
    edges = [
        row for row in read_csv(EDGE_CSV)
        if row.get("strong_graph_bridge", "").strip().lower() in {"true", "1"}
    ]
    found: dict[int, dict[str, str]] = {}
    for bridge_number, paths in TARGETS.items():
        matching = [
            row for row in edges
            if (row["image_path_a"], row["image_path_b"]) == paths
            or (row["image_path_b"], row["image_path_a"]) == paths
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected one saved edge for bridge #{bridge_number}; found {len(matching)}")
        found[bridge_number] = matching[0]
    return found


def edge_side(edge: dict[str, str], path: str) -> str:
    if edge["image_path_a"] == path:
        return "a"
    if edge["image_path_b"] == path:
        return "b"
    raise ValueError(f"Path is not an endpoint of this edge: {path}")


def format_orb(edge: dict[str, str]) -> str:
    inliers = int(edge["orb_inliers"])
    matches = int(edge["orb_good_matches"])
    fraction = float(edge["orb_inlier_fraction"])
    return f"{inliers} RANSAC inliers / {matches} good matches ({fraction:.3f} inlier fraction)"


st.set_page_config(page_title="Strong bridges #3 and #4", layout="wide")
st.title("Mobile/Battery strong-bridge review")
st.caption("Focused review of bridges #3 and #4 only. Images are read in memory from the original ZIP; nothing is extracted or modified.")
st.warning(
    "Visual similarity and graph connectivity do not prove that images show the same physical device. "
    "No duplicate or device decision is preselected. Decisions and notes are saved only when you click Save reviews."
)

archive_text = st.text_input("Original ZIP path", value=str(DEFAULT_ARCHIVE))
archive_path = Path(archive_text).expanduser()
if not EDGE_CSV.is_file() or not MANIFEST_CSV.is_file():
    st.error(f"Required v2 data is missing: {EDGE_CSV} or {MANIFEST_CSV}")
    st.stop()
if not archive_path.is_file():
    st.error(f"Original archive is unavailable: {archive_path}. No extraction was attempted.")
    st.stop()

try:
    bridge_rows = load_bridge_rows()
    manifest = {row["image_path"]: row for row in read_csv(MANIFEST_CSV)}
    archive = zipfile.ZipFile(archive_path, "r")
except (OSError, csv.Error, KeyError, ValueError, zipfile.BadZipFile) as exc:
    st.error(f"Could not load review evidence: {exc}")
    st.stop()

try:
    with archive:
        for bridge_number in (3, 4):
            edge = bridge_rows[bridge_number]
            title = f"Bridge #{bridge_number} — {manifest[edge['image_path_a']]['strong_group_id']}"
            st.header(title)
            if bridge_number == 3:
                st.info(
                    "Removing this edge disconnects the 214-image all-evidence candidate component into "
                    "212 and 2 images. It also crosses original test/train splits."
                )
            else:
                st.info(
                    "Removing this edge disconnects the 214-image all-evidence candidate component into "
                    "213 and 1 images. It also crosses original test/validation splits."
                )

            columns = st.columns(2)
            endpoints = (edge["image_path_a"], edge["image_path_b"])
            for column, path in zip(columns, endpoints):
                side = edge_side(edge, path)
                with column:
                    st.markdown(f"**{path}**")
                    st.write(
                        f"Original class: `{edge[f'original_class_{side}']}`  ·  "
                        f"Original split: `{edge[f'original_split_{side}']}`"
                    )
                    try:
                        image_bytes = archive.read(path)
                        actual_sha = hashlib.sha256(image_bytes).hexdigest()
                        recorded_sha = edge[f"sha256_{side}"]
                        if actual_sha != recorded_sha:
                            st.error("Archive image SHA-256 does not match the saved v2 edge record.")
                        st.caption(f"SHA-256: `{actual_sha}`")
                        st.image(Image.open(io.BytesIO(image_bytes)), caption=path, use_container_width=True)
                    except (KeyError, OSError, ValueError) as exc:
                        st.error(f"Image unavailable or unreadable in the ZIP: {path} ({exc})")

            st.markdown("**Pair evidence from the v2 edge table**")
            hash_equal = edge["sha256_equal"].strip().lower() in {"true", "1"}
            st.write(f"SHA-256 comparison: **{'match' if hash_equal else 'different'}**")
            st.write(f"pHash Hamming distance: **{edge['phash_distance']}**")
            st.write(f"ORB/RANSAC: **{format_orb(edge)}**")
            st.write(f"Filename timestamp match: **{'yes' if edge['same_timestamp'].lower() in {'true', '1'} else 'no'}**")
            st.caption(
                "The edge meets the v2 strong visual-candidate rule. These measurements indicate image similarity, "
                "not physical-device identity."
            )
            st.selectbox(
                f"Decision for bridge #{bridge_number}",
                options=REVIEW_STATES,
                index=0,
                key=f"review_decision_{bridge_number}",
                help="same_device means the same physical object. It does not mean the image files are identical.",
            )
            st.text_area(
                f"Notes for bridge #{bridge_number} (optional)",
                key=f"review_note_{bridge_number}",
                height=90,
                placeholder="Record only what you observe; leave physical identity uncertain if not established.",
            )
            st.divider()
except (OSError, zipfile.BadZipFile) as exc:
    st.error(f"Could not read the original ZIP: {exc}")

st.subheader("Save reviews")
review_name = st.text_input("New versioned CSV filename", value=DEFAULT_REVIEW_NAME)
if Path(review_name).name != review_name or not review_name.lower().endswith(".csv"):
    st.error("Enter a CSV filename only (no folder path).")
    review_path = None
else:
    review_path = PROCESSED_DIR / review_name

if review_path is not None and review_path.exists():
    st.error(f"This file already exists and will not be overwritten: {review_path}. Choose a new versioned filename.")
elif review_path is not None:
    st.caption(f"Reviews will be saved to: `{review_path}`")
    if st.button("Save reviews", type="primary"):
        from review_store import save_reviews_exclusive

        reviews = [
            {
                "bridge_id": f"#{bridge_number}",
                "image_path_a": edge["image_path_a"],
                "image_path_b": edge["image_path_b"],
                "decision": st.session_state.get(f"review_decision_{bridge_number}", "pending"),
                "notes": st.session_state.get(f"review_note_{bridge_number}", ""),
            }
            for bridge_number, edge in bridge_rows.items()
        ]
        try:
            saved = save_reviews_exclusive(review_path, reviews)
            st.success(f"Saved both bridge reviews to {saved}")
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
            st.error(str(exc))
