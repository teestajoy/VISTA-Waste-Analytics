"""Read-only local gallery for the mobile/battery image review worksheet."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (
        parent
        for parent in APP_DIR.parents
        if (parent / "data/processed/mobile_battery_related_image_review_v1.csv").is_file()
    ),
    APP_DIR.parents[1],
)
WORKSHEET = PROJECT_ROOT / "data/processed/mobile_battery_related_image_review_v1.csv"
REVIEW_GUIDE = PROJECT_ROOT / "docs/mobile_battery_related_image_review_v1.md"
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "VISTA_mobile_dataset_raw.zip"


def load_rows() -> list[dict[str, str]]:
    with WORKSHEET.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def candidate_kind(group_id: str) -> str:
    if group_id.startswith("TS-"):
        return "Filename timestamp candidate"
    if group_id.startswith("ND-"):
        return "Perceptual-hash / ORB-RANSAC visual candidate"
    return "Candidate group"


st.set_page_config(page_title="Mobile and Battery Candidate Gallery", layout="wide")
st.title("Mobile and Battery candidate image gallery")
st.caption(
    "Read-only view of the review worksheet. The gallery does not change labels, splits, "
    "review states, or device identities. Images are read directly from the ZIP in memory; "
    "nothing is extracted."
)

if not WORKSHEET.is_file():
    st.error(f"Review worksheet is unavailable: {WORKSHEET}")
    st.stop()

try:
    rows = load_rows()
except (OSError, csv.Error, UnicodeError) as exc:
    st.error(f"Could not read the review worksheet: {exc}")
    st.stop()

if not rows:
    st.info("The review worksheet contains no candidate images.")
    st.stop()

groups: dict[str, list[dict[str, str]]] = {}
for row in rows:
    for group_id in (item.strip() for item in row.get("candidate_group_id", "").split(";")):
        if group_id:
            groups.setdefault(group_id, []).append(row)

st.sidebar.header("Gallery")
group_id = st.sidebar.selectbox("Candidate group", sorted(groups))
st.sidebar.caption(f"{len(groups)} groups · {len(rows)} unique worksheet images")
archive_text = st.sidebar.text_input("Read-only ZIP path", value=str(DEFAULT_ARCHIVE))
archive_path = Path(archive_text).expanduser()

group_rows = groups[group_id]
st.subheader(f"{group_id} · {candidate_kind(group_id)}")
st.info(
    "A timestamp or visual-similarity match is only a review candidate. It does not establish "
    "that images show the same physical device. All worksheet decisions remain pending."
)

archive: zipfile.ZipFile | None = None
if archive_path.is_file():
    try:
        archive = zipfile.ZipFile(archive_path, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        st.error(f"The image archive could not be opened: {archive_path} ({exc})")
else:
    st.warning(
        f"Image archive is unavailable at {archive_path}. Images will be marked unavailable. "
        "No extraction was attempted."
    )

if archive is not None:
    with archive:
        for start in range(0, len(group_rows), 2):
            columns = st.columns(2)
            for column, row in zip(columns, group_rows[start : start + 2]):
                with column:
                    image_path = row["image_path"]
                    st.markdown(f"**{image_path}**")
                    st.write(f"Original class: `{row['original_class']}`")
                    st.write(f"Original split: `{row['original_split']}`")
                    st.write(f"Human decision: `{row.get('human_decision') or 'pending'}`")
                    st.caption(f"Exact-hash relationship: {row['exact_hash_relationship']}")
                    st.caption(f"Similarity evidence: {row['similarity_evidence']}")
                    if group_id not in row.get("candidate_group_id", "").split(";"):
                        st.warning("This row does not list the selected group ID.")
                    try:
                        image_bytes = archive.read(image_path)
                        st.image(image_bytes, caption=image_path, use_column_width=True)
                    except (KeyError, OSError, RuntimeError) as exc:
                        st.warning(f"Image unavailable in the ZIP: {image_path} ({exc})")
else:
    for start in range(0, len(group_rows), 2):
        columns = st.columns(2)
        for column, row in zip(columns, group_rows[start : start + 2]):
            with column:
                st.markdown(f"**{row['image_path']}**")
                st.write(f"Original class: `{row['original_class']}`")
                st.write(f"Original split: `{row['original_split']}`")
                st.write(f"Human decision: `{row.get('human_decision') or 'pending'}`")
                st.caption(f"Exact-hash relationship: {row['exact_hash_relationship']}")
                st.caption(f"Similarity evidence: {row['similarity_evidence']}")
                st.warning(f"Image unavailable: archive not found ({archive_path})")

if REVIEW_GUIDE.is_file():
    with st.expander("Review guidance", expanded=False):
        st.markdown(REVIEW_GUIDE.read_text(encoding="utf-8"))
else:
    st.warning(f"Review guidance is unavailable: {REVIEW_GUIDE}")

st.caption("This gallery is read-only. To record review decisions, edit a separate review copy outside this app.")
