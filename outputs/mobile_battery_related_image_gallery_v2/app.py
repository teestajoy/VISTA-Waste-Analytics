"""Local Streamlit gallery with versioned, explicit human-review controls."""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from review_store import REVIEW_STATES, apply_group_review, group_ids, serialize_rows, write_review_csv_exclusive

PROJECT_ROOT = next(
    (
        parent
        for parent in APP_DIR.parents
        if (parent / "data/processed/mobile_battery_related_image_review_v2.csv").is_file()
    ),
    APP_DIR.parents[1],
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WORKSHEET = PROCESSED_DIR / "mobile_battery_related_image_review_v2.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "mobile_battery_related_image_review_v3.csv"
REVIEW_GUIDE = PROJECT_ROOT / "docs" / "mobile_battery_related_image_review_v1.md"
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "VISTA_mobile_dataset_raw.zip"


def load_rows() -> list[dict[str, str]]:
    with WORKSHEET.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("The starting review worksheet is empty")
    required = {
        "candidate_group_id", "image_path", "original_class", "original_split",
        "exact_hash_relationship", "similarity_evidence", "human_decision", "notes",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"The starting worksheet is missing columns: {', '.join(sorted(missing))}")
    return rows


def candidate_kind(group_id: str) -> str:
    if group_id.startswith("TS-"):
        return "Filename timestamp candidate"
    if group_id.startswith("ND-"):
        return "Perceptual-hash / ORB-RANSAC visual candidate"
    return "Candidate group"


def safe_output_path(value: str) -> Path:
    """Require exports to be direct CSV children of data/processed."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROCESSED_DIR / candidate
    candidate = candidate.resolve()
    processed_resolved = PROCESSED_DIR.resolve()
    if candidate.parent != processed_resolved:
        raise ValueError(f"Choose an output CSV directly inside {processed_resolved}")
    if candidate.suffix.lower() != ".csv":
        raise ValueError("Review exports must use a .csv filename")
    if candidate == WORKSHEET.resolve():
        raise ValueError("The v2 starting worksheet is read-only; choose a new versioned filename")
    return candidate


st.set_page_config(page_title="Mobile and Battery Review Gallery", layout="wide")
st.title("Mobile and Battery candidate review gallery")
st.caption(
    "Images are read directly from the ZIP in memory. This app never extracts or edits images, "
    "labels, splits, or model artifacts. Review edits stay in this browser session until exported."
)
st.info(
    "`same_device` means the images show the same physical object; it does not mean the image "
    "files are identical. If evidence only suggests the same source image, augmentation, or a "
    "near-identical capture, choose `uncertain` and explain that suspected relationship in notes."
)

if not WORKSHEET.is_file():
    st.error(f"Starting worksheet is unavailable: {WORKSHEET}")
    st.stop()
if "review_rows" not in st.session_state:
    try:
        st.session_state.review_rows = load_rows()
    except (OSError, csv.Error, UnicodeError, ValueError) as exc:
        st.error(f"Could not read the v2 starting worksheet: {exc}")
        st.stop()

rows = st.session_state.review_rows
groups: dict[str, list[dict[str, str]]] = {}
for row in rows:
    for group_id in group_ids(row):
        groups.setdefault(group_id, []).append(row)
if not groups:
    st.warning("No candidate groups were found in the starting worksheet.")
    st.stop()

st.sidebar.header("Review session")
selected_group = st.sidebar.selectbox("Candidate group", sorted(groups))
st.sidebar.caption(f"{len(groups)} groups · {len(rows)} source image rows")
archive_text = st.sidebar.text_input("Read-only ZIP path", value=str(DEFAULT_ARCHIVE))
archive_path = Path(archive_text).expanduser()
try:
    output_path = safe_output_path(st.sidebar.text_input("New versioned CSV output", value=str(DEFAULT_OUTPUT)))
    output_path_error = None
except (OSError, ValueError) as exc:
    output_path = None
    output_path_error = str(exc)

members = groups[selected_group]
member_paths = {row["image_path"] for row in members}
overlapping = sorted(
    other_group
    for other_group, other_members in groups.items()
    if other_group != selected_group and any(row["image_path"] in member_paths for row in other_members)
)

st.subheader(f"{selected_group} · {candidate_kind(selected_group)}")
st.caption(f"{len(members)} worksheet image row(s) in this candidate group")
if overlapping:
    st.warning(
        "Some images also belong to other candidate groups: " + ", ".join(overlapping) + ". "
        "The worksheet stores one decision and notes field per image row. Applying this group "
        "review updates every member row, including shared rows; a later review of an overlapping "
        "group can update those same rows."
    )

decisions_here = {row.get("human_decision", "pending") or "pending" for row in members}
if len(decisions_here) == 1 and next(iter(decisions_here)) in REVIEW_STATES:
    initial_decision = next(iter(decisions_here))
else:
    initial_decision = "pending"
    if len(decisions_here) > 1:
        st.warning("Existing member decisions differ. Review the per-image states below before applying a group-wide choice.")

decision_key = f"decision::{selected_group}"
note_key = f"note::{selected_group}"
if decision_key not in st.session_state:
    st.session_state[decision_key] = initial_decision
if note_key not in st.session_state:
    st.session_state[note_key] = ""

st.selectbox("Decision for all image rows in this group", REVIEW_STATES, key=decision_key)
st.caption(
    "Use `same_device` only if all reviewed members show the same physical object. "
    "Use `different_device` when the candidate relationship does not represent one shared physical object. "
    "Use `uncertain` when you cannot establish physical identity."
)
st.text_area(
    "Additional review notes (appended to existing notes; leave blank to preserve them)",
    key=note_key,
    height=100,
    placeholder="For example: appears to be the same source image or an augmented copy; physical-device identity is unconfirmed.",
)
if st.button("Apply this review to the group", type="primary"):
    try:
        updated, matched = apply_group_review(
            rows,
            selected_group,
            st.session_state[decision_key],
            st.session_state[note_key],
        )
        st.session_state.review_rows = updated
        st.success(f"Applied {st.session_state[decision_key]} to {matched} image row(s) in {selected_group}.")
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

st.markdown("### Candidate images")
archive: zipfile.ZipFile | None = None
if archive_path.is_file():
    try:
        archive = zipfile.ZipFile(archive_path, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        st.error(f"Could not open image archive: {archive_path} ({exc})")
else:
    st.warning(f"Image archive unavailable at {archive_path}; images will be marked unavailable. No extraction was attempted.")

if archive is not None:
    with archive:
        for start in range(0, len(members), 2):
            columns = st.columns(2)
            for column, row in zip(columns, members[start : start + 2]):
                with column:
                    st.markdown(f"**{row['image_path']}**")
                    st.write(f"Original class: `{row['original_class']}` · split: `{row['original_split']}`")
                    st.write(f"Current review decision: `{row.get('human_decision') or 'pending'}`")
                    st.caption(f"Existing notes: {row.get('notes') or '(none)'}")
                    st.caption(f"Exact-hash relationship: {row['exact_hash_relationship']}")
                    st.caption(f"Similarity evidence: {row['similarity_evidence']}")
                    try:
                        st.image(archive.read(row["image_path"]), caption=row["image_path"], use_column_width=True)
                    except (KeyError, OSError, RuntimeError) as exc:
                        st.warning(f"Image unavailable in the ZIP: {row['image_path']} ({exc})")
else:
    for start in range(0, len(members), 2):
        columns = st.columns(2)
        for column, row in zip(columns, members[start : start + 2]):
            with column:
                st.markdown(f"**{row['image_path']}**")
                st.write(f"Original class: `{row['original_class']}` · split: `{row['original_split']}`")
                st.write(f"Current review decision: `{row.get('human_decision') or 'pending'}`")
                st.caption(f"Existing notes: {row.get('notes') or '(none)'}")
                st.caption(f"Exact-hash relationship: {row['exact_hash_relationship']}")
                st.caption(f"Similarity evidence: {row['similarity_evidence']}")
                st.warning(f"Image unavailable: archive not found ({archive_path})")

st.markdown("### Export reviewed worksheet")
st.write(f"Starting file: `{WORKSHEET}`")
if output_path_error:
    st.error(output_path_error)
elif output_path is not None and output_path.exists():
    st.error(f"Output already exists; it will not be overwritten: {output_path}. Choose a new versioned filename.")
else:
    st.caption(f"New output path: `{output_path}`")
    if st.button("Save reviewed CSV (create once, no overwrite)", disabled=output_path is None):
        try:
            saved = write_review_csv_exclusive(output_path, WORKSHEET, rows)
            st.success(f"Saved reviewed worksheet: {saved}")
        except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
            st.error(str(exc))

try:
    export_bytes = serialize_rows(rows).encode("utf-8-sig")
    export_filename = output_path.name if output_path is not None else "mobile_battery_related_image_review_export.csv"
    st.download_button(
        "Download current review CSV",
        data=export_bytes,
        file_name=export_filename,
        mime="text/csv",
        help="Downloads the current in-session review data; it does not overwrite any local file.",
    )
except ValueError as exc:
    st.error(f"Could not prepare CSV download: {exc}")

if REVIEW_GUIDE.is_file():
    with st.expander("Review guide", expanded=False):
        st.markdown(REVIEW_GUIDE.read_text(encoding="utf-8"))
else:
    st.warning(f"Review guide unavailable: {REVIEW_GUIDE}")
