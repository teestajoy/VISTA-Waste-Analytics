"""Fast, explicit human review page for the provisional test-image batch."""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get(
    "VISTA_TEST_REVIEW_DATASET",
    r"C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset",
))
BATCH_CSV = Path(os.environ.get(
    "VISTA_TEST_REVIEW_QUEUE", str(PROJECT_ROOT / "data" / "processed" / "reviewed_test_review_batch.csv")
))
DECISIONS_CSV = Path(os.environ.get(
    "VISTA_TEST_REVIEW_DECISIONS", str(PROJECT_ROOT / "data" / "processed" / "reviewed_test_review_decisions.csv")
))
DECISION_FIELDS = (
    "source_relative_path", "sha256", "review_type", "decision",
    "reviewed_label", "reason", "reviewed_utc",
)
REQUIRED_BATCH_FIELDS = {
    "source_relative_path", "sha256", "source_folder", "split",
}
FAMILY_LABELS = ("MSW", "E_WASTE")
CATEGORY_LABELS = ("paper", "cardboard", "plastic", "glass", "organic", "battery")
REVIEW_TYPES = ("family", "category")
HOLD_REASON = "Unable to identify confidently from the image."
BLIND_REVIEW = os.environ.get("VISTA_TEST_REVIEW_BLIND", "").strip().lower() in {
    "1", "true", "yes"
}
PROTECTED_PATHS = {
    (PROJECT_ROOT / "data" / "processed" / name).resolve()
    for name in (
        "family_review_decisions.csv",
        "family_msw_review_batch_decisions.csv",
        "family_review_decisions_msw_batch.csv",
        "paper_cardboard_pilot_review.csv",
        "paper_cardboard_pilot_split_manifest.csv",
        "family_provisional_split_manifest.csv",
        "family_review_queue.csv",
        "family_review_audit.csv",
        "reviewed_test_review_batch.csv",
        "reviewed_test_model_evaluation.json",
    )
}
_SHORTCUT_COMPONENT = None


def _validate_relative_path(value: str) -> str:
    normalized = (value or "").strip().replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value or "")
    if (
        not normalized or posix.is_absolute() or windows.is_absolute() or windows.drive
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ValueError(f"Unsafe source-relative image path: {value!r}")
    return posix.as_posix()


def load_batch(batch_path: Path) -> list[dict[str, str]]:
    with Path(batch_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_BATCH_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Review batch is missing columns: " + ", ".join(sorted(missing)))
        rows = []
        seen = set()
        for line, raw in enumerate(reader, start=2):
            row = dict(raw)
            path = _validate_relative_path(row.get("source_relative_path", ""))
            if path in seen:
                raise ValueError(f"Review batch repeats image path on line {line}: {path}")
            seen.add(path)
            if row.get("split") != "test":
                raise ValueError(f"Review batch contains a non-test row: {path}")
            if row.get("source_folder") != path.split("/", 1)[0]:
                raise ValueError(f"Source folder does not match path: {path}")
            digest = (row.get("sha256") or "").strip().lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Invalid SHA-256 in review batch for {path}")
            if any((row.get(field) or "").strip() for field in (
                "review_status", "reviewed_family", "decision", "reviewed_label", "reason"
            )):
                raise ValueError(f"Batch CSV must remain an undecided queue; found a decision for {path}")
            row["source_relative_path"] = path
            row["sha256"] = digest
            rows.append(row)
    if not rows:
        raise ValueError("Review batch contains no images.")
    return rows


def resolve_source_image(dataset_root: Path, relative_path: str) -> Path:
    relative = _validate_relative_path(relative_path)
    root = Path(dataset_root).resolve(strict=True)
    resolved = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Image path resolves outside the dataset root: {relative}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"Image is not a file: {relative}")
    return resolved


def load_review_image(dataset_root: Path, row: dict[str, str]) -> Image.Image:
    path = resolve_source_image(dataset_root, row["source_relative_path"])
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != row["sha256"]:
        raise ValueError(f"Image changed since the batch was created: {row['source_relative_path']}")
    try:
        with Image.open(path) as source:
            source.load()
            return source.convert("RGB").copy()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"Could not read {row['source_relative_path']}: {error}") from error


def validate_decision(review_type: str, choice: str, reason: str = "") -> tuple[str, str]:
    if review_type not in REVIEW_TYPES:
        raise ValueError("Choose Family or Category review.")
    if choice == "HOLD":
        if not reason.strip():
            raise ValueError("Hold requires a brief reason.")
        return "hold", ""
    labels = FAMILY_LABELS if review_type == "family" else CATEGORY_LABELS
    if choice not in labels:
        raise ValueError(f"Choose one explicit {review_type} label or Hold.")
    return "include", choice


def load_decisions(path: Path, rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return {}
    queue = {row["source_relative_path"]: row for row in rows}
    decisions = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != DECISION_FIELDS:
            raise ValueError("Decision CSV has an unexpected schema; it was not changed.")
        for line, row in enumerate(reader, start=2):
            if not any((row.get(field) or "").strip() for field in DECISION_FIELDS):
                continue
            relative = _validate_relative_path(row.get("source_relative_path", ""))
            review_type = (row.get("review_type") or "").strip()
            key = (relative, review_type)
            if relative not in queue or queue[relative]["sha256"] != row.get("sha256"):
                raise ValueError(f"Saved decision on line {line} does not match this review batch.")
            if review_type not in REVIEW_TYPES or key in decisions:
                raise ValueError(f"Invalid or repeated saved decision on line {line}.")
            decision, label = validate_decision(
                review_type,
                "HOLD" if row.get("decision") == "hold" else (row.get("reviewed_label") or ""),
                row.get("reason") or "",
            )
            if decision != row.get("decision") or label != (row.get("reviewed_label") or ""):
                raise ValueError(f"Saved decision on line {line} is invalid.")
            if decision == "hold" and not (row.get("reason") or "").strip():
                raise ValueError(f"Saved Hold on line {line} has no reason.")
            decisions[key] = dict(row)
    return decisions


def _validate_output_path(output_path: Path, batch_path: Path, dataset_root: Path) -> Path:
    output = Path(output_path)
    if output.is_symlink():
        raise ValueError(f"Refusing to write through a symlink: {output}")
    resolved = output.resolve()
    protected = set(PROTECTED_PATHS) | {Path(batch_path).resolve()}
    if resolved in protected:
        raise ValueError(f"Review decisions require a separate output file: {resolved}")
    try:
        resolved.relative_to(Path(dataset_root).resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Review decisions must be saved outside the source dataset.")
    return output


def save_decision(
    output_path: Path,
    batch_path: Path,
    dataset_root: Path,
    rows: list[dict[str, str]],
    decisions: dict[tuple[str, str], dict[str, str]],
    source_relative_path: str,
    review_type: str,
    choice: str,
    *,
    confirm_replace: bool = False,
) -> dict[tuple[str, str], dict[str, str]]:
    relative = _validate_relative_path(source_relative_path)
    queue = {row["source_relative_path"]: row for row in rows}
    if relative not in queue:
        raise ValueError(f"Image is not in this review batch: {relative}")
    status, label = validate_decision(review_type, choice, HOLD_REASON if choice == "HOLD" else "")
    if status == "include":
        image = load_review_image(dataset_root, queue[relative])
        image.close()
    key = (relative, review_type)
    existing = decisions.get(key)
    if existing:
        same = existing["decision"] == status and existing["reviewed_label"] == label
        if not same and not confirm_replace:
            raise ValueError("A decision already exists; confirm replacement before changing it.")
        if same:
            return decisions
    record = {
        "source_relative_path": relative,
        "sha256": queue[relative]["sha256"],
        "review_type": review_type,
        "decision": status,
        "reviewed_label": label,
        "reason": HOLD_REASON if status == "hold" else "",
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
    }
    updated = dict(decisions)
    updated[key] = record
    output = _validate_output_path(output_path, batch_path, dataset_root)
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Decision output folder does not exist: {output.parent}")
    ordered = []
    queue_order = {row["source_relative_path"]: index for index, row in enumerate(rows)}
    for _, saved in sorted(
        updated.items(), key=lambda item: (queue_order[item[0][0]], REVIEW_TYPES.index(item[0][1]))
    ):
        ordered.append(saved)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    return updated


def progress_counts(rows: list[dict[str, str]], decisions, review_type: str) -> dict[str, int]:
    selected = [value for (path, kind), value in decisions.items() if kind == review_type]
    return {
        "reviewed": len(selected),
        "included": sum(row["decision"] == "include" for row in selected),
        "held": sum(row["decision"] == "hold" for row in selected),
        "remaining": max(0, len(rows) - len(selected)),
    }


def default_review_type(batch_path: Path) -> str:
    """Use Category mode by default for a category-named batch queue."""
    return "category" if "category_review_batch" in Path(batch_path).stem.lower() else "family"


def next_pending_index(rows, decisions, review_type: str, current: int = -1) -> int | None:
    for index, row in enumerate(rows):
        if index > current and (row["source_relative_path"], review_type) not in decisions:
            return index
    for index, row in enumerate(rows):
        if (row["source_relative_path"], review_type) not in decisions:
            return index
    return None


def shortcut_action(key: str, review_type: str) -> str | None:
    key = key.strip().lower()
    if key in {"arrowleft", "left"}:
        return "previous"
    if key in {"arrowright", "right"}:
        return "next"
    if key == "h":
        return "hold"
    if key == "s":
        return "skip"
    if review_type == "family":
        return {"1": "MSW", "2": "E_WASTE"}.get(key)
    return dict(zip("123456", CATEGORY_LABELS)).get(key)


def review_display_identity(row: dict[str, str], index: int, *, blind: bool | None = None) -> tuple[str, str, str]:
    """Return reviewer-facing identity without leaking source labels in blind mode."""
    if blind is None:
        blind = BLIND_REVIEW
    if blind:
        return (
            f"Review image {index + 1}",
            f"Blind review ID: {(row.get('sha256') or '')[:16]}",
            "Image shown without source filename or folder label",
        )
    path = row["source_relative_path"]
    return (
        Path(path).name,
        f"Source folder: {row['source_folder']} (provenance only) · Image ID: {row['sha256'][:16]}",
        Path(path).name,
    )


def review_confirmation_target(confirmation: dict[str, str], *, blind: bool | None = None) -> str:
    """Avoid revealing a label-bearing source path in blind-review confirmations."""
    if blind is None:
        blind = BLIND_REVIEW
    return (
        f"review image ID {confirmation['image_id']}"
        if blind else confirmation["path"]
    )


def review_error_message(error: Exception, *, blind: bool | None = None) -> str:
    """Hide path-bearing errors from the reviewer when blind mode is enabled."""
    if blind is None:
        blind = BLIND_REVIEW
    if blind:
        return "Could not load or save this image decision. Check the review queue and source file outside the blind review screen."
    return str(error)


def _shortcut_component(st):
    global _SHORTCUT_COMPONENT
    if _SHORTCUT_COMPONENT is None:
        _SHORTCUT_COMPONENT = st.components.v1.declare_component(
            "vista_test_batch_keyboard_shortcuts",
            path=str(Path(__file__).resolve().parent / "review_shortcuts"),
        )
    return _SHORTCUT_COMPONENT(
        task_label="family: 1 MSW · 2 E-waste | category: 1-6 category choices | H Hold · S Skip | ←/→ move",
        key="vista_test_review_shortcut_listener",
        default=None,
    )


def _review_one(st, rows, decisions, index, review_type):
    row = rows[index]
    path = row["source_relative_path"]
    title, identity_caption, image_caption = review_display_identity(row, index)
    existing = decisions.get((path, review_type))
    if existing:
        st.info(f"Saved {review_type} decision: {existing['decision']} {existing['reviewed_label']}")
    else:
        st.caption("No choice is selected. Only clicking a label button saves a human decision.")
    st.markdown(f"### {title}")
    if BLIND_REVIEW:
        st.caption(f"Image {index + 1} of {len(rows)} · {identity_caption}")
        st.warning("Source labels and model predictions are hidden. Judge only what is visible; use Hold or Skip when unsure.")
    else:
        st.code(path, language=None)
        st.caption(f"Image {index + 1} of {len(rows)} · {identity_caption}")
        st.warning("The source folder is not an expected label. Judge only what is visible; use Hold or Skip when unsure.")
    preview, detail = st.columns([1.2, 0.8], gap="large")
    image_readable = False
    with preview:
        try:
            image = load_review_image(DATASET_ROOT, row)
            st.image(image, caption=image_caption, width="stretch")
            image.close()
            image_readable = True
        except (OSError, ValueError) as error:
            st.error("Image unavailable. The source file may be missing, unreadable, or changed since the batch was created." if BLIND_REVIEW else f"Image unavailable: {error}")
            st.info("This image cannot be approved while it is unreadable. You may Hold it or Skip without saving.")
    with detail:
        saved = bool(existing)
        replace = False
        if saved:
            replace = st.checkbox("Confirm replacing this saved decision", key=f"review_replace_{review_type}_{index}")
        if review_type == "family":
            choices = (("MSW", "Municipal Solid Waste"), ("E_WASTE", "E-waste"))
            columns = st.columns(2)
        else:
            choices = tuple((label, label.title()) for label in CATEGORY_LABELS)
            columns = st.columns(2)
        for i, (value, label) in enumerate(choices):
            if columns[i % len(columns)].button(
                label, key=f"review_choice_{review_type}_{index}_{value}",
                type="primary", width="stretch", disabled=(saved and not replace) or not image_readable,
            ):
                try:
                    st.session_state.vista_review_decisions = save_decision(
                        DECISIONS_CSV, BATCH_CSV, DATASET_ROOT, rows, decisions,
                        path, review_type, value, confirm_replace=replace,
                    )
                    _remember_save_confirmation(
                        st, path, review_type,
                        st.session_state.vista_review_decisions[(path, review_type)],
                    )
                    _advance(st, rows, st.session_state.vista_review_decisions, review_type, index)
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(review_error_message(error))
        hold_col, skip_col = st.columns(2)
        if hold_col.button("Hold · unsure", key=f"review_hold_{review_type}_{index}", width="stretch"):
            try:
                st.session_state.vista_review_decisions = save_decision(
                    DECISIONS_CSV, BATCH_CSV, DATASET_ROOT, rows, decisions,
                    path, review_type, "HOLD", confirm_replace=replace,
                )
                _remember_save_confirmation(
                    st, path, review_type,
                    st.session_state.vista_review_decisions[(path, review_type)],
                )
                _advance(st, rows, st.session_state.vista_review_decisions, review_type, index)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(review_error_message(error))
        if skip_col.button("Skip · do not save", key=f"review_skip_{review_type}_{index}", width="stretch"):
            st.session_state[f"vista_review_index_{review_type}"] = (index + 1) % len(rows)
            st.rerun()


def _advance(st, rows, decisions, review_type, current):
    pending = next_pending_index(rows, decisions, review_type, current)
    st.session_state[f"vista_review_index_{review_type}"] = current if pending is None else pending


def _remember_save_confirmation(st, path, review_type, record):
    st.session_state.vista_last_saved_confirmation = {
        "path": path,
        "image_id": (record.get("sha256") or "")[:16],
        "review_type": review_type,
        "decision": record["decision"],
        "label": record["reviewed_label"] or "Hold",
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="VISTA · Test Batch Review", page_icon="♻️", layout="wide")
    st.title("VISTA — Held-out image review")
    st.caption("Explicit human labels only. No model predictions or source-folder labels are used as decisions.")
    try:
        rows = load_batch(BATCH_CSV)
        decisions = load_decisions(DECISIONS_CSV, rows)
    except (OSError, ValueError) as error:
        message = (
            "Could not open the blind review batch. Validate the queue and image files outside the reviewer view."
            if BLIND_REVIEW else f"Could not open review batch: {error}"
        )
        st.error(message)
        return
    st.session_state.vista_review_decisions = decisions
    review_type = st.radio(
        "Choose what to label for each image",
        ("family", "category"),
        index=0 if default_review_type(BATCH_CSV) == "family" else 1,
        horizontal=True,
        format_func=lambda value: value.title(),
        key=f"vista_test_review_type_{BATCH_CSV.stem}",
    )
    mode_label = "CATEGORY" if review_type == "category" else "FAMILY"
    st.markdown(f"## {mode_label} REVIEW MODE")
    if review_type == "category":
        st.info("Choose a visible material category: paper, cardboard, plastic, glass, organic, or battery. This is a human label, not a model prediction.")
    else:
        st.info("Choose the visible waste family: MSW or E-waste. This is a human label, not a model prediction.")
    confirmation = st.session_state.pop("vista_last_saved_confirmation", None)
    if confirmation:
        target = review_confirmation_target(confirmation)
        st.success(
            f"Saved {confirmation['review_type'].title()} decision: {confirmation['label']} "
            f"({confirmation['decision']}) for {target}"
        )
    task_key = f"vista_review_index_{review_type}"
    if task_key not in st.session_state:
        st.session_state[task_key] = next_pending_index(rows, decisions, review_type, -1) or 0
    index = max(0, min(int(st.session_state[task_key]), len(rows) - 1))
    st.session_state[task_key] = index
    counts = progress_counts(rows, decisions, review_type)
    st.progress(counts["reviewed"] / len(rows), text=f"{counts['reviewed']} reviewed · {counts['remaining']} remaining for {review_type}")
    progress_cols = st.columns(3)
    progress_cols[0].metric("Reviewed", counts["reviewed"])
    progress_cols[1].metric("Included", counts["included"])
    progress_cols[2].metric("Held", counts["held"])
    nav = st.columns(4)
    if nav[0].button("← Previous", disabled=index == 0, width="stretch"):
        st.session_state[task_key] = index - 1
        st.rerun()
    if nav[1].button("Next →", disabled=index >= len(rows) - 1, width="stretch"):
        st.session_state[task_key] = index + 1
        st.rerun()
    pending = next_pending_index(rows, decisions, review_type, index)
    if nav[2].button("Resume unfinished", disabled=pending is None, width="stretch"):
        if pending is not None:
            st.session_state[task_key] = pending
            st.rerun()
    nav[3].caption("Use arrow keys to navigate, H to Hold, S to Skip.")
    event = _shortcut_component(st)
    if isinstance(event, dict):
        event_id = event.get("id")
        if event_id and event_id != st.session_state.get("vista_last_shortcut_event"):
            st.session_state.vista_last_shortcut_event = event_id
            action = shortcut_action(str(event.get("key", "")), review_type)
            if action == "previous":
                st.session_state[task_key] = max(0, index - 1)
                st.rerun()
            elif action == "next":
                st.session_state[task_key] = min(len(rows) - 1, index + 1)
                st.rerun()
            elif action == "skip":
                st.session_state[task_key] = (index + 1) % len(rows)
                st.rerun()
            elif action == "hold" or action in (FAMILY_LABELS if review_type == "family" else CATEGORY_LABELS):
                st.session_state.vista_shortcut_action = action
                st.rerun()
    action = st.session_state.pop("vista_shortcut_action", None)
    if action:
        choice = "HOLD" if action == "hold" else action
        try:
            st.session_state.vista_review_decisions = save_decision(
                DECISIONS_CSV, BATCH_CSV, DATASET_ROOT, rows, decisions,
                rows[index]["source_relative_path"], review_type, choice,
            )
            current_path = rows[index]["source_relative_path"]
            _remember_save_confirmation(
                st, current_path, review_type,
                st.session_state.vista_review_decisions[(current_path, review_type)],
            )
            _advance(st, rows, st.session_state.vista_review_decisions, review_type, index)
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))
    st.divider()
    _review_one(st, rows, decisions, index, review_type)
    st.caption(f"Queue: `{BATCH_CSV}` · Separate decisions: `{DECISIONS_CSV}`")


if __name__ == "__main__":
    main()
