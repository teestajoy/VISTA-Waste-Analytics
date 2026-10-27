"""Small, explicit human-review workflow for the paper/cardboard pilot queue."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "Waste_classification_dataset"
    / "final_merged_dataset"
)
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "msw_review_sample.csv"
REVIEW_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "paper_cardboard_pilot_review.csv"
CATEGORIES = ("paper", "cardboard")
REVIEW_FIELDS = (
    "source_relative_path",
    "decision",
    "reviewed_label",
    "reason",
)
QUEUE_FIELDS = {"source_relative_path", "source_folder", "proposed_label"}
DECISIONS = ("include", "hold", "exclude")


@dataclass(frozen=True)
class ReviewQueueItem:
    source_relative_path: str
    source_folder: str
    proposed_label: str


def load_review_queue(sample_csv: Path) -> list[ReviewQueueItem]:
    """Read the sample CSV as the complete queue; do not discover images."""
    with sample_csv.open("r", encoding="utf-8-sig", newline="") as sample_file:
        reader = csv.DictReader(sample_file)
        missing = QUEUE_FIELDS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Sample CSV is missing required columns: " + ", ".join(sorted(missing))
            )
        items: list[ReviewQueueItem] = []
        seen_paths: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            relative_path = (row.get("source_relative_path") or "").strip()
            source_folder = (row.get("source_folder") or "").strip()
            proposed_label = (row.get("proposed_label") or "").strip()
            _validate_relative_path(relative_path)
            parts = PurePosixPath(relative_path).parts
            if source_folder not in CATEGORIES or proposed_label not in CATEGORIES:
                raise ValueError(f"Sample CSV line {line_number} has an unsupported category.")
            if not parts or parts[0] != source_folder:
                raise ValueError(
                    f"Sample CSV line {line_number} path does not match its source_folder: "
                    f"{relative_path}"
                )
            if relative_path in seen_paths:
                raise ValueError(f"Sample CSV contains a duplicate path: {relative_path}")
            seen_paths.add(relative_path)
            items.append(ReviewQueueItem(relative_path, source_folder, proposed_label))
    if not items:
        raise ValueError(f"The review queue is empty: {sample_csv}")
    return items


def _validate_relative_path(relative_path: str) -> PurePosixPath:
    """Reject absolute, traversal, drive-qualified, and blank input paths."""
    normalized = relative_path.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(relative_path)
    raw_parts = normalized.split("/")
    if (
        not relative_path.strip()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise ValueError(f"Unsafe source-relative image path: {relative_path!r}")
    return posix_path


def resolve_image_path(dataset_root: Path, item: ReviewQueueItem) -> Path:
    """Resolve an image path and reject traversal or symlinks outside the root."""
    relative = _validate_relative_path(item.source_relative_path)
    if relative.parts[0] != item.source_folder:
        raise ValueError(
            f"Image path does not match category {item.source_folder!r}: "
            f"{item.source_relative_path}"
        )
    root = dataset_root.resolve()
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Image path resolves outside the configured dataset folder: "
            f"{item.source_relative_path}"
        ) from error
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Image file was not found in the dataset: {item.source_relative_path}"
        )
    return candidate


def load_review_image(dataset_root: Path, item: ReviewQueueItem) -> Image.Image:
    """Safely load a queued image and return an independent RGB copy."""
    image_path = resolve_image_path(dataset_root, item)
    try:
        with Image.open(image_path) as image:
            image.load()
            return image.convert("RGB").copy()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(
            f"Could not read image {item.source_relative_path}: {error}"
        ) from error


def validate_decision(decision: str, reviewed_label: str = "", reason: str = "") -> None:
    """Validate explicit human input using the preparation CSV's rules."""
    decision = decision.strip().lower()
    reviewed_label = reviewed_label.strip().lower()
    if decision not in DECISIONS:
        raise ValueError("Choose Include, Hold, or Exclude before saving.")
    if decision == "include" and reviewed_label not in CATEGORIES:
        raise ValueError("For Include, explicitly choose paper or cardboard as the correct label.")
    if decision in {"hold", "exclude"} and not reason.strip():
        raise ValueError(f"Enter a reason before saving a {decision.title()} decision.")


def load_saved_decisions(review_csv: Path, queue: list[ReviewQueueItem]) -> dict[str, dict[str, str]]:
    """Load prior explicit decisions; blank template rows remain pending."""
    if not review_csv.exists():
        return {}
    allowed_paths = {item.source_relative_path for item in queue}
    decisions: dict[str, dict[str, str]] = {}
    with review_csv.open("r", encoding="utf-8-sig", newline="") as review_file:
        reader = csv.DictReader(review_file)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ValueError(
                "Review CSV must have exactly these columns: " + ",".join(REVIEW_FIELDS)
            )
        for line_number, row in enumerate(reader, start=2):
            path = (row.get("source_relative_path") or "").strip()
            decision = (row.get("decision") or "").strip().lower()
            label = (row.get("reviewed_label") or "").strip().lower()
            reason = (row.get("reason") or "").strip()
            if not any((path, decision, label, reason)):
                continue
            if not path:
                raise ValueError(f"Review CSV line {line_number} has a blank image path.")
            _validate_relative_path(path)
            if path not in allowed_paths:
                raise ValueError(f"Review CSV contains an image outside the sample queue: {path}")
            # The existing review CSV may contain one blank template row per
            # queue item. Such rows are pending, not decisions.
            if not decision and not label and not reason:
                continue
            if path in decisions:
                raise ValueError(f"Review CSV contains more than one decision for: {path}")
            validate_decision(decision, label, reason)
            if decision != "include" and label:
                raise ValueError("Only Include decisions may have a reviewed label.")
            decisions[path] = {"decision": decision, "reviewed_label": label, "reason": reason}
    return decisions


def progress_counts(
    queue: list[ReviewQueueItem], decisions: dict[str, dict[str, str]]
) -> dict[str, int]:
    counts = {"reviewed": 0, "included": 0, "held": 0, "excluded": 0}
    for item in queue:
        decision = decisions.get(item.source_relative_path, {}).get("decision")
        if decision in DECISIONS:
            counts["reviewed"] += 1
            counts[{"include": "included", "hold": "held", "exclude": "excluded"}[decision]] += 1
    return counts


def _render_progress(st: Any, progress_slot: Any, queue: list[ReviewQueueItem], decisions) -> None:
    """Draw the progress area so it can be refreshed after a successful save."""
    counts = progress_counts(queue, decisions)
    progress_slot.empty()
    with progress_slot.container():
        st.progress(
            counts["reviewed"] / len(queue),
            text=f"{counts['reviewed']} of {len(queue)} images reviewed",
        )
        metric_columns = st.columns(4)
        for column, label, key in zip(
            metric_columns,
            ("Reviewed", "Included", "Held", "Excluded"),
            ("reviewed", "included", "held", "excluded"),
        ):
            column.metric(label, counts[key])


def save_decision(
    review_csv: Path,
    queue: list[ReviewQueueItem],
    source_relative_path: str,
    decision: str,
    reviewed_label: str,
    reason: str,
    *,
    confirm_replace: bool = False,
) -> None:
    """Persist explicit decisions only, preserving other valid queue entries."""
    save_decisions(
        review_csv,
        queue,
        {
            source_relative_path: {
                "decision": decision,
                "reviewed_label": reviewed_label,
                "reason": reason,
            }
        },
        confirmed_replacements={source_relative_path} if confirm_replace else set(),
    )


def save_decisions(
    review_csv: Path,
    queue: list[ReviewQueueItem],
    selected_decisions: dict[str, dict[str, str]],
    *,
    confirmed_replacements: set[str] | None = None,
) -> None:
    """Save explicit decisions as one batch while preserving prior decisions."""
    confirmed_replacements = confirmed_replacements or set()
    queue_paths = {item.source_relative_path for item in queue}
    unknown_paths = set(selected_decisions) - queue_paths
    if unknown_paths:
        raise ValueError(
            "Selected image is not in the review queue: " + ", ".join(sorted(unknown_paths))
        )

    existing_decisions = load_saved_decisions(review_csv, queue)
    updated_decisions = dict(existing_decisions)
    for source_relative_path, raw_value in selected_decisions.items():
        decision = raw_value.get("decision", "").strip().lower()
        reviewed_label = raw_value.get("reviewed_label", "").strip().lower()
        reason = raw_value.get("reason", "").strip()
        validate_decision(decision, reviewed_label, reason)
        new_value = {
            "decision": decision,
            "reviewed_label": reviewed_label if decision == "include" else "",
            "reason": reason,
        }
        old_value = existing_decisions.get(source_relative_path)
        if (
            old_value is not None
            and old_value != new_value
            and source_relative_path not in confirmed_replacements
        ):
            raise ValueError(
                f"Confirm replacing the existing decision for {source_relative_path}."
            )
        updated_decisions[source_relative_path] = new_value

    review_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source_relative_path": item.source_relative_path,
            **updated_decisions[item.source_relative_path],
        }
        for item in queue
        if item.source_relative_path in updated_decisions
    ]
    with review_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def render_review_page(st: Any) -> None:
    """Render the pilot review interface without inference or training."""
    st.subheader("Paper / cardboard pilot review")
    st.caption("Review the existing sample queue. Nothing is saved until you choose Save decision.")
    try:
        queue = load_review_queue(SAMPLE_CSV_PATH)
        decisions = load_saved_decisions(REVIEW_CSV_PATH, queue)
    except (OSError, ValueError) as error:
        st.error(f"Could not load the pilot review files: {error}")
        return

    progress_slot = st.empty()
    _render_progress(st, progress_slot, queue, decisions)

    review_mode = st.radio(
        "Review mode",
        ("Single image", "Batch review"),
        horizontal=True,
        key="pilot_review_mode",
    )
    if review_mode == "Batch review":
        _render_batch_review(st, queue, decisions, progress_slot)
        return

    _render_single_review(st, queue, decisions, progress_slot)


def _render_single_review(st: Any, queue, decisions, progress_slot) -> None:
    """Keep the original one-image-at-a-time review path available."""

    selected_index = st.selectbox(
        "Review queue image",
        options=range(len(queue)),
        format_func=lambda index: (
            f"{index + 1} / {len(queue)} · {queue[index].proposed_label} · "
            f"{queue[index].source_relative_path}"
        ),
    )
    item = queue[selected_index]
    st.code(item.source_relative_path, language=None)
    st.write(f"**Proposed category:** {item.proposed_label}")
    try:
        image = load_review_image(DATASET_ROOT, item)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return
    st.image(image, caption=f"{item.proposed_label} · {item.source_relative_path}", width="stretch")

    existing = decisions.get(item.source_relative_path)
    widget_id = str(abs(hash(item.source_relative_path)))
    decision_key = f"pilot_review_decision_{widget_id}"
    decision_options = ("Choose a decision…", "Include", "Hold", "Exclude")
    if decision_key not in st.session_state:
        saved_decision = existing["decision"].title() if existing else decision_options[0]
        st.session_state[decision_key] = saved_decision
    chosen_decision = st.selectbox("Decision", decision_options, key=decision_key)
    normalized_decision = chosen_decision.lower()

    label_key = f"pilot_review_label_{widget_id}"
    reason_key = f"pilot_review_reason_{widget_id}"
    if label_key not in st.session_state:
        st.session_state[label_key] = existing["reviewed_label"] if existing else "Choose a label…"
    if reason_key not in st.session_state:
        st.session_state[reason_key] = existing["reason"] if existing else ""
    reviewed_label = ""
    reason = ""
    if normalized_decision == "include":
        reviewed_label = st.selectbox(
            "Confirm the correct label",
            ("Choose a label…", *CATEGORIES),
            key=label_key,
        )
        reviewed_label = "" if reviewed_label == "Choose a label…" else reviewed_label
    elif normalized_decision in {"hold", "exclude"}:
        reason = st.text_area(
            "Reason (required)", key=reason_key, max_chars=1000
        )

    new_value = {
        "decision": normalized_decision if normalized_decision in DECISIONS else "",
        "reviewed_label": reviewed_label,
        "reason": reason.strip(),
    }
    changed = existing is not None and new_value != existing
    confirmed = True
    if changed:
        confirmed = st.checkbox(
            "I confirm replacing this image's saved decision.",
            key=f"pilot_review_replace_{widget_id}",
        )
    can_save = bool(new_value["decision"])
    try:
        validate_decision(new_value["decision"], reviewed_label, reason)
    except ValueError as error:
        can_save = False
        if new_value["decision"]:
            st.info(str(error))
    if st.button(
        "Save decision",
        type="primary",
        disabled=not can_save or (changed and not confirmed),
    ):
        try:
            save_decision(
                REVIEW_CSV_PATH,
                queue,
                item.source_relative_path,
                new_value["decision"],
                reviewed_label,
                reason,
                confirm_replace=confirmed,
            )
        except (OSError, ValueError) as error:
            st.error(f"Decision was not saved: {error}")
        else:
            st.success("Decision saved.")
            # Refresh the progress panel in this same interaction. A full-page
            # rerun here cleared the success message and left the first image
            # selected, making a successful save look like a no-op.
            updated_decisions = load_saved_decisions(REVIEW_CSV_PATH, queue)
            _render_progress(st, progress_slot, queue, updated_decisions)


def _batch_widget_id(source_relative_path: str) -> str:
    return hashlib.sha256(source_relative_path.encode("utf-8")).hexdigest()[:12]


def _render_batch_review(st: Any, queue, existing_decisions, progress_slot) -> None:
    """Render a contact-sheet grid with independent explicit decisions."""
    st.caption(
        "Review multiple queue images at once. Proposed categories are suggestions "
        "only; Include always needs your own label confirmation."
    )
    selected_decisions: dict[str, dict[str, str]] = {}
    confirmed_replacements: set[str] = set()
    selected_paths: list[str] = []

    for row_start in range(0, len(queue), 4):
        columns = st.columns(4)
        for column, item in zip(columns, queue[row_start : row_start + 4]):
            with column:
                st.markdown(f"**{item.proposed_label.title()}**")
                try:
                    image = load_review_image(DATASET_ROOT, item)
                except (OSError, ValueError) as error:
                    st.error(str(error))
                    continue
                st.image(image, width="stretch")
                st.caption(item.source_relative_path)

                existing = existing_decisions.get(item.source_relative_path)
                widget_id = _batch_widget_id(item.source_relative_path)
                decision_key = f"pilot_batch_decision_{widget_id}"
                label_key = f"pilot_batch_label_{widget_id}"
                reason_key = f"pilot_batch_reason_{widget_id}"
                confirm_key = f"pilot_batch_confirm_{widget_id}"
                decision_options = ("Choose a decision…", "Include", "Hold", "Exclude")
                if decision_key not in st.session_state:
                    st.session_state[decision_key] = (
                        existing["decision"].title() if existing else decision_options[0]
                    )
                if label_key not in st.session_state:
                    st.session_state[label_key] = (
                        existing["reviewed_label"] if existing else "Choose a label…"
                    )
                if reason_key not in st.session_state:
                    st.session_state[reason_key] = existing["reason"] if existing else ""

                chosen = st.selectbox(
                    "Decision",
                    decision_options,
                    key=decision_key,
                    label_visibility="collapsed",
                )
                decision = chosen.lower()
                reviewed_label = ""
                reason = ""
                if decision == "include":
                    label_choice = st.selectbox(
                        "Confirm label",
                        ("Choose a label…", *CATEGORIES),
                        key=label_key,
                    )
                    reviewed_label = (
                        "" if label_choice == "Choose a label…" else label_choice
                    )
                elif decision in {"hold", "exclude"}:
                    reason = st.text_input(
                        "Reason (required)",
                        key=reason_key,
                        placeholder="Why is this held or excluded?",
                    )

                if decision not in DECISIONS:
                    continue
                try:
                    validate_decision(decision, reviewed_label, reason)
                except ValueError as error:
                    st.caption(str(error))
                    continue

                value = {
                    "decision": decision,
                    "reviewed_label": reviewed_label,
                    "reason": reason.strip(),
                }
                changed = existing is not None and value != existing
                confirmed = True
                if changed:
                    confirmed = st.checkbox(
                        "Confirm replacement",
                        key=confirm_key,
                        help="The existing saved decision will be replaced.",
                    )
                if existing is None or not changed or confirmed:
                    selected_decisions[item.source_relative_path] = value
                    selected_paths.append(item.source_relative_path)
                    if changed and confirmed:
                        confirmed_replacements.add(item.source_relative_path)

    if not selected_decisions:
        st.info("Select valid decisions above to enable bulk save.")
        return

    if st.button("Save selected decisions", type="primary"):
        try:
            save_decisions(
                REVIEW_CSV_PATH,
                queue,
                selected_decisions,
                confirmed_replacements=confirmed_replacements,
            )
        except (OSError, ValueError) as error:
            st.error(f"Batch decisions were not saved: {error}")
        else:
            st.success(f"Saved {len(selected_paths)} explicit decision(s).")
            updated_decisions = load_saved_decisions(REVIEW_CSV_PATH, queue)
            _render_progress(st, progress_slot, queue, updated_decisions)
