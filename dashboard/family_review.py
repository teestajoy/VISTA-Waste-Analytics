"""Local one-image-at-a-time reviewer for the separate family review queue."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import family_dataset_workflow as family_workflow  # noqa: E402


def _configured_path(environment_name: str, default: Path) -> Path:
    configured = os.environ.get(environment_name, "").strip()
    if not configured:
        return default
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


QUEUE_PATH = _configured_path("VISTA_FAMILY_REVIEW_QUEUE", family_workflow.QUEUE_PATH)
DECISIONS_PATH = _configured_path(
    "VISTA_FAMILY_REVIEW_DECISIONS", family_workflow.DECISIONS_PATH
)
AUDIT_PATH = PROJECT_ROOT / "data" / "processed" / "family_review_audit.csv"
DATASET_DIR = family_workflow.DATASET_DIR
AUDIT_FIELDS = (
    "changed_utc", "source_relative_path", "previous_review_status",
    "previous_reviewed_family", "previous_reason", "new_review_status",
    "new_reviewed_family", "new_reason",
)
UI_CHOICES = ("PENDING", "MSW", "E_WASTE", "HOLD", "EXCLUDE")
BATCH_CHOICES = ("MSW", "E_WASTE", "HOLD", "EXCLUDE")
SUGGESTION_FILTERS = (
    "MSW candidate",
    "MSW candidate — confirm food waste",
    "E_WASTE candidate — verify content",
    "No suggestion — hold for inspection",
    "Unsupported",
    "Conflicting duplicate — manual review",
)


def load_review_queue(queue_path: Path, dataset_root: Path) -> list[dict[str, str]]:
    """Load and validate queue paths without scanning or changing the dataset."""
    queue_path = Path(queue_path)
    if not queue_path.is_file():
        raise FileNotFoundError(f"Family review queue not found: {queue_path}")

    with queue_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing_fields = set(family_workflow.QUEUE_FIELDS) - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                "Family review queue is missing columns: "
                + ", ".join(sorted(missing_fields))
            )
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"Family review queue is empty: {queue_path}")

    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        relative = family_workflow.normalize_relative_path(
            row.get("source_relative_path") or ""
        )
        if relative in seen:
            raise ValueError(f"Family review queue repeats path on line {line}: {relative}")
        seen.add(relative)
        folder = (row.get("source_folder") or "").strip()
        if (
            folder not in family_workflow.SOURCE_FOLDERS
            or Path(relative).parts[0] != folder
        ):
            raise ValueError(f"Family review queue has an inconsistent source folder for {relative}")
        if row.get("proposed_review_route") != family_workflow.REVIEW_ROUTES[folder]:
            raise ValueError(f"Family review queue has an invalid review route for {relative}")
        digest = (row.get("sha256") or "").strip().lower()
        if digest and (
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Family review queue has an invalid SHA-256 for {relative}")
        expected_group = f"sha256:{digest}" if digest else ""
        if (row.get("duplicate_group_id") or "") != expected_group:
            raise ValueError(f"Family review queue has an invalid duplicate group for {relative}")
        # Resolve every path now so traversal and symlink escapes fail before UI access.
        family_workflow.safe_source_path(Path(dataset_root), relative)
        row["source_relative_path"] = relative
        row["sha256"] = digest
        row["source_folder"] = folder

    return rows


def load_review_image(
    dataset_root: Path,
    source_relative_path: str,
    *,
    expected_sha256: str = "",
) -> Image.Image:
    """Return a fully loaded RGB copy, rejecting paths outside the dataset root."""
    image_path = family_workflow.safe_source_path(
        Path(dataset_root), source_relative_path
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"Review image is missing: {source_relative_path}")
    if expected_sha256 and family_workflow.sha256_file(image_path) != expected_sha256:
        raise ValueError(f"Image changed since the review queue was created: {source_relative_path}")
    try:
        with Image.open(image_path) as source:
            source.load()
            return source.convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise ValueError(
            f"Could not read review image {source_relative_path}: {error}"
        ) from error


def _inventory_rows(queue_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Adapt queue records to the preparation script's decision validator."""
    return [
        {
            **row,
            "reviewed_family": "",
            "review_status": "pending",
            "inventory_error": row.get("inventory_error", "") or "",
        }
        for row in queue_rows
    ]


def load_saved_decisions(
    decisions_path: Path, queue_rows: list[dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Read existing decisions using the preparation script's exact rules."""
    decisions_path = Path(decisions_path)
    if not decisions_path.exists():
        return {}
    if not decisions_path.is_file():
        raise ValueError(f"Decision output path is not a file: {decisions_path}")
    return family_workflow._load_decisions(
        decisions_path, _inventory_rows(queue_rows), queue_rows
    )


def decision_record(
    queue_row: dict[str, str],
    choice: str,
    reason: str = "",
    *,
    confirm_content: bool = False,
) -> dict[str, str]:
    """Convert an explicit UI choice into the CSV schema expected by prepare."""
    if choice not in UI_CHOICES:
        raise ValueError(f"Choose one of: {', '.join(UI_CHOICES)}")

    relative = queue_row["source_relative_path"]
    folder = queue_row["source_folder"]
    reason = "" if choice == "PENDING" else reason.strip()
    status = "include" if choice in family_workflow.FAMILIES else choice.lower()
    family = choice if choice in family_workflow.FAMILIES else ""

    if status in {"hold", "exclude"} and not reason:
        raise ValueError(f"{choice.title()} decision for {relative} requires a reason.")
    if status == "pending" and reason:
        raise ValueError(f"Pending image {relative} cannot have a decision reason.")
    if status == "include":
        if not queue_row.get("sha256") or queue_row.get("inventory_error"):
            raise ValueError(f"Unreadable or unhashed image {relative} cannot be included.")
        if queue_row.get("proposed_review_route") == "unsupported":
            raise ValueError(f"{relative} is unsupported for family inclusion until policy changes.")
        if folder == "organic":
            if family != "MSW":
                raise ValueError(f"Organic image {relative} can only be included as MSW.")
            if not confirm_content:
                raise ValueError(f"Confirm visible food waste before including {relative}.")
            if not reason:
                raise ValueError(f"Organic image {relative} needs a reason confirming visible food waste.")
        if folder == "battery":
            if family != "E_WASTE":
                raise ValueError(f"Battery image {relative} can only be included as E_WASTE.")
            if not confirm_content:
                raise ValueError(f"Confirm visible battery/electronic content before including {relative}.")
            if not reason:
                raise ValueError(f"Battery image {relative} needs a reason confirming visible content.")

    return {
        "source_relative_path": relative,
        "review_status": status,
        "reviewed_family": family,
        "reason": reason,
    }


def validate_decision_output_path(
    output_path: Path,
    *,
    expected_path: Path | None = None,
    dataset_root: Path | None = None,
) -> Path:
    """Restrict writes to the dedicated family-review decisions CSV."""
    output_path = Path(output_path)
    expected = Path(expected_path or DECISIONS_PATH).resolve()
    if output_path.is_symlink():
        raise ValueError(f"Refusing to write through a symlink: {output_path}")
    resolved = output_path.resolve()
    if resolved != expected:
        raise ValueError(f"Review decisions may only be written to: {expected}")
    if dataset_root is not None:
        root = Path(dataset_root).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            pass
        else:
            raise ValueError(f"Review decisions must stay outside the source dataset: {resolved}")
    protected = set(family_workflow.PILOT_PROTECTED_PATHS)
    protected.update({
        family_workflow.INVENTORY_PATH.resolve(),
        family_workflow.QUEUE_PATH.resolve(),
        family_workflow.MANIFEST_PATH.resolve(),
    })
    if resolved in protected:
        raise ValueError(f"Refusing to overwrite a protected project file: {resolved}")
    pilot_model = (PROJECT_ROOT / "models" / "paper_cardboard_resnet18_pilot").resolve()
    try:
        resolved.relative_to(pilot_model)
    except ValueError:
        pass
    else:
        raise ValueError(f"Refusing to write decisions in the pilot model directory: {resolved}")
    return resolved


def _validate_audit_path(dataset_root: Path) -> Path:
    path = Path(AUDIT_PATH)
    if path.is_symlink():
        raise ValueError(f"Refusing to write audit log through a symlink: {path}")
    resolved = path.resolve()
    processed_root = (PROJECT_ROOT / "data" / "processed").resolve()
    try:
        resolved.relative_to(processed_root)
    except ValueError as error:
        raise ValueError("Family review audit log must stay under data/processed.") from error
    try:
        resolved.relative_to(Path(dataset_root).resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Family review audit log must stay outside the source dataset.")
    protected = set(family_workflow.PILOT_PROTECTED_PATHS)
    protected.update({
        Path(DECISIONS_PATH).resolve(),
        family_workflow.INVENTORY_PATH.resolve(),
        family_workflow.QUEUE_PATH.resolve(),
        family_workflow.MANIFEST_PATH.resolve(),
    })
    if resolved in protected:
        raise ValueError(f"Refusing to use a protected file as the audit log: {resolved}")
    return resolved


def _preflight_audit_path(dataset_root: Path) -> Path:
    target = _validate_audit_path(dataset_root)
    if target.exists():
        if not target.is_file():
            raise ValueError(f"Family review audit path is not a file: {target}")
        with target.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            header = next(reader, [])
        if tuple(header) != AUDIT_FIELDS:
            raise ValueError(f"Family review audit log has an unexpected schema: {target}")
    return target


def _append_audit_changes(
    records: list[dict[str, str]],
    previous: dict[str, dict[str, str]],
    *,
    dataset_root: Path,
) -> None:
    """Append explicit decision changes to the separate family audit log."""
    target = _preflight_audit_path(dataset_root)
    changes = []
    for record in records:
        path = record["source_relative_path"]
        old = previous.get(path, {
            "review_status": "pending", "reviewed_family": "", "reason": "",
        })
        new = {
            "review_status": record["review_status"],
            "reviewed_family": record["reviewed_family"],
            "reason": record["reason"],
        }
        if old != new:
            changes.append({
                "changed_utc": datetime.now(timezone.utc).isoformat(),
                "source_relative_path": path,
                "previous_review_status": old["review_status"],
                "previous_reviewed_family": old["reviewed_family"],
                "previous_reason": old["reason"],
                "new_review_status": new["review_status"],
                "new_reviewed_family": new["reviewed_family"],
                "new_reason": new["reason"],
            })
    if not changes:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target.exists() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=AUDIT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(changes)


def save_review_decision(
    queue_rows: list[dict[str, str]],
    decisions_path: Path,
    source_relative_path: str,
    choice: str,
    reason: str = "",
    *,
    dataset_root: Path,
    confirm_content: bool = False,
    confirm_replace: bool = False,
) -> dict[str, str]:
    """Merge one explicit choice into the existing CSV without losing decisions."""
    target = validate_decision_output_path(
        decisions_path, expected_path=DECISIONS_PATH, dataset_root=dataset_root
    )
    queue_by_path = {row["source_relative_path"]: row for row in queue_rows}
    relative = family_workflow.normalize_relative_path(source_relative_path)
    if relative not in queue_by_path:
        raise ValueError(f"Image is not in the family review queue: {relative}")

    record = decision_record(
        queue_by_path[relative], choice, reason, confirm_content=confirm_content
    )
    if record["review_status"] == "include":
        image = load_review_image(
            dataset_root, relative, expected_sha256=queue_by_path[relative]["sha256"]
        )
        image.close()
    existing = load_saved_decisions(target, queue_rows)
    previous = existing.get(relative, {
        "review_status": "pending", "reviewed_family": "", "reason": "",
    })
    if previous["review_status"] != "pending" and previous != {
        "review_status": record["review_status"],
        "reviewed_family": record["reviewed_family"],
        "reason": record["reason"],
    } and not confirm_replace:
        raise ValueError(
            f"{relative} already has a saved decision; confirm replacement before changing it."
        )
    _preflight_audit_path(dataset_root)

    decisions = dict(existing)
    decisions[relative] = {
        "review_status": record["review_status"],
        "reviewed_family": record["reviewed_family"],
        "reason": record["reason"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=family_workflow.DECISION_FIELDS)
        writer.writeheader()
        for row in queue_rows:
            item_path = row["source_relative_path"]
            saved = decisions.get(item_path, {
                "review_status": "pending", "reviewed_family": "", "reason": "",
            })
            writer.writerow({"source_relative_path": item_path, **saved})
    _append_audit_changes([record], {relative: previous}, dataset_root=dataset_root)
    return record


def _choice_for_saved(decision: dict[str, str] | None) -> str:
    if not decision or decision["review_status"] == "pending":
        return "PENDING"
    if decision["review_status"] == "include":
        return decision["reviewed_family"]
    return decision["review_status"].upper()


def duplicate_conflicts(
    queue_rows: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Return paths in cross-folder or human-label-conflicting hash groups."""
    groups: dict[str, list[dict[str, str]]] = {}
    for row in queue_rows:
        digest = row.get("sha256", "")
        if digest:
            groups.setdefault(digest, []).append(row)

    conflicts: dict[str, tuple[str, ...]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        folders = {row["source_folder"] for row in members}
        labels = {
            decision.get("reviewed_family", "")
            for row in members
            if (decision := decisions.get(row["source_relative_path"]))
            and decision.get("review_status") == "include"
        }
        if len(folders) > 1 or len(labels) > 1:
            paths = tuple(sorted(row["source_relative_path"] for row in members))
            conflicts.update({path: paths for path in paths})
    return conflicts


def suggested_family_for_row(
    row: dict[str, str],
    conflicts: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """Create a display-only candidate; never treat it as a human decision."""
    relative = row["source_relative_path"]
    if conflicts and relative in conflicts:
        return "Conflicting duplicate — manual review"
    folder = row["source_folder"]
    if folder in {"paper", "cardboard", "plastic", "glass"}:
        return "MSW candidate"
    if folder == "organic":
        return "MSW candidate — confirm food waste"
    if folder == "battery":
        return "E_WASTE candidate — verify content"
    if folder in {"metal", "trash"}:
        return "No suggestion — hold for inspection"
    if folder == "textile":
        return "Unsupported"
    return "No suggestion — hold for inspection"


def batch_selectable_paths(
    all_rows: list[dict[str, str]],
    page_rows: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
    conflicts: dict[str, tuple[str, ...]],
    choice: str,
    readable_paths: set[str] | None = None,
) -> list[str]:
    """Find pending, canonical, readable candidates matching an explicit family choice.

    This only supports UI selection. It never creates or saves a review decision;
    the reviewer must inspect the contact sheet and use the existing confirmation.
    """
    if choice not in family_workflow.FAMILIES:
        return []
    canonical_by_hash: dict[str, str] = {}
    for row in all_rows:
        digest = row.get("sha256", "")
        if digest:
            relative = row["source_relative_path"]
            canonical_by_hash[digest] = min(
                canonical_by_hash.get(digest, relative), relative
            )

    selected: list[str] = []
    seen_hashes: set[str] = set()
    for row in page_rows:
        relative = row["source_relative_path"]
        digest = row.get("sha256", "")
        if (
            not digest
            or row.get("inventory_error")
            or row.get("proposed_review_route") == "unsupported"
            or _choice_for_saved(decisions.get(relative)) != "PENDING"
            or relative in conflicts
            or canonical_by_hash.get(digest) != relative
            or digest in seen_hashes
            or (readable_paths is not None and relative not in readable_paths)
        ):
            continue
        suggestion = suggested_family_for_row(row, conflicts)
        if suggestion == choice or suggestion.startswith(choice + " candidate"):
            selected.append(relative)
            seen_hashes.add(digest)
    return selected


def review_summary_rows(
    queue_rows: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
) -> list[dict[str, str | int]]:
    """Create a compact count table by source folder and displayed decision."""
    counts: dict[tuple[str, str], int] = {}
    for row in queue_rows:
        path = row["source_relative_path"]
        choice = _choice_for_saved(decisions.get(path))
        key = (row["source_folder"], choice)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"source_folder": folder, "decision": decision, "count": count}
        for (folder, decision), count in sorted(counts.items())
    ]


def verified_msw_hash_group_count(
    inventory_rows: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
    dataset_root: Path,
) -> tuple[int, int]:
    """Count approved MSW hash groups whose current files remain readable and intact.

    Returns ``(verified_groups, rejected_groups)``. Duplicate and label-conflict
    policy is shared with family manifest preparation; current bytes and image
    readability are checked before a group is counted.
    """
    eligible = family_workflow.approved_hash_groups(inventory_rows, decisions)
    rows_by_hash: dict[str, list[dict[str, str]]] = {}
    for row in inventory_rows:
        digest = (row.get("sha256") or "").strip()
        if digest:
            rows_by_hash.setdefault(digest, []).append(row)

    verified = rejected = 0
    for digest, (family, _) in eligible.items():
        if family != "MSW":
            continue
        try:
            for member in rows_by_hash[digest]:
                image_path = family_workflow.safe_source_path(
                    Path(dataset_root), member["source_relative_path"]
                )
                if not image_path.is_file() or family_workflow.sha256_file(image_path) != digest:
                    raise ValueError("source is missing or its SHA-256 has changed")
                family_workflow._verify_image(image_path)
        except (OSError, ValueError, KeyError):
            rejected += 1
        else:
            verified += 1
    return verified, rejected


def projected_msw_decisions(
    decisions: dict[str, dict[str, str]],
    queue_rows: list[dict[str, str]],
    selected_paths: list[str],
    choice: str,
    reason: str = "",
) -> dict[str, dict[str, str]]:
    """Return a copy of saved decisions with the proposed batch applied in memory."""
    projected = dict(decisions)
    by_path = {row["source_relative_path"]: row for row in queue_rows}
    for relative in selected_paths:
        if relative not in by_path:
            raise ValueError(f"Image is not in the review queue: {relative}")
        if choice in family_workflow.FAMILIES:
            projected[relative] = {
                "review_status": "include",
                "reviewed_family": choice,
                "reason": reason.strip(),
            }
        elif choice in {"HOLD", "EXCLUDE"}:
            projected[relative] = {
                "review_status": choice.lower(),
                "reviewed_family": "",
                "reason": reason.strip(),
            }
        else:
            raise ValueError(f"Unsupported batch decision: {choice}")
    return projected


def batch_page_sizes(row_count: int) -> list[int]:
    """Return standard page sizes plus an all-images option for larger queues."""
    sizes = [20, 30, 40]
    if row_count > 40:
        sizes.append(row_count)
    return sizes


def batch_grid_columns(page_size: int) -> int:
    """Use a denser six-column layout for the all-images compact view."""
    return 6 if page_size > 40 else 4


def batch_policy_summary(choice: str, folders: set[str]) -> str:
    """Explain the human policy confirmed by a batch decision."""
    if choice == "MSW":
        summary = (
            "You are recording MSW based on the visible contents of every selected image. "
            "Source folders and candidate suggestions are not ground truth; leave mixed or "
            "unclear images pending or HOLD."
        )
        if "organic" in folders:
            summary += " Organic is included only when visible food waste is confirmed."
        return summary
    if choice == "E_WASTE":
        summary = (
            "You are recording E_WASTE based on the visible contents of every selected image. "
            "Source folders and candidate suggestions are not ground truth; leave mixed or "
            "unclear images pending or HOLD."
        )
        if "battery" in folders:
            summary += " Battery images require visible battery/electronic content confirmation."
        return summary
    if choice == "HOLD":
        return "You are holding each selected image for later review; provide a reason."
    if choice == "EXCLUDE":
        return "You are excluding each selected image from the reviewed family set; provide a reason."
    raise ValueError(f"Unsupported batch decision: {choice}")


def _validate_batch(
    queue_rows: list[dict[str, str]],
    decisions_path: Path,
    selected_paths: list[str],
    choice: str,
    reason: str,
    *,
    dataset_root: Path,
    confirm_content: bool,
    confirm_replace: bool,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    if choice not in BATCH_CHOICES:
        raise ValueError(f"Choose one of: {', '.join(BATCH_CHOICES)}")
    if not selected_paths:
        raise ValueError("Select at least one image before preparing a batch decision.")
    normalized = [family_workflow.normalize_relative_path(path) for path in selected_paths]
    if len(normalized) != len(set(normalized)):
        raise ValueError("The batch contains the same image path more than once.")
    queue_by_path = {row["source_relative_path"]: row for row in queue_rows}
    unknown = [path for path in normalized if path not in queue_by_path]
    if unknown:
        raise ValueError(f"Image is not in the review queue: {unknown[0]}")
    for path in normalized:
        family_workflow.safe_source_path(dataset_root, path)

    existing = load_saved_decisions(decisions_path, queue_rows)
    conflicts = duplicate_conflicts(queue_rows, existing)
    if choice in family_workflow.FAMILIES:
        blocked = [path for path in normalized if path in conflicts]
        if blocked:
            conflict_paths = conflicts[blocked[0]]
            raise ValueError(
                "Batch family approval is blocked for conflicting duplicate group: "
                + ", ".join(conflict_paths)
                + ". Review or hold this group individually."
            )

    records: list[dict[str, str]] = []
    for path in normalized:
        row = queue_by_path[path]
        record = decision_record(
            row, choice, reason, confirm_content=confirm_content
        )
        if record["review_status"] == "include":
            image = load_review_image(
                dataset_root, path, expected_sha256=row.get("sha256", "")
            )
            image.close()
        previous = existing.get(path, {
            "review_status": "pending", "reviewed_family": "", "reason": "",
        })
        new_values = {
            "review_status": record["review_status"],
            "reviewed_family": record["reviewed_family"],
            "reason": record["reason"],
        }
        if (
            previous["review_status"] != "pending"
            and previous != new_values
            and not confirm_replace
        ):
            raise ValueError(
                f"{path} already has a saved decision; confirm replacement before changing it."
            )
        records.append(record)
    return records, existing


def preview_batch_decision(
    queue_rows: list[dict[str, str]],
    decisions_path: Path,
    selected_paths: list[str],
    choice: str,
    reason: str = "",
    *,
    dataset_root: Path,
    confirm_content: bool = False,
    confirm_replace: bool = False,
) -> dict[str, object]:
    """Validate a batch and return the confirmation details without writing."""
    _validate_batch(
        queue_rows, decisions_path, selected_paths, choice, reason,
        dataset_root=dataset_root,
        confirm_content=confirm_content,
        confirm_replace=confirm_replace,
    )
    paths = [family_workflow.normalize_relative_path(path) for path in selected_paths]
    folders: dict[str, int] = {}
    queue_by_path = {row["source_relative_path"]: row for row in queue_rows}
    for path in paths:
        folder = queue_by_path[path]["source_folder"]
        folders[folder] = folders.get(folder, 0) + 1
    return {
        "count": len(paths), "choice": choice, "folders": folders, "paths": paths,
        "policy": batch_policy_summary(choice, set(folders)),
    }


def save_review_batch(
    queue_rows: list[dict[str, str]],
    decisions_path: Path,
    selected_paths: list[str],
    choice: str,
    reason: str = "",
    *,
    dataset_root: Path,
    confirmed: bool = False,
    confirm_content: bool = False,
    confirm_replace: bool = False,
) -> list[dict[str, str]]:
    """Save an explicitly confirmed batch while preserving all other decisions."""
    if not confirmed:
        raise ValueError("Confirm the batch summary before saving these decisions.")
    target = validate_decision_output_path(
        decisions_path, expected_path=DECISIONS_PATH, dataset_root=dataset_root
    )
    records, existing = _validate_batch(
        queue_rows, target, selected_paths, choice, reason,
        dataset_root=dataset_root,
        confirm_content=confirm_content,
        confirm_replace=confirm_replace,
    )
    _preflight_audit_path(dataset_root)
    decisions = dict(existing)
    for record in records:
        decisions[record["source_relative_path"]] = {
            "review_status": record["review_status"],
            "reviewed_family": record["reviewed_family"],
            "reason": record["reason"],
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=family_workflow.DECISION_FIELDS)
        writer.writeheader()
        for row in queue_rows:
            path = row["source_relative_path"]
            saved = decisions.get(path, {
                "review_status": "pending", "reviewed_family": "", "reason": "",
            })
            writer.writerow({"source_relative_path": path, **saved})
    _append_audit_changes(records, existing, dataset_root=dataset_root)
    return records


def _next_pending_index(
    rows: list[dict[str, str]], decisions: dict[str, dict[str, str]], current: int
) -> int | None:
    for offset in range(1, len(rows) + 1):
        index = (current + offset) % len(rows)
        path = rows[index]["source_relative_path"]
        if _choice_for_saved(decisions.get(path)) == "PENDING":
            return index
    return None


def _selection_key(source_relative_path: str) -> str:
    return "family_batch_select_" + hashlib.sha1(source_relative_path.encode("utf-8")).hexdigest()


def _batch_signature(paths: list[str], choice: str, reason: str) -> str:
    value = "\n".join([choice, reason.strip(), *sorted(paths)])
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _render_single_review(st, all_rows, rows, decisions, conflicts, dataset_root):
    paths = [row["source_relative_path"] for row in rows]
    if not paths:
        st.info("No images match the current filters.")
        return
    if "family_review_index" not in st.session_state:
        st.session_state.family_review_index = next(
            (i for i, row in enumerate(rows)
             if _choice_for_saved(decisions.get(row["source_relative_path"])) == "PENDING"),
            0,
        )
    st.session_state.family_review_index = min(
        max(int(st.session_state.family_review_index), 0), len(rows) - 1
    )
    index = st.session_state.family_review_index
    nav = st.columns(3)
    if nav[0].button("← Previous", disabled=index == 0, width="stretch"):
        st.session_state.family_review_index = index - 1
        st.rerun()
    if nav[1].button("Next →", disabled=index >= len(rows) - 1, width="stretch"):
        st.session_state.family_review_index = index + 1
        st.rerun()
    next_pending = _next_pending_index(rows, decisions, index)
    if nav[2].button("Resume at next pending", disabled=next_pending is None, width="stretch"):
        if next_pending is not None:
            st.session_state.family_review_index = next_pending
            st.rerun()

    row = rows[index]
    relative = row["source_relative_path"]
    saved = decisions.get(relative)
    suggestion = suggested_family_for_row(row, conflicts)
    st.subheader(f"Image {index + 1} of {len(rows)}")
    st.code(relative, language=None)
    st.caption(
        f"Source folder: **{row['source_folder']}** · "
        f"Image ID: **{row['sha256'][:16] if row['sha256'] else 'unhashed'}** · "
        f"Suggested family: **{suggestion}** · "
        f"Current decision: **{_choice_for_saved(saved)}**"
    )
    if row["source_folder"] == "organic":
        st.caption("Organic suggestion is only a candidate; approve MSW only after confirming food waste.")
    elif row["source_folder"] == "battery":
        st.caption("Battery suggestion is only a candidate; verify visible content before E_WASTE approval.")
    if relative in conflicts:
        st.warning("This image belongs to a conflicting duplicate group: " + ", ".join(conflicts[relative]))

    preview, form_column = st.columns([1.25, 1], gap="large")
    image_readable = False
    with preview:
        try:
            image = load_review_image(
                dataset_root, relative, expected_sha256=row.get("sha256", "")
            )
            st.image(image, caption=relative, width="stretch")
            image.close()
            image_readable = True
        except (OSError, ValueError) as error:
            st.error(f"Image cannot be displayed: {error}")
        if row.get("inventory_error"):
            st.warning(f"Inventory reported: {row['inventory_error']}")

    saved_choice = _choice_for_saved(saved)
    widget_suffix = hashlib.sha1(relative.encode("utf-8")).hexdigest()
    with form_column:
        with st.form(f"family_decision_form_{widget_suffix}"):
            choice = st.selectbox(
                "Human review decision", UI_CHOICES,
                index=UI_CHOICES.index(saved_choice),
                key=f"family_choice_{widget_suffix}",
            )
            reason = st.text_area(
                "Reason / visible evidence", value=(saved or {}).get("reason", ""),
                key=f"family_reason_{widget_suffix}",
            )
            confirm_content = False
            if row["source_folder"] in {"battery", "organic"}:
                confirm_content = st.checkbox(
                    "I inspected this image and confirm its visible content",
                    key=f"family_content_confirmation_{widget_suffix}",
                )
            confirm_replace = False
            if saved and saved["review_status"] != "pending":
                confirm_replace = st.checkbox(
                    "Confirm replacing this previously saved decision",
                    key=f"family_replace_confirmation_{widget_suffix}",
                )
            submitted = st.form_submit_button("Save decision", width="stretch")
        if submitted:
            try:
                if choice in family_workflow.FAMILIES and not image_readable:
                    raise ValueError(f"Cannot include {relative}: the source image is not readable.")
                record = save_review_decision(
                    all_rows, DECISIONS_PATH, relative, choice, reason,
                    dataset_root=dataset_root,
                    confirm_content=confirm_content,
                    confirm_replace=confirm_replace,
                )
                st.session_state.family_review_notice = f"Saved {record['review_status']} decision for {relative}."
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(str(error))


def _render_batch_review(
    st, all_rows, rows, decisions, conflicts, dataset_root,
    readiness_inventory=None, readiness_decisions=None,
):
    if st.session_state.pop("family_clear_batch_pending", False):
        for key in list(st.session_state.keys()):
            if key.startswith((
                "family_batch_select_",
                "family_batch_content_confirmation_",
                "family_batch_replace_confirmation_",
                "family_batch_final_confirmation_",
            )):
                st.session_state.pop(key, None)
    if not rows:
        st.info("No images match the current filters.")
        return
    page_sizes = batch_page_sizes(len(rows))
    page_size = st.selectbox(
        "Images per page",
        page_sizes,
        format_func=lambda size: f"All {size} (compact review)" if size == len(rows) and size > 40 else str(size),
        key="family_batch_page_size",
    )
    if "family_batch_page" not in st.session_state:
        st.session_state.family_batch_page = 0
    page_count = max(1, (len(rows) + page_size - 1) // page_size)
    st.session_state.family_batch_page = min(
        max(int(st.session_state.family_batch_page), 0), page_count - 1
    )
    page_index = st.session_state.family_batch_page
    start = page_index * page_size
    page_rows = rows[start : start + page_size]
    readable_paths: set[str] = set()
    paging = st.columns([1, 2, 1])
    if paging[0].button("← Previous page", disabled=page_index == 0, width="stretch"):
        st.session_state.family_batch_page -= 1
        st.rerun()
    paging[1].markdown(f"**Page {page_index + 1} of {page_count}** · {len(rows)} filtered images")
    if paging[2].button("Next page →", disabled=page_index + 1 >= page_count, width="stretch"):
        st.session_state.family_batch_page += 1
        st.rerun()

    grid_columns = batch_grid_columns(page_size)
    for offset in range(0, len(page_rows), grid_columns):
        tile_columns = st.columns(grid_columns, gap="small")
        for column, row in zip(tile_columns, page_rows[offset : offset + grid_columns]):
            relative = row["source_relative_path"]
            path_token = hashlib.sha1(relative.encode("utf-8")).hexdigest()
            with column:
                try:
                    image = load_review_image(
                        dataset_root, relative, expected_sha256=row.get("sha256", "")
                    )
                    image.thumbnail((220, 160) if grid_columns == 6 else (420, 300))
                    st.image(image, width="stretch")
                    image.close()
                    readable_paths.add(relative)
                except (OSError, ValueError) as error:
                    st.error(f"Image unavailable: {error}")
                st.caption(
                    f"**{row['source_folder']}** · **{Path(relative).name}** · "
                    f"{_choice_for_saved(decisions.get(relative))}"
                )
                st.code(relative, language=None)
                st.caption(f"ID: {row['sha256'][:12] if row.get('sha256') else 'unhashed'}")
                st.caption("Suggested family: " + suggested_family_for_row(row, conflicts))
                if row["source_folder"] == "organic":
                    st.caption("Confirm visible food waste before an MSW decision.")
                elif row["source_folder"] == "battery":
                    st.caption("Verify visible battery/electronic content before E_WASTE.")
                if relative in conflicts:
                    st.warning("Conflicting duplicate group; batch family approval is disabled.")
                st.checkbox(
                    "Select for batch",
                    key=_selection_key(relative),
                    help="Selecting an image never approves it; a separate batch confirmation is required.",
                )

    selected_paths = [
        row["source_relative_path"] for row in all_rows
        if st.session_state.get(_selection_key(row["source_relative_path"]), False)
    ]
    selected_rows = [row for row in all_rows if row["source_relative_path"] in selected_paths]
    affected_folders: dict[str, int] = {}
    for row in selected_rows:
        folder = row["source_folder"]
        affected_folders[folder] = affected_folders.get(folder, 0) + 1
    st.markdown(f"**Selected:** {len(selected_paths)} image(s)")
    if affected_folders:
        st.caption("Selected folders: " + ", ".join(f"{folder} ({count})" for folder, count in sorted(affected_folders.items())))
    if st.button("Clear all batch selections"):
        st.session_state.pop("family_pending_batch", None)
        st.session_state.family_clear_batch_pending = True
        st.rerun()

    choice = st.selectbox("Apply this explicit decision", BATCH_CHOICES, key="family_batch_choice")
    if choice in family_workflow.FAMILIES:
        selectable = batch_selectable_paths(
            all_rows, page_rows, decisions, conflicts, choice, readable_paths
        )
        st.caption(
            "Bulk selection only checks matching pending candidates on this page. "
            "Inspect every selected image; this does not approve or save anything."
        )
        if st.button(
            f"Select {len(selectable)} matching pending {choice} candidate(s) on this page",
            disabled=not selectable,
        ):
            for path in selectable:
                st.session_state[_selection_key(path)] = True
            st.rerun()
    reason = st.text_input(
        "Shared reason (required for HOLD / EXCLUDE and organic / battery approvals)",
        key="family_batch_reason",
    )
    batch_signature = _batch_signature(selected_paths, choice, reason)
    selected_folders = {row["source_folder"] for row in selected_rows}
    confirm_content = False
    if choice in family_workflow.FAMILIES and selected_folders & {"organic", "battery"}:
        special_text = []
        if "organic" in selected_folders:
            special_text.append("organic images visibly show food waste")
        if "battery" in selected_folders:
            special_text.append("battery images visibly show battery/electronic content")
        confirm_content = st.checkbox(
            "I inspected every selected image and confirm that " + " and ".join(special_text),
            key=f"family_batch_content_confirmation_{batch_signature}",
        )
    intended_status = "include" if choice in family_workflow.FAMILIES else choice.lower()
    intended_family = choice if choice in family_workflow.FAMILIES else ""
    intended_reason = "" if choice == "PENDING" else reason.strip()
    changed_paths = []
    for path in selected_paths:
        previous = decisions.get(path)
        if not previous or previous["review_status"] == "pending":
            continue
        if previous != {
            "review_status": intended_status,
            "reviewed_family": intended_family,
            "reason": intended_reason,
        }:
            changed_paths.append(path)
    confirm_replace = False
    if changed_paths:
        confirm_replace = st.checkbox(
            f"Confirm replacing {len(changed_paths)} previously saved decision(s)",
            key=f"family_batch_replace_confirmation_{batch_signature}",
        )
    if st.button("Review batch before saving", disabled=not selected_paths):
        st.session_state.pop("family_pending_batch", None)
        try:
            preview = preview_batch_decision(
                all_rows, DECISIONS_PATH, selected_paths, choice, reason,
                dataset_root=dataset_root,
                confirm_content=confirm_content,
                confirm_replace=confirm_replace,
            )
            preview_version = int(st.session_state.get("family_batch_preview_version", 0)) + 1
            st.session_state.family_batch_preview_version = preview_version
            st.session_state.family_pending_batch = {
                **preview,
                "reason": reason,
                "confirm_content": confirm_content,
                "confirm_replace": confirm_replace,
                "preview_version": preview_version,
            }
            if choice == "MSW" and readiness_inventory is not None and readiness_decisions is not None:
                projected = projected_msw_decisions(
                    readiness_decisions, all_rows, preview["paths"], choice, reason
                )
                count, rejected = verified_msw_hash_group_count(
                    readiness_inventory, projected, dataset_root
                )
                st.session_state.family_pending_batch["verified_msw_groups"] = count
                st.session_state.family_pending_batch["rejected_msw_groups"] = rejected
        except (OSError, ValueError) as error:
            st.error(str(error))

    pending_batch = st.session_state.get("family_pending_batch")
    if pending_batch:
        st.warning(
            f"Confirm applying **{pending_batch['choice']}** to "
            f"**{pending_batch['count']} image(s)** across "
            + ", ".join(
                f"**{folder} ({count})**"
                for folder, count in sorted(pending_batch["folders"].items())
            )
            + ". This will write human review decisions."
        )
        st.info(pending_batch["policy"])
        if "verified_msw_groups" in pending_batch:
            st.success(
                "After this confirmed choice: "
                f"**{pending_batch['verified_msw_groups']} distinct, readable, "
                "non-conflicting approved MSW hash group(s)** in the full inventory."
            )
            if pending_batch.get("rejected_msw_groups"):
                st.warning(
                    f"{pending_batch['rejected_msw_groups']} otherwise eligible MSW hash "
                    "group(s) were not counted because a source file failed current hash/readability checks."
                )
        if pending_batch["reason"]:
            st.caption(f"Shared reason to be recorded: {pending_batch['reason']}")
        with st.expander("Review exact paths in this batch"):
            for path in pending_batch["paths"]:
                st.code(path, language=None)
        confirm_columns = st.columns(2)
        if confirm_columns[0].button("Confirm and save batch"):
            try:
                saved_records = save_review_batch(
                    all_rows, DECISIONS_PATH, pending_batch["paths"],
                    pending_batch["choice"], pending_batch["reason"],
                    dataset_root=dataset_root,
                    confirmed=True,
                    confirm_content=pending_batch["confirm_content"],
                    confirm_replace=pending_batch["confirm_replace"],
                )
                st.session_state.pop("family_pending_batch", None)
                st.session_state.family_clear_batch_pending = True
                st.session_state.family_review_notice = f"Saved {len(saved_records)} batch decision(s)."
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(str(error))
        if confirm_columns[1].button("Cancel batch"):
            st.session_state.pop("family_pending_batch", None)
            st.session_state.family_clear_batch_pending = True
            st.rerun()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="VISTA Family Review", page_icon="♻️", layout="wide")
    st.title("VISTA Family Dataset Review")
    st.caption("Human review queue · suggestions are not ground-truth labels")
    st.caption(f"Queue: `{QUEUE_PATH}` · Decisions: `{DECISIONS_PATH}`")
    st.info(
        "No image recognition is used here. Source-folder suggestions are display-only. "
        "Only your explicit, confirmed decisions are saved."
    )

    try:
        rows = load_review_queue(QUEUE_PATH, DATASET_DIR)
        decisions = load_saved_decisions(DECISIONS_PATH, rows)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return

    readiness_inventory = None
    readiness_decisions = None
    try:
        readiness_inventory = family_workflow._load_inventory(family_workflow.INVENTORY_PATH)
        full_queue = family_workflow._load_queue(
            family_workflow.QUEUE_PATH, readiness_inventory
        )
        readiness_decisions = family_workflow._load_decisions(
            family_workflow.DECISIONS_PATH,
            readiness_inventory,
            full_queue,
        )
        # A separately configured batch file overlays only explicit decisions;
        # its pending template rows never erase the original five E_WASTE labels.
        for path, decision in decisions.items():
            if decision["review_status"] != "pending":
                readiness_decisions[path] = decision
        msw_groups, rejected_msw_groups = verified_msw_hash_group_count(
            readiness_inventory, readiness_decisions, DATASET_DIR
        )
        st.metric("Verified approved MSW hash groups", msw_groups)
        if rejected_msw_groups:
            st.warning(
                f"{rejected_msw_groups} eligible MSW hash group(s) were not counted "
                "because current source hash/readability checks failed."
            )
    except (OSError, ValueError) as error:
        st.warning(f"Verified MSW group count is unavailable: {error}")

    conflicts = duplicate_conflicts(rows, decisions)
    summary = review_summary_rows(rows, decisions)
    summary_stream = io.StringIO()
    summary_writer = csv.DictWriter(summary_stream, fieldnames=("source_folder", "decision", "count"))
    summary_writer.writeheader()
    summary_writer.writerows(summary)
    st.download_button(
        "Export review summary CSV",
        data=summary_stream.getvalue(),
        file_name="family_review_summary.csv",
        mime="text/csv",
    )

    sidebar = st.sidebar
    sidebar.header("Filter review queue")
    folder_filter = sidebar.selectbox(
        "Source folder", ("All", *family_workflow.SOURCE_FOLDERS), key="family_filter_folder"
    )
    status_filter = sidebar.selectbox(
        "Review status", ("All", "Pending", "Reviewed"), key="family_filter_status"
    )
    suggestion_filter = sidebar.selectbox(
        "Suggested family", ("All", *SUGGESTION_FILTERS), key="family_filter_suggestion"
    )
    filtered_rows = []
    for row in rows:
        path = row["source_relative_path"]
        status = _choice_for_saved(decisions.get(path))
        suggestion = suggested_family_for_row(row, conflicts)
        if folder_filter != "All" and row["source_folder"] != folder_filter:
            continue
        if status_filter == "Pending" and status != "PENDING":
            continue
        if status_filter == "Reviewed" and status == "PENDING":
            continue
        if suggestion_filter != "All" and suggestion != suggestion_filter:
            continue
        filtered_rows.append(row)

    mode = st.radio(
        "Review mode", ("Single image", "Batch review"), horizontal=True,
        key="family_review_mode",
    )
    if st.session_state.get("family_review_notice"):
        st.success(st.session_state.pop("family_review_notice"))

    status_by_path = {
        row["source_relative_path"]: _choice_for_saved(decisions.get(row["source_relative_path"]))
        for row in rows
    }
    reviewed = sum(status != "PENDING" for status in status_by_path.values())
    included = sum(status in family_workflow.FAMILIES for status in status_by_path.values())
    held = sum(status == "HOLD" for status in status_by_path.values())
    excluded = sum(status == "EXCLUDE" for status in status_by_path.values())
    metric_columns = st.columns(4)
    metric_columns[0].metric("Reviewed", f"{reviewed} / {len(rows)}")
    metric_columns[1].metric("Included", included)
    metric_columns[2].metric("Held", held)
    metric_columns[3].metric("Excluded", excluded)
    st.progress(reviewed / len(rows))
    if mode == "Single image":
        _render_single_review(st, rows, filtered_rows, decisions, conflicts, DATASET_DIR)
    else:
        _render_batch_review(
            st, rows, filtered_rows, decisions, conflicts, DATASET_DIR,
            readiness_inventory, readiness_decisions,
        )


if __name__ == "__main__":
    main()
