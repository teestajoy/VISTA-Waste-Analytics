"""Inventory, review, and prepare a separate MSW/E-waste family dataset.

Folder names create review routes only. They never become reviewed family
labels. Source images are read-only; all generated files are separate outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "Waste_classification_dataset"
    / "final_merged_dataset"
)
INVENTORY_PATH = PROJECT_ROOT / "data" / "processed" / "family_dataset_inventory.csv"
QUEUE_PATH = PROJECT_ROOT / "data" / "processed" / "family_review_queue.csv"
DECISIONS_PATH = PROJECT_ROOT / "data" / "processed" / "family_review_decisions.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "family_split_manifest.csv"
MERGED_DECISIONS_PATH = PROJECT_ROOT / "data" / "processed" / "family_review_decisions_msw_batch.csv"
SHEETS_DIR = PROJECT_ROOT / "test-data" / "family_review"

SOURCE_FOLDERS = (
    "plastic", "cardboard", "glass", "metal", "organic", "paper",
    "battery", "textile", "trash",
)
REVIEW_ROUTES = {
    "paper": "candidate_msw",
    "cardboard": "candidate_msw",
    "plastic": "candidate_msw",
    "glass": "candidate_msw",
    "organic": "candidate_msw_food_waste_only",
    "battery": "candidate_e_waste_verify_content",
    "metal": "hold_manual_review",
    "trash": "hold_manual_review",
    "textile": "unsupported",
}
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
FAMILIES = frozenset({"MSW", "E_WASTE"})
REVIEW_STATUSES = frozenset({"pending", "include", "hold", "exclude"})
INVENTORY_FIELDS = (
    "source_relative_path", "sha256", "source_folder", "proposed_review_route",
    "reviewed_family", "review_status", "duplicate_group_id", "inventory_error",
)
QUEUE_FIELDS = (
    "source_relative_path", "sha256", "source_folder", "proposed_review_route",
    "duplicate_group_id", "inventory_error",
)
DECISION_FIELDS = ("source_relative_path", "review_status", "reviewed_family", "reason")
MANIFEST_FIELDS = (
    "source_relative_path", "sha256", "source_folder", "proposed_review_route",
    "reviewed_family", "review_status", "review_reason", "duplicate_group_id",
    "inventory_error", "group_status", "split", "review_exclusion_reason",
)
PILOT_PROTECTED_PATHS = {
    (PROJECT_ROOT / "data" / "processed" / "paper_cardboard_pilot_review.csv").resolve(),
    (PROJECT_ROOT / "data" / "processed" / "paper_cardboard_pilot_split_manifest.csv").resolve(),
}
TILE_SIZE = (220, 170)
PAGE_COLUMNS = 4
PAGE_ROWS = 3
PAGE_SIZE = (1040, 960)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative_path(value: str) -> str:
    """Accept repository-style relative paths and reject traversal/absolute paths."""
    normalized = value.strip().replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value.strip())
    if (
        not normalized
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ValueError(f"Unsafe source_relative_path: {value!r}")
    return posix.as_posix()


def safe_source_path(dataset_root: Path, source_relative_path: str) -> Path:
    relative = normalize_relative_path(source_relative_path)
    root = dataset_root.resolve()
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Source path traverses a symlink: {source_relative_path}")
    target = candidate.resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Source path resolves outside the dataset root: {source_relative_path}"
        ) from error
    return target


def _is_symlink_path(path: Path, root: Path) -> bool:
    current = path
    while current != root and current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def discover_inventory(dataset_root: Path) -> list[dict[str, str]]:
    """Recursively hash and verify supported images under the nine known folders."""
    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    rows: list[dict[str, str]] = []
    for folder in SOURCE_FOLDERS:
        category_dir = root / folder
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Required dataset category folder not found: {category_dir}")
        for path in sorted(category_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if (
                path.suffix.lower() not in IMAGE_EXTENSIONS
                or not path.is_file()
                or _is_symlink_path(path, root)
            ):
                continue
            relative = path.relative_to(root).as_posix()
            file_hash = ""
            error_text = ""
            try:
                file_hash = sha256_file(path)
                _verify_image(path)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                error_text = f"Could not read or decode image: {error}"
            rows.append({
                "source_relative_path": relative,
                "sha256": file_hash,
                "source_folder": folder,
                "proposed_review_route": REVIEW_ROUTES[folder],
                "reviewed_family": "",
                "review_status": "pending",
                "duplicate_group_id": f"sha256:{file_hash}" if file_hash else "",
                "inventory_error": error_text,
            })
    return sorted(rows, key=lambda row: row["source_relative_path"].casefold())


def _write_csv_new(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, required_fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(required_fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def _validate_new_output(path: Path, dataset_root: Path) -> Path:
    resolved = path.resolve()
    root = dataset_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(f"Output must be outside the source dataset: {resolved}")
    if resolved in PILOT_PROTECTED_PATHS:
        raise ValueError(f"Refusing to reuse a paper/cardboard pilot file: {resolved}")
    pilot_model_dir = (PROJECT_ROOT / "models" / "paper_cardboard_resnet18_pilot").resolve()
    try:
        resolved.relative_to(pilot_model_dir)
    except ValueError:
        pass
    else:
        raise ValueError(f"Refusing to write family-review output in the paper/cardboard model directory: {resolved}")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {resolved}")
    return resolved


def write_inventory(dataset_root: Path, output_path: Path) -> list[dict[str, str]]:
    output_path = _validate_new_output(output_path, dataset_root)
    rows = discover_inventory(dataset_root)
    _write_csv_new(output_path, INVENTORY_FIELDS, rows)
    return rows


def _load_inventory(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path, INVENTORY_FIELDS)
    known = set(SOURCE_FOLDERS)
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        relative = normalize_relative_path(row.get("source_relative_path", ""))
        if relative in seen:
            raise ValueError(f"Inventory repeats source path on line {line}: {relative}")
        seen.add(relative)
        folder = (row.get("source_folder") or "").strip()
        if folder not in known or PurePosixPath(relative).parts[0] != folder:
            raise ValueError(f"Inventory has inconsistent source folder for {relative}")
        if row.get("proposed_review_route") != REVIEW_ROUTES[folder]:
            raise ValueError(f"Inventory has an unexpected review route for {relative}")
        digest = (row.get("sha256") or "").strip()
        if digest and (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower())):
            raise ValueError(f"Inventory has an invalid SHA-256 for {relative}")
        if row.get("sha256") and row.get("duplicate_group_id") != f"sha256:{row['sha256']}":
            raise ValueError(f"Inventory has inconsistent duplicate group for {relative}")
        if row.get("review_status") != "pending":
            raise ValueError(f"Inventory must remain an unreviewed queue: {relative}")
    return rows


def create_review_queue(
    inventory_path: Path,
    dataset_root: Path,
    *,
    per_folder: int = 40,
    seed: int = 42,
    queue_path: Path = QUEUE_PATH,
    decisions_path: Path = DECISIONS_PATH,
    sheets_dir: Path = SHEETS_DIR,
) -> list[dict[str, str]]:
    """Select a reproducible review sample and render labelled, multi-page PDFs."""
    if per_folder < 1:
        raise ValueError("per_folder must be at least 1")
    rows = _load_inventory(inventory_path)
    queue_path = _validate_new_output(queue_path, dataset_root)
    decisions_path = _validate_new_output(decisions_path, dataset_root)
    sheets_dir = _validate_new_output(sheets_dir, dataset_root)
    destinations = (queue_path, decisions_path, sheets_dir)
    if len(set(destinations)) != 3:
        raise ValueError("Queue CSV, decision CSV, and contact-sheet directory must differ")
    for index, first in enumerate(destinations):
        for second in destinations[index + 1 :]:
            if first in second.parents or second in first.parents:
                raise ValueError("Queue, decision, and contact-sheet output paths cannot contain one another")

    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            by_hash[row["sha256"]].append(row)
    cross_folder_hashes = {
        digest for digest, members in by_hash.items()
        if len({row["source_folder"] for row in members}) > 1
    }
    chosen: dict[str, dict[str, str]] = {}
    for digest in cross_folder_hashes:
        for row in by_hash[digest]:
            chosen[row["source_relative_path"]] = row

    for folder_index, folder in enumerate(SOURCE_FOLDERS):
        if folder == "textile":
            continue
        canonical: dict[str, dict[str, str]] = {}
        for row in rows:
            if (
                row["source_folder"] == folder
                and row["sha256"]
                and not row["inventory_error"]
                and row["sha256"] not in cross_folder_hashes
            ):
                previous = canonical.get(row["sha256"])
                if previous is None or row["source_relative_path"] < previous["source_relative_path"]:
                    canonical[row["sha256"]] = row
        candidates = sorted(canonical.values(), key=lambda item: item["source_relative_path"])
        random.Random(seed + folder_index).shuffle(candidates)
        for row in candidates[:per_folder]:
            chosen[row["source_relative_path"]] = row

    queue = sorted(chosen.values(), key=lambda row: row["source_relative_path"].casefold())
    if not queue:
        raise ValueError("No readable, hashed images are available for the review queue")
    queue_rows = [{field: row.get(field, "") or "" for field in QUEUE_FIELDS} for row in queue]
    queue_rows.sort(key=lambda row: row["source_relative_path"].casefold())

    # Build all sheets before writing outputs so a stale/unsafe source cannot leave
    # a partial decision queue behind.
    pages_by_folder: dict[str, list[Image.Image]] = {}
    for folder in SOURCE_FOLDERS:
        folder_rows = [row for row in queue_rows if row["source_folder"] == folder]
        if folder_rows:
            pages_by_folder[folder] = _render_pdf_pages(folder_rows, dataset_root)
    _write_csv_new(queue_path, QUEUE_FIELDS, queue_rows)
    blank_decisions = [
        {"source_relative_path": row["source_relative_path"], "review_status": "pending",
         "reviewed_family": "", "reason": ""}
        for row in queue_rows
    ]
    _write_csv_new(decisions_path, DECISION_FIELDS, blank_decisions)
    sheets_dir.mkdir(parents=True, exist_ok=False)
    for folder, pages in pages_by_folder.items():
        pdf_path = sheets_dir / f"{folder}.pdf"
        with pdf_path.open("xb") as output:
            pages[0].save(output, format="PDF", save_all=True, append_images=pages[1:], resolution=100)
        for page in pages:
            page.close()
    return queue_rows


def _wrap_pixel_width(text: str, font, max_width: int) -> list[str]:
    """Wrap without dropping or changing any characters in a source path."""
    lines: list[str] = []
    for original_line in text.splitlines() or [""]:
        current = ""
        for char in original_line:
            if current and font.getlength(current + char) > max_width:
                lines.append(current)
                current = char
            else:
                current += char
        lines.append(current)
    return lines


def _render_pdf_pages(rows: list[dict[str, str]], dataset_root: Path) -> list[Image.Image]:
    font = ImageFont.load_default()
    pages: list[Image.Image] = []
    per_page = PAGE_COLUMNS * PAGE_ROWS
    for start in range(0, len(rows), per_page):
        page_rows = rows[start : start + per_page]
        captions = [
            _wrap_pixel_width(
                f"{row['source_folder']} | {row['proposed_review_route']}\n{row['source_relative_path']}",
                font,
                242,
            )
            for row in page_rows
        ]
        max_caption_lines = max(map(len, captions), default=1)
        line_height = 13
        cell_height = TILE_SIZE[1] + 8 + max_caption_lines * line_height + 12
        sheet_height = 18 + PAGE_ROWS * cell_height
        sheet = Image.new("RGB", (PAGE_SIZE[0], sheet_height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page_rows):
            column, line = index % PAGE_COLUMNS, index // PAGE_COLUMNS
            x, y = 18 + column * 255, 12 + line * cell_height
            if row["inventory_error"]:
                draw.rectangle((x, y, x + TILE_SIZE[0], y + TILE_SIZE[1]), fill="#f1dddd")
                draw.text((x + 8, y + 8), "Unreadable at inventory time", fill="#751414", font=font)
            else:
                image_path = safe_source_path(dataset_root, row["source_relative_path"])
                if not image_path.is_file():
                    sheet.close()
                    raise ValueError(f"Missing review image: {row['source_relative_path']}")
                if sha256_file(image_path) != row["sha256"]:
                    sheet.close()
                    raise ValueError(f"Image changed since inventory: {row['source_relative_path']}")
                try:
                    with Image.open(image_path) as source:
                        source.load()
                        thumbnail = source.convert("RGB")
                        thumbnail.thumbnail(TILE_SIZE)
                except (OSError, ValueError, UnidentifiedImageError) as error:
                    sheet.close()
                    raise ValueError(f"Could not render {row['source_relative_path']}: {error}") from error
                sheet.paste(thumbnail, (x + (TILE_SIZE[0] - thumbnail.width) // 2, y))
                thumbnail.close()
            for caption_index, caption_line in enumerate(captions[index]):
                draw.text((x, y + TILE_SIZE[1] + 4 + caption_index * line_height), caption_line, fill="black", font=font)
        pages.append(sheet)
    return pages


def _load_queue(path: Path, inventory_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    queue = _read_csv(path, QUEUE_FIELDS)
    inventory_by_path = {row["source_relative_path"]: row for row in inventory_rows}
    seen: set[str] = set()
    for line, row in enumerate(queue, start=2):
        relative = normalize_relative_path(row.get("source_relative_path", ""))
        if relative in seen:
            raise ValueError(f"Review queue repeats path on line {line}: {relative}")
        seen.add(relative)
        source = inventory_by_path.get(relative)
        if source is None:
            raise ValueError(f"Review queue contains a path absent from inventory: {relative}")
        for field in QUEUE_FIELDS:
            if field == "source_relative_path":
                continue
            if (row.get(field) or "") != (source.get(field) or ""):
                raise ValueError(f"Review queue does not match inventory for {relative}: {field}")
    return queue


def _load_decisions(
    path: Path,
    inventory_rows: list[dict[str, str]],
    queue_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    rows = _read_csv(path, DECISION_FIELDS)
    known_paths = {row["source_relative_path"] for row in queue_rows}
    decisions: dict[str, dict[str, str]] = {}
    inventory_by_path = {row["source_relative_path"]: row for row in inventory_rows}
    for line, row in enumerate(rows, start=2):
        relative = normalize_relative_path(row.get("source_relative_path", ""))
        if relative not in known_paths:
            raise ValueError(f"Decision CSV line {line} has an unknown path: {relative}")
        if relative in decisions:
            raise ValueError(f"Decision CSV repeats path on line {line}: {relative}")
        status = (row.get("review_status") or "").strip().lower()
        family = (row.get("reviewed_family") or "").strip().upper()
        reason = (row.get("reason") or "").strip()
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Decision for {relative} must be pending, include, hold, or exclude")
        if status == "include":
            if family not in FAMILIES:
                raise ValueError(f"Included image {relative} needs reviewed_family MSW or E_WASTE")
            inventory_row = inventory_by_path[relative]
            if not inventory_row["sha256"] or inventory_row["inventory_error"]:
                raise ValueError(f"Unreadable or unhashed image {relative} cannot be included")
            if inventory_row["proposed_review_route"] == "unsupported":
                raise ValueError(f"{relative} is unsupported for family inclusion until policy changes")
            if inventory_row["source_folder"] == "organic":
                if family != "MSW":
                    raise ValueError(f"Organic image {relative} can only be included as MSW after food-waste review")
                if not reason:
                    raise ValueError(f"Organic image {relative} needs a reason confirming visible food waste")
            if inventory_row["source_folder"] == "battery":
                if family != "E_WASTE":
                    raise ValueError(f"Battery image {relative} can only be included as E_WASTE after content review")
                if not reason:
                    raise ValueError(f"Battery image {relative} needs a reason confirming visible battery/electronic context")
        elif family:
            raise ValueError(f"Only included image {relative} may have a reviewed_family")
        if status in {"hold", "exclude"} and not reason:
            raise ValueError(f"{status.title()} decision for {relative} requires a reason")
        if status == "pending" and reason:
            raise ValueError(f"Pending image {relative} cannot have a decision reason")
        decisions[relative] = {"review_status": status, "reviewed_family": family, "reason": reason}
    return decisions


def _validate_current_sources(dataset_root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        path = safe_source_path(dataset_root, row["source_relative_path"])
        if _is_symlink_path(path, dataset_root.resolve()):
            raise ValueError(f"Source path traverses a symlink: {row['source_relative_path']}")
        if row["inventory_error"]:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Inventoried image is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != row["sha256"]:
            raise ValueError(f"SHA-256 changed since inventory: {row['source_relative_path']}")
        try:
            _verify_image(path)
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise ValueError(f"Image is no longer readable: {row['source_relative_path']}: {error}") from error


def _split_counts(count: int) -> tuple[int, int, int]:
    if count < 3:
        raise ValueError("At least three unique approved images per family are required")
    targets = (count * 0.70, count * 0.15, count * 0.15)
    result = [1, 1, 1]
    for _ in range(count - 3):
        index = max(range(3), key=lambda current: (targets[current] - result[current], -current))
        result[index] += 1
    return tuple(result)  # type: ignore[return-value]


def approved_hash_groups(
    inventory_rows: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
) -> dict[str, tuple[str, dict[str, str]]]:
    """Return eligible SHA-256 groups using the preparation rules, without splitting.

    A hash is counted once, at its deterministic canonical source path. Any
    unreadable member, hold/exclude decision, conflicting label, or unresolved
    cross-folder duplicate blocks that hash group.
    """
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for source in inventory_rows:
        digest = (source.get("sha256") or "").strip()
        if digest:
            decision = decisions.get(source["source_relative_path"], {
                "review_status": "pending", "reviewed_family": "", "reason": "",
            })
            by_hash[digest].append({**source, **decision})

    eligible: dict[str, tuple[str, dict[str, str]]] = {}
    for digest, members in by_hash.items():
        members.sort(key=lambda row: row["source_relative_path"])
        folders = {row["source_folder"] for row in members}
        statuses = {row.get("review_status", "pending") for row in members}
        labels = {
            row.get("reviewed_family", "")
            for row in members
            if row.get("review_status") == "include"
        }
        if any(row.get("inventory_error") for row in members):
            continue
        if len(labels) > 1 or "hold" in statuses or "exclude" in statuses:
            continue
        if len(folders) > 1 and not (statuses == {"include"} and len(labels) == 1):
            continue
        canonical = members[0]
        if canonical.get("review_status") != "include" or len(labels) != 1:
            continue
        family = next(iter(labels))
        if family in FAMILIES:
            eligible[digest] = (family, canonical)
    return eligible


def family_readiness_counts(
    inventory_rows: list[dict[str, str]],
    decisions: dict[str, dict[str, str]],
) -> dict[str, int]:
    """Count distinct, eligible approved hashes for each family."""
    groups = approved_hash_groups(inventory_rows, decisions)
    return {
        family: sum(label == family for label, _ in groups.values())
        for family in sorted(FAMILIES)
    }


def merge_review_batch_decisions(
    inventory_path: Path,
    queue_path: Path,
    decisions_path: Path,
    batch_queue_path: Path,
    batch_decisions_path: Path,
    dataset_root: Path,
    output_path: Path,
    *,
    confirmed: bool = False,
) -> list[dict[str, str]]:
    """Write a new merged decisions CSV, preserving all source review decisions.

    The output is always a new file. Batch decisions must refer to paths already
    present in the original queue, and included images are re-hashed and checked
    for canonical duplicate-group eligibility before the output is written.
    """
    if not confirmed:
        raise ValueError("No output written. Explicitly confirm the batch merge before saving.")
    output_path = _validate_new_output(output_path, dataset_root)
    inventory = _load_inventory(inventory_path)
    queue = _load_queue(queue_path, inventory)
    existing = _load_decisions(decisions_path, inventory, queue)
    batch_queue = _load_queue(batch_queue_path, inventory)
    queue_by_path = {row["source_relative_path"]: row for row in queue}
    for row in batch_queue:
        safe_source_path(dataset_root, row["source_relative_path"])
        original = queue_by_path.get(row["source_relative_path"])
        if original is None or any(
            (row.get(field) or "") != (original.get(field) or "")
            for field in QUEUE_FIELDS if field != "source_relative_path"
        ):
            raise ValueError(
                f"Batch image is not an unchanged member of the original review queue: {row['source_relative_path']}"
            )
    batch_decisions = _load_decisions(batch_decisions_path, inventory, batch_queue)
    merged = dict(existing)
    changed_paths: list[str] = []
    for relative, decision in batch_decisions.items():
        if decision["review_status"] == "pending":
            continue
        previous = existing.get(relative, {
            "review_status": "pending", "reviewed_family": "", "reason": "",
        })
        if previous["review_status"] != "pending" and previous != decision:
            raise ValueError(
                f"Batch decision would replace an existing decision for {relative}; resolve it manually first."
            )
        merged[relative] = decision
        changed_paths.append(relative)

    if changed_paths:
        eligible = approved_hash_groups(inventory, merged)
        inventory_by_path = {row["source_relative_path"]: row for row in inventory}
        for relative in changed_paths:
            decision = merged[relative]
            if decision["review_status"] != "include":
                continue
            source = inventory_by_path[relative]
            digest = source["sha256"]
            if digest not in eligible or eligible[digest][1]["source_relative_path"] != relative:
                raise ValueError(
                    f"Included batch decision for {relative} is not a canonical, conflict-free hash group."
                )
            path = safe_source_path(dataset_root, relative)
            if not path.is_file():
                raise FileNotFoundError(f"Reviewed batch image is missing: {relative}")
            if sha256_file(path) != digest:
                raise ValueError(f"SHA-256 changed since inventory: {relative}")
            try:
                _verify_image(path)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                raise ValueError(f"Reviewed batch image is unreadable: {relative}: {error}") from error

    output_rows = [
        {
            "source_relative_path": row["source_relative_path"],
            **merged.get(row["source_relative_path"], {
                "review_status": "pending", "reviewed_family": "", "reason": "",
            }),
        }
        for row in queue
    ]
    _write_csv_new(output_path, DECISION_FIELDS, output_rows)
    return output_rows


def prepare_family_manifest(
    inventory_path: Path,
    review_csv: Path,
    dataset_root: Path,
    output_path: Path,
    *,
    queue_path: Path = QUEUE_PATH,
    seed: int = 42,
) -> list[dict[str, str]]:
    output_path = _validate_new_output(output_path, dataset_root)
    inventory = _load_inventory(inventory_path)
    _validate_current_sources(dataset_root, inventory)
    queue = _load_queue(queue_path, inventory)
    decisions = _load_decisions(review_csv, inventory, queue)
    rows: list[dict[str, str]] = []
    for source in inventory:
        decision = decisions.get(source["source_relative_path"], {
            "review_status": "pending", "reviewed_family": "", "reason": "",
        })
        rows.append({
            **source,
            "review_status": decision["review_status"],
            "reviewed_family": decision["reviewed_family"],
            "review_reason": decision["reason"],
            "group_status": "pending",
            "split": "",
            "review_exclusion_reason": source["inventory_error"],
        })

    for row in rows:
        if not row["sha256"]:
            row["group_status"] = "exclude"
            row["review_exclusion_reason"] = row["inventory_error"] or "No SHA-256 was recorded; this path cannot be used in a split."

    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            by_hash[row["sha256"]].append(row)
    for digest, members in by_hash.items():
        members.sort(key=lambda item: item["source_relative_path"])
        folders = {member["source_folder"] for member in members}
        family_labels = {member["reviewed_family"] for member in members if member["review_status"] == "include"}
        statuses = {member["review_status"] for member in members}
        reason = ""
        group_status = "pending"
        if any(member["inventory_error"] for member in members):
            group_status, reason = "exclude", "Exact-hash group contains an unreadable image recorded in inventory."
        elif len(family_labels) > 1:
            group_status, reason = "hold", "Exact duplicate paths have conflicting human-reviewed family labels."
        elif "exclude" in statuses:
            group_status, reason = "exclude", "At least one exact-duplicate path was explicitly excluded."
        elif "hold" in statuses:
            group_status, reason = "hold", "At least one exact-duplicate path is held for manual review."
        elif len(folders) > 1 and not (statuses == {"include"} and len(family_labels) == 1):
            group_status, reason = "hold", "Exact bytes occur in multiple source folders; every path must be explicitly included with one consistent family label."
        elif members[0]["review_status"] != "include":
            group_status, reason = "pending", "The deterministic canonical path must be explicitly included before this hash group can enter a split."
        elif not family_labels:
            group_status, reason = "pending", "No explicit human family approval exists for this hash group."
        elif len(family_labels) == 1:
            group_status = "include"
            reason = "Human-reviewed hash group; only its deterministic canonical path can enter a split."
        for member in members:
            member["group_status"] = group_status
            if reason:
                member["review_exclusion_reason"] = reason
            if group_status == "include" and member is not members[0]:
                member["group_status"] = "duplicate_copy"
                member["review_exclusion_reason"] = f"Exact duplicate copy; canonical path is {members[0]['source_relative_path']}. It is not a separate training row."

    rows_by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            rows_by_hash[row["sha256"]].append(row)
    for members in rows_by_hash.values():
        members.sort(key=lambda row: row["source_relative_path"])
    eligible_groups = {
        digest: (family, rows_by_hash[digest][0])
        for digest, (family, _) in approved_hash_groups(inventory, decisions).items()
    }

    approved_by_family = {
        family: sorted(
            ((digest, canonical) for digest, (label, canonical) in eligible_groups.items() if label == family),
            key=lambda item: item[1]["source_relative_path"],
        )
        for family in sorted(FAMILIES)
    }
    counts = {family: len(groups) for family, groups in approved_by_family.items()}
    enough = all(counts[family] >= 3 for family in FAMILIES)
    if enough:
        rng = random.Random(seed)
        for family in sorted(FAMILIES):
            groups = approved_by_family[family][:]
            rng.shuffle(groups)
            train_count, validation_count, _ = _split_counts(len(groups))
            for index, (_, canonical) in enumerate(groups):
                canonical["split"] = (
                    "train" if index < train_count
                    else "validation" if index < train_count + validation_count
                    else "test"
                )
    else:
        reason = "No family splits assigned: need at least 3 unique approved hash groups for each family; current counts: " + ", ".join(f"{family}={counts[family]}" for family in sorted(FAMILIES)) + "."
        for _, canonical in eligible_groups.values():
            canonical["review_exclusion_reason"] = reason

    output_rows = [{field: row.get(field, "") or "" for field in MANIFEST_FIELDS} for row in rows]
    output_rows.sort(key=lambda row: row["source_relative_path"].casefold())
    _write_csv_new(output_path, MANIFEST_FIELDS, output_rows)
    return output_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory_parser = commands.add_parser("inventory", help="hash and verify all known source folders")
    inventory_parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    inventory_parser.add_argument("--output", type=Path, default=INVENTORY_PATH)
    queue_parser = commands.add_parser("queue", help="make a deterministic sample, blank review CSV, and contact sheets")
    queue_parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    queue_parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    queue_parser.add_argument("--per-folder", type=int, default=40)
    queue_parser.add_argument("--seed", type=int, default=42)
    queue_parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    queue_parser.add_argument("--decisions", type=Path, default=DECISIONS_PATH)
    queue_parser.add_argument("--sheets-dir", type=Path, default=SHEETS_DIR)
    prepare_parser = commands.add_parser("prepare", help="validate explicit human decisions and create a family manifest")
    prepare_parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    prepare_parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    prepare_parser.add_argument("--review-csv", type=Path, default=DECISIONS_PATH)
    prepare_parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    prepare_parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    prepare_parser.add_argument("--seed", type=int, default=42)
    readiness_parser = commands.add_parser(
        "readiness", help="count approved distinct hash groups and check split feasibility"
    )
    readiness_parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    readiness_parser.add_argument("--review-csv", type=Path, default=DECISIONS_PATH)
    readiness_parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    merge_parser = commands.add_parser(
        "merge-review-batch", help="merge explicit batch decisions into a new, separate CSV"
    )
    merge_parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    merge_parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    merge_parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    merge_parser.add_argument("--review-csv", type=Path, default=DECISIONS_PATH)
    merge_parser.add_argument("--batch-queue", type=Path, required=True)
    merge_parser.add_argument("--batch-decisions", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, default=MERGED_DECISIONS_PATH)
    merge_parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            rows = write_inventory(args.dataset_root, args.output)
            print(f"Inventory created: {args.output} ({len(rows)} paths; no family labels assigned).")
            for folder in SOURCE_FOLDERS:
                count = sum(row["source_folder"] == folder for row in rows)
                print(f"  {folder}: {count}")
            unreadable = sum(bool(row["inventory_error"]) for row in rows)
            print(f"Unreadable or invalid images recorded: {unreadable}")
        elif args.command == "queue":
            rows = create_review_queue(args.inventory, args.dataset_root, per_folder=args.per_folder, seed=args.seed, queue_path=args.queue, decisions_path=args.decisions, sheets_dir=args.sheets_dir)
            print(f"Review queue: {args.queue} ({len(rows)} paths)")
            print(f"Blank decision template: {args.decisions}")
            print(f"Contact-sheet PDFs: {args.sheets_dir}")
        elif args.command == "readiness":
            inventory = _load_inventory(args.inventory)
            queue = _load_queue(args.queue, inventory)
            decisions = _load_decisions(args.review_csv, inventory, queue)
            counts = family_readiness_counts(inventory, decisions)
            for family in sorted(FAMILIES):
                print(f"{family} distinct approved hash groups: {counts[family]}")
            enough = all(counts[family] >= 3 for family in FAMILIES)
            print("Three-way split possible (at least 3 groups per family): " + ("yes" if enough else "no"))
            if enough:
                for family in sorted(FAMILIES):
                    train, validation, test = _split_counts(counts[family])
                    print(f"  {family} 70/15/15 allocation: train={train}, validation={validation}, test={test}")
            else:
                print("Need at least 3 distinct approved hash groups for each family before preparation can assign all three splits.")
            print("This minimum only populates each partition; it does not establish statistically reliable evaluation.")
            print("Counts use saved inventory SHA-256 values. The prepare command rechecks current source bytes and readability before writing a manifest.")
        elif args.command == "merge-review-batch":
            batch_rows = _read_csv(args.batch_decisions, DECISION_FIELDS)
            rows = merge_review_batch_decisions(
                args.inventory, args.queue, args.review_csv,
                args.batch_queue, args.batch_decisions, args.dataset_root,
                args.output, confirmed=args.confirm,
            )
            changed = sum(row["review_status"].strip().lower() != "pending" for row in batch_rows)
            print(f"Merged decisions written to new file: {args.output}")
            print(f"Batch paths in merged queue: {changed}")
            print("Original queue and decision CSV were not modified.")
        else:
            rows = prepare_family_manifest(args.inventory, args.review_csv, args.dataset_root, args.output, queue_path=args.queue, seed=args.seed)
            print(f"Family manifest: {args.output}")
            for family in sorted(FAMILIES):
                print(f"{family} canonical approved rows: {sum(row['reviewed_family'] == family and row['group_status'] == 'include' for row in rows)}")
            for status in sorted(REVIEW_STATUSES):
                print(f"{status.title()} review rows: {sum(row['review_status'] == status for row in rows)}")
            print(f"Rows with a split: {sum(bool(row['split']) for row in rows)}")
            if not any(row["split"] for row in rows):
                print("Insufficient approved groups for a train/validation/test split; review more images from both families.")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
