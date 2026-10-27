"""Create a small, human-labeled review batch from held-out test images.

Source folders are used only to balance sampling and are shown as provenance.
They are never written as expected or reviewed labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset"
)
DEFAULT_FAMILY_QUEUE = PROJECT_ROOT / "data" / "processed" / "family_review_queue.csv"
DEFAULT_FAMILY_DECISIONS = PROJECT_ROOT / "data" / "processed" / "family_review_decisions.csv"
DEFAULT_MSW_QUEUE = PROJECT_ROOT / "data" / "processed" / "family_msw_review_batch_queue.csv"
DEFAULT_MSW_DECISIONS = PROJECT_ROOT / "data" / "processed" / "family_msw_review_batch_decisions.csv"
DEFAULT_CATEGORY_DECISIONS = PROJECT_ROOT / "data" / "processed" / "paper_cardboard_pilot_review.csv"
DEFAULT_TEST_BATCH_DECISIONS = PROJECT_ROOT / "data" / "processed" / "reviewed_test_review_decisions.csv"
DEFAULT_FAMILY_MANIFEST = PROJECT_ROOT / "data" / "processed" / "family_provisional_split_manifest.csv"
DEFAULT_CATEGORY_MANIFEST = PROJECT_ROOT / "models" / "waste_category_provisional" / "split_manifest.csv"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "data" / "processed" / "reviewed_test_review_batch.csv"
DEFAULT_SHEET_OUTPUT = PROJECT_ROOT / "test-data" / "reviewed_test_review_batch.jpg"
SUPPORTED_FOLDERS = ("battery", "cardboard", "glass", "organic", "paper", "plastic")
CSV_FIELDS = (
    "source_relative_path", "sha256", "source_folder", "split", "selection_source",
    "review_status", "reviewed_family", "decision", "reviewed_label", "reason",
)
SEED = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_valid(value: str) -> bool:
    value = (value or "").lower()
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _resolve_image(dataset_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source-relative path: {relative_path!r}")
    root = dataset_root.resolve(strict=True)
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Source path resolves outside the dataset root: {relative_path!r}") from error
    if not resolved.is_file():
        raise ValueError(f"Source path is not a file: {relative_path!r}")
    return resolved


def select_candidates(
    dataset_root: Path,
    family_queue: list[dict[str, str]],
    family_decisions: list[dict[str, str]],
    msw_queue: list[dict[str, str]],
    msw_decisions: list[dict[str, str]],
    category_decisions: list[dict[str, str]],
    family_manifest: list[dict[str, str]],
    category_manifest: list[dict[str, str]],
    *,
    per_folder: int = 5,
    seed: int = SEED,
    test_batch_decisions: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Select reproducible, readable canonical test images with blank decisions."""
    if per_folder < 1:
        raise ValueError("per_folder must be at least 1")

    manifests = (family_manifest, category_manifest)
    all_rows_by_path: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    train_validation_hashes = set()
    test_paths_by_hash: dict[str, set[str]] = defaultdict(set)
    for task, manifest in (("family", family_manifest), ("category", category_manifest)):
        for row in manifest:
            path = row.get("source_relative_path", "")
            all_rows_by_path[path].append((task, row))
            digest = (row.get("sha256") or "").lower()
            if row.get("split") in {"train", "validation"} and _sha256_valid(digest):
                train_validation_hashes.add(digest)
            if (
                row.get("split") == "test"
                and row.get("canonical_path") == path
                and "include" in (row.get("group_status") or "").lower()
                and _sha256_valid(digest)
            ):
                test_paths_by_hash[digest].add(path)
    conflicting_hashes = {
        digest for digest, paths in test_paths_by_hash.items() if len(paths) > 1
    }

    decisions_by_path: dict[str, list[str]] = defaultdict(list)
    for row in family_decisions:
        decisions_by_path[row.get("source_relative_path", "")].append(
            (row.get("review_status") or "").strip().lower()
        )
    for row in msw_decisions:
        decisions_by_path[row.get("source_relative_path", "")].append(
            (row.get("review_status") or "").strip().lower()
        )
    for row in category_decisions:
        decisions_by_path[row.get("source_relative_path", "")].append(
            (row.get("decision") or "").strip().lower()
        )
    for row in test_batch_decisions or []:
        decisions_by_path[row.get("source_relative_path", "")].append(
            (row.get("decision") or "").strip().lower()
        )

    queue_paths: dict[str, str] = {}
    for source_name, rows in (("family review queue", family_queue), ("MSW review queue", msw_queue)):
        for row in rows:
            path = (row.get("source_relative_path") or "").strip()
            if path:
                queue_paths.setdefault(path, source_name)

    eligible_by_folder: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    exclusions: Counter[str] = Counter()
    for path, tagged_rows in all_rows_by_path.items():
        folder = path.replace("\\", "/").split("/", 1)[0]
        if folder not in SUPPORTED_FOLDERS:
            continue
        existing_statuses = decisions_by_path.get(path, [])
        if any(status in {"include", "hold", "exclude"} for status in existing_statuses):
            exclusions["already has an explicit human decision"] += 1
            continue
        family_rows = [row for task, row in tagged_rows if task == "family"]
        category_rows = [row for task, row in tagged_rows if task == "category"]
        # Every selected item must be canonical and in the test partition for both
        # models, so neither result can reuse a train/validation image.
        if len(family_rows) != 1 or len(category_rows) != 1:
            exclusions["missing or ambiguous model-manifest match"] += 1
            continue
        family_row, category_row = family_rows[0], category_rows[0]
        if family_row.get("source_folder") != folder or category_row.get("source_folder") != folder:
            exclusions["source-folder metadata does not match path"] += 1
            continue
        if family_row.get("split") != "test" or category_row.get("split") != "test":
            exclusions["not held out in both model manifests"] += 1
            continue
        if family_row.get("canonical_path") != path or category_row.get("canonical_path") != path:
            exclusions["not canonical in both model manifests"] += 1
            continue
        if any("include" not in (row.get("group_status") or "").lower() for row in (family_row, category_row)):
            exclusions["duplicate group or manifest row excluded"] += 1
            continue
        family_hash = (family_row.get("sha256") or "").lower()
        category_hash = (category_row.get("sha256") or "").lower()
        if not _sha256_valid(family_hash) or family_hash != category_hash:
            exclusions["manifest hash missing or conflicting"] += 1
            continue
        if family_hash in train_validation_hashes:
            exclusions["hash appears in training or validation data"] += 1
            continue
        if family_hash in conflicting_hashes:
            exclusions["hash is shared by multiple canonical test paths"] += 1
            continue
        queued_hashes = {
            (row.get("sha256") or "").lower()
            for rows in (family_queue, msw_queue)
            for row in rows
            if row.get("source_relative_path") == path and _sha256_valid(row.get("sha256", ""))
        }
        if queued_hashes and queued_hashes != {family_hash}:
            exclusions["review-queue hash does not match test manifests"] += 1
            continue
        eligible_by_folder[folder][path] = {
            "source_relative_path": path,
            "sha256": family_hash,
            "source_folder": folder,
            "split": "test",
            "selection_source": queue_paths.get(path, "held-out test manifest supplement"),
            "review_status": "",
            "reviewed_family": "",
            "decision": "",
            "reviewed_label": "",
            "reason": "",
            "_image_path": "",
            "_expected_hash": family_hash,
        }

    selected = []
    randomizer = random.Random(seed)
    for folder in SUPPORTED_FOLDERS:
        candidates = list(eligible_by_folder.get(folder, {}).values())
        queued = [row for row in candidates if row["selection_source"] != "held-out test manifest supplement"]
        supplements = [row for row in candidates if row["selection_source"] == "held-out test manifest supplement"]
        for choices in (queued, supplements):
            choices.sort(key=lambda row: row["source_relative_path"])
            randomizer.shuffle(choices)
        chosen = []
        for row in queued + supplements:
            if len(chosen) >= per_folder:
                break
            try:
                image_path = _resolve_image(dataset_root, row["source_relative_path"])
                current_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if current_hash != row.pop("_expected_hash"):
                    exclusions["current source hash differs from test manifests"] += 1
                    continue
                with Image.open(image_path) as image:
                    image.load()
                row["_image_path"] = str(image_path)
                chosen.append(row)
            except (OSError, ValueError, UnidentifiedImageError):
                exclusions["source image missing, unsafe, or unreadable"] += 1
        selected.extend(chosen)
    return selected, dict(exclusions)


def _load_font(size: int):
    for font_path in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ):
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_contact_sheet(rows: list[dict[str, str]], *, columns: int = 5) -> Image.Image:
    """Render reviewed images with paths and provenance, never suggested labels."""
    if not rows:
        raise ValueError("No eligible images were selected for the contact sheet.")
    tile_width, image_height, caption_height = 260, 170, 92
    tile_height = image_height + caption_height
    row_count = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, row_count * tile_height + 26), "white")
    draw = ImageDraw.Draw(sheet)
    font = _load_font(15)
    small_font = _load_font(12)
    draw.text((12, 8), "Human review batch — source folder is provenance only, not a label", fill="black", font=font)
    for index, row in enumerate(rows):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height + 26
        with Image.open(row["_image_path"]) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 16, image_height - 10))
            sheet.paste(image, (x + (tile_width - image.width) // 2, y + (image_height - image.height) // 2))
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline="#777777")
        path_text = row["source_relative_path"]
        wrapped = textwrap.wrap(path_text, width=38, break_long_words=True, break_on_hyphens=False)
        caption = [f"{index + 1}. Folder: {row['source_folder']} (provenance only)", *wrapped]
        for line_number, line in enumerate(caption[:6]):
            draw.text((x + 7, y + image_height + line_number * 14), line, fill="black", font=small_font)
    return sheet


def write_batch(rows: list[dict[str, str]], csv_path: Path, sheet_path: Path) -> None:
    """Create both outputs exclusively; never overwrite an earlier review batch."""
    if csv_path.exists() or sheet_path.exists():
        raise FileExistsError("Review batch output already exists; choose new output paths.")
    if not csv_path.parent.is_dir() or not sheet_path.parent.is_dir():
        raise FileNotFoundError("Both output parent directories must already exist.")
    sheet = render_contact_sheet(rows)
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    try:
        with sheet_path.open("xb") as handle:
            sheet.save(handle, format="JPEG", quality=92)
    except Exception:
        # Don't leave a CSV that looks like a complete batch if sheet writing failed.
        csv_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--per-folder", type=int, default=5, help="Maximum samples from each supported source folder (default: 5).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--sheet-output", type=Path, default=DEFAULT_SHEET_OUTPUT)
    parser.add_argument("--family-queue", type=Path, default=DEFAULT_FAMILY_QUEUE)
    parser.add_argument("--family-decisions", type=Path, default=DEFAULT_FAMILY_DECISIONS)
    parser.add_argument("--msw-queue", type=Path, default=DEFAULT_MSW_QUEUE)
    parser.add_argument("--msw-decisions", type=Path, default=DEFAULT_MSW_DECISIONS)
    parser.add_argument("--category-decisions", type=Path, default=DEFAULT_CATEGORY_DECISIONS)
    parser.add_argument("--test-batch-decisions", type=Path, default=DEFAULT_TEST_BATCH_DECISIONS,
                        help="Previously saved Pilot Test Review decisions to exclude from this new batch.")
    parser.add_argument("--additional-test-batch-decisions", type=Path, action="append", default=[],
                        help="Another saved Pilot Test Review decisions CSV to exclude; may be repeated.")
    parser.add_argument("--family-manifest", type=Path, default=DEFAULT_FAMILY_MANIFEST)
    parser.add_argument("--category-manifest", type=Path, default=DEFAULT_CATEGORY_MANIFEST)
    args = parser.parse_args(argv)
    if not args.dataset_root.is_dir():
        parser.error(f"Dataset root not found: {args.dataset_root}")
    for output in (args.csv_output, args.sheet_output):
        if output.exists():
            parser.error(f"Refusing to overwrite existing output: {output}")
    prior_test_decisions = (
        read_csv(args.test_batch_decisions) if args.test_batch_decisions.is_file() else []
    )
    for decisions_path in args.additional_test_batch_decisions:
        prior_test_decisions.extend(read_csv(decisions_path))
    selected, exclusions = select_candidates(
        args.dataset_root,
        read_csv(args.family_queue), read_csv(args.family_decisions),
        read_csv(args.msw_queue), read_csv(args.msw_decisions),
        read_csv(args.category_decisions), read_csv(args.family_manifest),
        read_csv(args.category_manifest), per_folder=args.per_folder, seed=args.seed,
        test_batch_decisions=prior_test_decisions,
    )
    if not selected:
        parser.error("No readable, canonical held-out test images were eligible for review.")
    write_batch(selected, args.csv_output, args.sheet_output)
    print(f"Selected {len(selected)} images; selection is balanced by source folder for sampling only.")
    for folder, count in sorted(Counter(row['source_folder'] for row in selected).items()):
        print(f"  {folder}: {count}")
    print("Human decision fields are blank; source folders are not labels.")
    print(f"Review CSV: {args.csv_output.resolve()}")
    print(f"Contact sheet: {args.sheet_output.resolve()}")
    if exclusions:
        print("Candidate exclusion counts:")
        for reason, count in sorted(exclusions.items()):
            print(f"  {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
