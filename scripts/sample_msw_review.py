"""Create a read-only contact sheet of canonical paper/cardboard manifest images."""

import argparse
import csv
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "Waste_classification_dataset"
    / "final_merged_dataset"
)
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "msw_split_manifest.csv"
CONTACT_SHEET_PATHS = {
    "paper": PROJECT_ROOT / "test-data" / "msw_review_sample_paper.jpg",
    "cardboard": PROJECT_ROOT / "test-data" / "msw_review_sample_cardboard.jpg",
}
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "msw_review_sample.csv"
DEFAULT_EXPANDED_SAMPLE_CSV_PATH = (
    PROJECT_ROOT / "data" / "processed" / "paper_cardboard_expanded_review_sample.csv"
)
ORIGINAL_REVIEW_CSV_PATH = (
    PROJECT_ROOT / "data" / "processed" / "paper_cardboard_pilot_review.csv"
)
DEFAULT_PRESERVED_SAMPLE_PATH = SAMPLE_CSV_PATH
DEFAULT_OUTPUT_SAMPLE_CSV_PATH = DEFAULT_EXPANDED_SAMPLE_CSV_PATH
DEFAULT_CONTACT_SHEET_PATHS = {
    "paper": PROJECT_ROOT / "test-data" / "paper_cardboard_expanded_paper.jpg",
    "cardboard": PROJECT_ROOT / "test-data" / "paper_cardboard_expanded_cardboard.jpg",
}
CATEGORIES = ("paper", "cardboard")
SAMPLE_COUNT_PER_CLASS = 60
RANDOM_SEED = 42
COLUMNS = 4
THUMB_WIDTH = 220
THUMB_HEIGHT = 180
PADDING = 12
LABEL_LINE_HEIGHT = 15
LABEL_GAP = 4
LABEL_WRAP_WIDTH = 34
MANIFEST_FIELDS = {
    "source_relative_path",
    "sha256",
    "source_folder",
    "proposed_label",
}
SAMPLE_FIELDS = (
    "source_relative_path",
    "sha256",
    "source_folder",
    "proposed_label",
)


@dataclass(frozen=True)
class ReviewSample:
    source_relative_path: str
    sha256: str
    source_folder: str
    proposed_label: str
    thumbnail: Image.Image


def _safe_source_path(dataset_root: Path, source_relative_path: str) -> Path:
    """Resolve a manifest path while preventing absolute or escaping paths."""
    relative = PurePosixPath(source_relative_path.replace("\\", "/"))
    if (
        not source_relative_path.strip()
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"Invalid source-relative image path: {source_relative_path!r}")

    root = dataset_root.resolve()
    image_path = root.joinpath(*relative.parts).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Manifest image path resolves outside the dataset root: {source_relative_path}"
        ) from error
    return image_path


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_fields = MANIFEST_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Manifest is missing required columns: {missing}")
        return list(reader)


def select_samples(
    manifest_path: Path,
    dataset_root: Path,
    *,
    seed: int = RANDOM_SEED,
    per_class: int = SAMPLE_COUNT_PER_CLASS,
    preserve_sample_path: Path | None = None,
) -> list[ReviewSample]:
    """Select readable canonical images listed in the manifest, without directory scans."""
    if per_class < 1:
        raise ValueError("per_class must be at least 1")

    rows = _load_manifest(manifest_path)
    paths_by_hash: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        source_folder = (row.get("source_folder") or "").strip()
        image_hash = (row.get("sha256") or "").strip()
        source_path = (row.get("source_relative_path") or "").strip()
        if source_folder in CATEGORIES and image_hash and source_path:
            paths_by_hash.setdefault(image_hash, []).append(row)

    conflicting_hashes = {
        image_hash
        for image_hash, hash_rows in paths_by_hash.items()
        if {row.get("source_folder", "").strip() for row in hash_rows}
        >= set(CATEGORIES)
    }
    for image_hash in sorted(conflicting_hashes):
        paths = sorted(
            (row.get("source_relative_path") or "").strip()
            for row in paths_by_hash[image_hash]
            if (row.get("source_folder") or "").strip() in CATEGORIES
        )
        print(
            f"Warning: excluding cross-category SHA-256 {image_hash}; "
            f"it appears in both paper and cardboard: {', '.join(paths)}"
        )

    # Match prepare_msw_dataset.py: within the scoped hash group, the lexically
    # smallest source-relative path is canonical. Unrelated manifest categories
    # do not participate. Cross-category and explicitly excluded groups are skipped.
    excluded_hashes = {
        image_hash
        for image_hash, hash_rows in paths_by_hash.items()
        if any((row.get("review_status") or "").strip() == "excluded" for row in hash_rows)
    }
    canonical_path_by_hash = {
        image_hash: min(
            (row["source_relative_path"].strip() for row in hash_rows),
        )
        for image_hash, hash_rows in paths_by_hash.items()
        if image_hash not in conflicting_hashes and image_hash not in excluded_hashes
    }

    preserved_rows: list[dict[str, str]] = []
    preserved_paths: set[str] = set()
    preserved_hashes: set[str] = set()
    if preserve_sample_path is not None:
        if not preserve_sample_path.is_file():
            raise FileNotFoundError(
                f"Existing sample to preserve was not found: {preserve_sample_path}"
            )
        preserved_rows = _load_manifest(preserve_sample_path)
        for row in preserved_rows:
            source_path = (row.get("source_relative_path") or "").strip()
            image_hash = (row.get("sha256") or "").strip()
            category = (row.get("source_folder") or "").strip()
            proposed_label = (row.get("proposed_label") or "").strip()
            if category not in CATEGORIES or proposed_label != category:
                raise ValueError(
                    f"Existing sample has an unsupported category for {source_path!r}."
                )
            if source_path in preserved_paths:
                raise ValueError(f"Existing sample repeats image path {source_path!r}.")
            matching_rows = [
                candidate
                for candidate in paths_by_hash.get(image_hash, [])
                if (candidate.get("source_relative_path") or "").strip() == source_path
                and (candidate.get("source_folder") or "").strip() == category
                and (candidate.get("proposed_label") or "").strip() == category
            ]
            if not matching_rows:
                raise ValueError(
                    f"Existing sample path/hash is absent from the current manifest: "
                    f"{source_path!r}."
                )
            if image_hash in conflicting_hashes:
                raise ValueError(
                    f"Existing sample path is now in a cross-category duplicate group: "
                    f"{source_path!r}."
                )
            if image_hash in excluded_hashes:
                raise ValueError(
                    f"Existing sample path is now in an explicitly excluded group: "
                    f"{source_path!r}."
                )
            if canonical_path_by_hash.get(image_hash) != source_path:
                raise ValueError(
                    f"Existing sample path is not the canonical path for its exact "
                    f"duplicate group: {source_path!r}."
                )
            if image_hash in preserved_hashes:
                raise ValueError(
                    f"Existing sample contains duplicate image bytes at {source_path!r}."
                )
            preserved_paths.add(source_path)
            preserved_hashes.add(image_hash)

    samples: list[ReviewSample] = []
    for category_index, category in enumerate(CATEGORIES):
        existing_for_category = [
            row
            for row in preserved_rows
            if (row.get("source_folder") or "").strip() == category
        ]
        if len(existing_for_category) > per_class:
            raise ValueError(
                f"Cannot preserve {len(existing_for_category)} existing {category} "
                f"paths when per_class is {per_class}."
            )
        candidates = [
            row
            for row in rows
            if (row.get("source_folder") or "").strip() == category
            and (row.get("proposed_label") or "").strip() == category
            and (row.get("sha256") or "").strip()
            and (row.get("sha256") or "").strip() not in conflicting_hashes
            and (row.get("sha256") or "").strip() not in excluded_hashes
            and (row.get("source_relative_path") or "").strip()
            == canonical_path_by_hash.get((row.get("sha256") or "").strip())
            and (row.get("source_relative_path") or "").strip()
            not in preserved_paths
            and (row.get("sha256") or "").strip() not in preserved_hashes
        ]
        candidates.sort(key=lambda row: row["source_relative_path"])
        random.Random(seed + category_index).shuffle(candidates)

        class_samples: list[ReviewSample] = []
        ordered_rows = existing_for_category + candidates
        for row in ordered_rows:
            if len(class_samples) >= per_class:
                break
            relative_path = row["source_relative_path"].strip()
            try:
                image_path = _safe_source_path(dataset_root, relative_path)
                with Image.open(image_path) as source:
                    source.load()
                    thumbnail = ImageOps.contain(
                        source.convert("RGB"), (THUMB_WIDTH, THUMB_HEIGHT)
                    )
            except (OSError, ValueError) as error:
                print(f"Skipped unreadable manifest image {relative_path}: {error}")
                continue

            class_samples.append(
                ReviewSample(
                    source_relative_path=relative_path,
                    sha256=row["sha256"].strip(),
                    source_folder=category,
                    proposed_label=category,
                    thumbnail=thumbnail,
                )
            )

        samples.extend(class_samples)
        if len(class_samples) < per_class:
            print(
                f"Only {len(class_samples)} readable canonical {category} images "
                f"were available; requested up to {per_class}."
            )

    if not samples:
        raise ValueError("The manifest contains no readable canonical paper/cardboard images.")
    return samples


def save_sample_csv(output_path: Path, samples: list[ReviewSample]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as sample_file:
        writer = csv.DictWriter(sample_file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "source_relative_path": sample.source_relative_path,
                    "sha256": sample.sha256,
                    "source_folder": sample.source_folder,
                    "proposed_label": sample.proposed_label,
                }
            )


def save_contact_sheet(output_path: Path, samples: list[ReviewSample]) -> None:
    if not samples:
        raise ValueError("Cannot create a contact sheet without samples.")

    wrapped_labels = [
        textwrap.wrap(
            f"{sample.proposed_label}: {sample.source_relative_path}",
            width=LABEL_WRAP_WIDTH,
            break_long_words=True,
            break_on_hyphens=False,
        )
        or [""]
        for sample in samples
    ]
    label_lines = max(len(lines) for lines in wrapped_labels)
    tile_height = THUMB_HEIGHT + LABEL_GAP + label_lines * LABEL_LINE_HEIGHT
    row_count = (len(samples) + COLUMNS - 1) // COLUMNS
    sheet = Image.new(
        "RGB",
        (
            COLUMNS * (THUMB_WIDTH + PADDING) + PADDING,
            row_count * (tile_height + PADDING) + PADDING,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, (sample, lines) in enumerate(zip(samples, wrapped_labels)):
        column = index % COLUMNS
        row = index // COLUMNS
        x = PADDING + column * (THUMB_WIDTH + PADDING)
        y = PADDING + row * (tile_height + PADDING)
        image_x = x + (THUMB_WIDTH - sample.thumbnail.width) // 2
        image_y = y + (THUMB_HEIGHT - sample.thumbnail.height) // 2
        sheet.paste(sample.thumbnail, (image_x, image_y))
        for line_number, line in enumerate(lines):
            draw.text(
                (x, y + THUMB_HEIGHT + LABEL_GAP + line_number * LABEL_LINE_HEIGHT),
                line,
                fill="black",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as sheet_file:
        sheet.save(sheet_file, format="JPEG")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--per-class", type=int, default=SAMPLE_COUNT_PER_CLASS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--preserve-sample", type=Path, default=DEFAULT_PRESERVED_SAMPLE_PATH)
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_OUTPUT_SAMPLE_CSV_PATH)
    parser.add_argument("--paper-sheet", type=Path, default=DEFAULT_CONTACT_SHEET_PATHS["paper"])
    parser.add_argument("--cardboard-sheet", type=Path, default=DEFAULT_CONTACT_SHEET_PATHS["cardboard"])
    args = parser.parse_args(argv)
    if args.per_class < 1:
        parser.error("--per-class must be at least 1")

    output_paths = {
        args.sample_csv.resolve(),
        args.paper_sheet.resolve(),
        args.cardboard_sheet.resolve(),
    }
    protected_paths = {
        args.manifest.resolve(),
        args.preserve_sample.resolve(),
        ORIGINAL_REVIEW_CSV_PATH.resolve(),
        SAMPLE_CSV_PATH.resolve(),
    }
    if len(output_paths) != 3:
        parser.error("--sample-csv, --paper-sheet, and --cardboard-sheet must be distinct paths")
    protected_overlap = output_paths & protected_paths
    if protected_overlap:
        parser.error(
            "Refusing to overwrite an existing manifest, original sample, or review file: "
            + ", ".join(str(path) for path in sorted(protected_overlap))
        )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        parser.error(
            "Refusing to overwrite existing expanded output(s): "
            + ", ".join(str(path) for path in sorted(existing_outputs))
        )

    samples = select_samples(
        args.manifest,
        args.dataset_root,
        seed=args.seed,
        per_class=args.per_class,
        preserve_sample_path=args.preserve_sample,
    )
    save_sample_csv(args.sample_csv, samples)
    contact_sheet_paths = {
        "paper": args.paper_sheet,
        "cardboard": args.cardboard_sheet,
    }
    for category in CATEGORIES:
        category_samples = [
            sample for sample in samples if sample.proposed_label == category
        ]
        count = len(category_samples)
        print(f"Selected {count} {category} images for reference review.")
        if category_samples:
            save_contact_sheet(contact_sheet_paths[category], category_samples)
            print(f"{category.capitalize()} contact sheet: {contact_sheet_paths[category]}")
    print(f"Reference sample CSV written to: {args.sample_csv}")
    print("These files are for visual review only; no review decisions were recorded.")


if __name__ == "__main__":
    main()
