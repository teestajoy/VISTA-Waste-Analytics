import argparse
import hashlib
from collections import defaultdict
from itertools import combinations
from pathlib import Path


DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "Waste_classification_dataset"
    / "final_merged_dataset"
)
CATEGORIES = ["plastic", "paper", "cardboard", "glass", "organic"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file's bytes, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Find exact duplicate image files using SHA-256 hashes."
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="print paths and categories for every exact duplicate group",
    )
    args = parser.parse_args(argv)

    files_by_hash: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    images_checked_by_category = {category: 0 for category in CATEGORIES}
    unreadable_files = 0

    if not DATASET_DIR.is_dir():
        print(f"Dataset folder not found: {DATASET_DIR}")
    else:
        for category in CATEGORIES:
            category_dir = DATASET_DIR / category
            if not category_dir.is_dir():
                print(f"Missing category folder; skipping: {category_dir}")
                continue

            image_paths = sorted(
                path
                for path in category_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for path in image_paths:
                try:
                    file_hash = sha256_file(path)
                except OSError as error:
                    unreadable_files += 1
                    print(f"Could not read image file {path}: {error}")
                    continue

                images_checked_by_category[category] += 1
                files_by_hash[file_hash].append((category, path))

    duplicate_groups = {
        file_hash: files
        for file_hash, files in files_by_hash.items()
        if len(files) >= 2
    }

    duplicate_groups_by_category = {category: 0 for category in CATEGORIES}
    cross_category_groups = 0
    files_in_duplicate_groups = 0
    for files in duplicate_groups.values():
        categories_in_group: dict[str, list[Path]] = defaultdict(list)
        for category, path in files:
            categories_in_group[category].append(path)

        files_in_duplicate_groups += len(files)
        if len(categories_in_group) >= 2:
            cross_category_groups += 1
        for category, paths in categories_in_group.items():
            if len(paths) >= 2:
                duplicate_groups_by_category[category] += 1

    print("\nImages checked per category:")
    for category in CATEGORIES:
        print(f"  {category}: {images_checked_by_category[category]}")
    print(f"Unreadable files: {unreadable_files}")

    print("Duplicate groups per category (at least two files in that category):")
    for category in CATEGORIES:
        print(f"  {category}: {duplicate_groups_by_category[category]}")

    print(
        f"Cross-category duplicate groups: {cross_category_groups}\n"
        f"Total files involved in duplicate groups: {files_in_duplicate_groups}\n"
        f"Unique images after exact duplicates: {len(files_by_hash)}\n"
        f"Total duplicate groups: {len(duplicate_groups)}"
    )

    if args.details:
        if duplicate_groups:
            print("\nExact duplicate image groups:")
            for group_number, (file_hash, files) in enumerate(
                sorted(duplicate_groups.items()), start=1
            ):
                categories_in_group: dict[str, list[Path]] = defaultdict(list)
                for category, path in files:
                    categories_in_group[category].append(path)

                print(f"\nGroup {group_number} (SHA-256: {file_hash})")
                for category, path in files:
                    relative_path = path.relative_to(DATASET_DIR)
                    print(f"  [{category}] {relative_path}")

                for category in sorted(categories_in_group):
                    if len(categories_in_group[category]) >= 2:
                        print(f"  Duplicate within category: {category}")

                for first_category, second_category in combinations(
                    sorted(categories_in_group), 2
                ):
                    print(
                        "  Duplicate across categories: "
                        f"{first_category} and {second_category}"
                    )
        else:
            print("\nNo exact duplicate image files found.")


if __name__ == "__main__":
    main()
