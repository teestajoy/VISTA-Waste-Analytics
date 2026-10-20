
from pathlib import Path
from collections import Counter

DATASET_DIR = Path.home() / "Downloads" / "Waste_classification_dataset" / "final_merged_dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

total_images = 0

print(f"Dataset: {DATASET_DIR}\n")
print(f"{'Category':<15} {'Images':>10}")
print("-" * 27)

for category_dir in sorted(DATASET_DIR.iterdir()):
    if not category_dir.is_dir():
        continue

    image_count = sum(
        1
        for path in category_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"{category_dir.name:<15} {image_count:>10}")
    total_images += image_count

print("-" * 27)
print(f"{'TOTAL':<15} {total_images:>10}")
