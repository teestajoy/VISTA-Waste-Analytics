
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "Waste_classification_dataset"
    / "final_merged_dataset"
)
OUTPUT_PATH = Path("test-data") / "dataset_samples.jpg"

CATEGORIES = ["metal", "trash", "textile"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLES_PER_CATEGORY = 5

THUMB_WIDTH = 220
THUMB_HEIGHT = 180
LABEL_HEIGHT = 32
PADDING = 12

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

columns = SAMPLES_PER_CATEGORY
rows = len(CATEGORIES)

sheet = Image.new(
    "RGB",
    (
        columns * (THUMB_WIDTH + PADDING) + PADDING,
        rows * (THUMB_HEIGHT + LABEL_HEIGHT + PADDING) + PADDING,
    ),
    "white",
)
draw = ImageDraw.Draw(sheet)

for row, category in enumerate(CATEGORIES):
    category_dir = DATASET_DIR / category
    if not category_dir.is_dir():
        print(f"Skipping missing category folder: {category_dir}")
        continue

    images = sorted(
        path
        for path in category_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for column, path in enumerate(images[:SAMPLES_PER_CATEGORY]):
        try:
            with Image.open(path) as source:
                thumbnail = ImageOps.contain(
                    source.convert("RGB"),
                    (THUMB_WIDTH, THUMB_HEIGHT),
                )

            x = PADDING + column * (THUMB_WIDTH + PADDING)
            y = PADDING + row * (
                THUMB_HEIGHT + LABEL_HEIGHT + PADDING
            )

            sheet.paste(thumbnail, (x, y))
            draw.text(
                (x, y + THUMB_HEIGHT + 4),
                f"{category}: {path.name[:22]}",
                fill="black",
            )

        except (OSError, ValueError) as error:
            print(f"Skipped {path}: {error}")

sheet.save(OUTPUT_PATH, quality=90)
print(f"Saved sample grid to: {OUTPUT_PATH.resolve()}")
