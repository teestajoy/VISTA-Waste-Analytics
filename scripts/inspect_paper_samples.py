import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "Waste_classification_dataset"
    / "final_merged_dataset"
)
CATEGORY = "paper"
CATEGORY_DIR = DATASET_DIR / CATEGORY
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42
SAMPLE_COUNT = 20
COLUMNS = 4
ROWS = 5
THUMB_WIDTH = 220
THUMB_HEIGHT = 180
PADDING = 12
CHARACTERS_PER_FILENAME_LINE = 30
FILENAME_LINE_HEIGHT = 14
CATEGORY_LABEL_HEIGHT = 18


def main() -> None:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")
    if not CATEGORY_DIR.is_dir():
        raise FileNotFoundError(f"Category folder not found: {CATEGORY_DIR}")

    image_paths = sorted(
        path
        for path in CATEGORY_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.Random(RANDOM_SEED).shuffle(image_paths)

    samples: list[tuple[str, Image.Image]] = []
    for path in image_paths:
        try:
            with Image.open(path) as source:
                source.load()
                thumbnail = ImageOps.contain(
                    source.convert("RGB"),
                    (THUMB_WIDTH, THUMB_HEIGHT),
                )
            samples.append((path.name, thumbnail))
            if len(samples) == SAMPLE_COUNT:
                break
        except (OSError, ValueError) as error:
            print(f"Skipped unreadable image {path}: {error}")

    if not samples:
        raise ValueError(f"No readable supported images found in: {CATEGORY_DIR}")

    wrapped_filenames = [
        textwrap.wrap(
            filename,
            width=CHARACTERS_PER_FILENAME_LINE,
            break_long_words=True,
            break_on_hyphens=False,
        )
        or [""]
        for filename, _ in samples
    ]
    filename_lines = max(len(lines) for lines in wrapped_filenames)
    label_height = CATEGORY_LABEL_HEIGHT + filename_lines * FILENAME_LINE_HEIGHT
    tile_height = THUMB_HEIGHT + label_height

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * (THUMB_WIDTH + PADDING) + PADDING,
            ROWS * (tile_height + PADDING) + PADDING,
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, ((filename, thumbnail), filename_lines_for_sample) in enumerate(
        zip(samples, wrapped_filenames)
    ):
        column = index % COLUMNS
        row = index // COLUMNS
        x = PADDING + column * (THUMB_WIDTH + PADDING)
        y = PADDING + row * (tile_height + PADDING)
        image_x = x + (THUMB_WIDTH - thumbnail.width) // 2
        image_y = y + (THUMB_HEIGHT - thumbnail.height) // 2

        sheet.paste(thumbnail, (image_x, image_y))
        label_y = y + THUMB_HEIGHT + 2
        draw.text((x, label_y), CATEGORY, fill="black")
        for line_number, line in enumerate(filename_lines_for_sample):
            draw.text(
                (x, label_y + CATEGORY_LABEL_HEIGHT + line_number * FILENAME_LINE_HEIGHT),
                line,
                fill="black",
            )

    print(f"Displaying {len(samples)} readable {CATEGORY} images.")
    sheet.show(title=f"{CATEGORY} dataset samples")


if __name__ == "__main__":
    main()
