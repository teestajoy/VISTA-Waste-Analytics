"""Render the existing paper/cardboard review sample as readable multi-page PDFs."""

import csv
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "Waste_classification_dataset"
    / "final_merged_dataset"
)
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "msw_review_sample.csv"
OUTPUT_PATHS = {
    "paper": PROJECT_ROOT / "test-data" / "msw_review_sample_paper_detailed.pdf",
    "cardboard": PROJECT_ROOT / "test-data" / "msw_review_sample_cardboard_detailed.pdf",
}
CATEGORIES = ("paper", "cardboard")
REQUIRED_FIELDS = {"source_relative_path", "proposed_label"}
PAGE_WIDTH = 1800
PAGE_HEIGHT = 1400
MARGIN = 55
GAP = 40
COLUMNS = 2
IMAGE_AREA_HEIGHT = 470
CAPTION_AREA_HEIGHT = 145
CAPTION_GAP = 12
PATH_FONT_SIZE = 24
CATEGORY_FONT_SIZE = 30
LINE_SPACING = 7
MIN_PATH_FONT_SIZE = 12


@dataclass(frozen=True)
class ReviewImage:
    source_relative_path: str
    proposed_label: str
    image: Image.Image


def _safe_source_path(dataset_root: Path, source_relative_path: str) -> Path:
    """Resolve a CSV path without allowing it to escape the MWCD root."""
    normalized = source_relative_path.replace("\\", "/")
    relative = PurePosixPath(normalized)
    windows_path = PureWindowsPath(source_relative_path)
    if (
        not source_relative_path.strip()
        or relative.is_absolute()
        or windows_path.drive
        or windows_path.root
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ValueError(f"Invalid source-relative image path: {source_relative_path!r}")

    root = dataset_root.resolve()
    image_path = root.joinpath(*relative.parts).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Image path resolves outside the dataset root: {source_relative_path}"
        ) from error
    return image_path


def load_review_images(
    sample_csv: Path, dataset_root: Path, category: str
) -> list[ReviewImage]:
    """Load only rows for the requested category from the supplied sample CSV."""
    if category not in CATEGORIES:
        raise ValueError(f"Unsupported review category: {category}")
    if not sample_csv.is_file():
        raise FileNotFoundError(f"Review sample CSV not found: {sample_csv}")

    loaded: list[ReviewImage] = []
    with sample_csv.open("r", encoding="utf-8-sig", newline="") as sample_file:
        reader = csv.DictReader(sample_file)
        missing_fields = REQUIRED_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Review sample CSV is missing required columns: {missing}")

        for line_number, row in enumerate(reader, start=2):
            proposed_label = (row.get("proposed_label") or "").strip()
            source_relative_path = (row.get("source_relative_path") or "").strip()
            if proposed_label != category:
                continue
            # Unsafe paths are a malformed manifest, not unreadable images.
            image_path = _safe_source_path(dataset_root, source_relative_path)
            try:
                with Image.open(image_path) as source:
                    source.load()
                    image = source.convert("RGB")
            except OSError as error:
                print(
                    f"Skipped unreadable sample on CSV line {line_number} "
                    f"({source_relative_path!r}): {error}"
                )
                continue

            loaded.append(
                ReviewImage(
                    source_relative_path=source_relative_path,
                    proposed_label=proposed_label,
                    image=image,
                )
            )
    return loaded


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows_directory = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = [
        windows_directory / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf"),
        windows_directory / "Fonts" / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
        Path("/Library/Fonts") / ("Arial Bold.ttf" if bold else "Arial.ttf"),
        Path(ImageFont.__file__).parent / "fonts" / "DejaVuSans.ttf",
    ]
    for font_path in candidates:
        if font_path.is_file():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue

    # Pillow's bitmap fallback is readable at its native size. Recent Pillow
    # versions can scale it; older versions retain a smaller but safe fallback.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow versions before the size argument was added.
        return ImageFont.load_default()


def _text_width(text: str, font: ImageFont.ImageFont) -> float:
    """Measure rendered text width using the selected font's glyph metrics."""
    return float(font.getlength(text))


def _max_line_height(lines: list[str], font: ImageFont.ImageFont) -> int:
    """Return the tallest rendered glyph bounds among the given lines."""
    return max(
        (font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines),
        default=0,
    )


def _validate_caption_layout(
    category_lines: list[str],
    category_font: ImageFont.ImageFont,
    path_lines: list[str],
    path_font: ImageFont.ImageFont,
    tile_width: int,
    caption_height: int,
) -> None:
    """Verify rendered caption widths and vertical space before drawing."""
    for caption_name, lines, font in (
        ("category", category_lines, category_font),
        ("source path", path_lines, path_font),
    ):
        for line in lines:
            rendered_width = _text_width(line, font)
            if rendered_width > tile_width:
                raise ValueError(
                    f"The {caption_name} caption line is {rendered_width:.1f}px wide, "
                    f"exceeding its {tile_width}px area."
                )

    positioned_lines, required_height = _caption_line_positions(
        category_lines, category_font, path_lines, path_font
    )
    for line, font, draw_y in positioned_lines:
        bbox = font.getbbox(line)
        rendered_top = draw_y + bbox[1]
        rendered_bottom = draw_y + bbox[3]
        if (
            rendered_top < 0
            or rendered_top > caption_height
            or rendered_bottom < rendered_top
            or rendered_bottom > caption_height
            or rendered_bottom + LINE_SPACING > caption_height
        ):
            raise ValueError(
                f"Caption line {line!r} with rendered vertical extent "
                f"{rendered_top}px to {rendered_bottom}px (including "
                f"{LINE_SPACING}px line spacing) exceeds the allocated "
                f"{caption_height}px caption area; {required_height}px is required."
            )


def _caption_line_positions(
    category_lines: list[str],
    category_font: ImageFont.ImageFont,
    path_lines: list[str],
    path_font: ImageFont.ImageFont,
) -> tuple[list[tuple[str, ImageFont.ImageFont, int]], int]:
    """Return draw positions and required height using each line's font bounds."""
    category_height = _max_line_height(category_lines, category_font)
    path_height = _max_line_height(path_lines, path_font)
    positioned_lines: list[tuple[str, ImageFont.ImageFont, int]] = []

    for line_number, line in enumerate(category_lines):
        text_y = line_number * (category_height + LINE_SPACING)
        bbox = category_font.getbbox(line)
        positioned_lines.append((line, category_font, text_y - bbox[1]))

    path_start = len(category_lines) * (category_height + LINE_SPACING) + LINE_SPACING
    for line_number, line in enumerate(path_lines):
        text_y = path_start + line_number * (path_height + LINE_SPACING)
        bbox = path_font.getbbox(line)
        positioned_lines.append((line, path_font, text_y - bbox[1]))

    required_height = max(
        (
            max(
                draw_y + font.getbbox(line)[1],
                draw_y + font.getbbox(line)[3] + LINE_SPACING,
            )
            for line, font, draw_y in positioned_lines
        ),
        default=0,
    )
    return positioned_lines, required_height


def _wrap_to_pixel_width(
    text: str, font: ImageFont.ImageFont, max_width: int
) -> list[str]:
    """Wrap text by rendered pixel width without dropping any path characters."""
    if max_width <= 0:
        raise ValueError("max_width must be greater than zero")

    lines: list[str] = []
    current = ""
    for character in text:
        if _text_width(character, font) > max_width:
            raise ValueError("max_width is narrower than a rendered character")
        candidate = current + character
        if current and _text_width(candidate, font) > max_width:
            # Move the last space to the next visual line instead of deleting it.
            # Joining the output lines therefore reconstructs the exact input.
            last_space = current.rfind(" ")
            if last_space > 0:
                lines.append(current[:last_space])
                current = current[last_space:] + character
            else:
                lines.append(current)
                current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _create_page(images: list[ReviewImage]) -> Image.Image:
    category_font = _font(CATEGORY_FONT_SIZE, bold=True)
    tile_width = (PAGE_WIDTH - 2 * MARGIN - GAP) // COLUMNS
    caption_layouts: list[
        tuple[
            list[str],
            list[str],
            ImageFont.ImageFont,
            list[tuple[str, ImageFont.ImageFont, int]],
        ]
    ] = []
    caption_area_height = CAPTION_AREA_HEIGHT
    for sample in images:
        category_text = f"Proposed category: {sample.proposed_label}"
        category_lines = _wrap_to_pixel_width(category_text, category_font, tile_width)
        path_text = f"Source path: {sample.source_relative_path}"
        path_font_size = PATH_FONT_SIZE
        while True:
            path_font = _font(path_font_size)
            path_lines = _wrap_to_pixel_width(path_text, path_font, tile_width)
            positioned_lines, required_height = _caption_line_positions(
                category_lines, category_font, path_lines, path_font
            )
            if required_height <= CAPTION_AREA_HEIGHT or path_font_size <= MIN_PATH_FONT_SIZE:
                break
            path_font_size -= 1
        caption_layouts.append(
            (category_lines, path_lines, path_font, positioned_lines)
        )
        caption_area_height = max(caption_area_height, required_height)

    for category_lines, path_lines, path_font, _ in caption_layouts:
        _validate_caption_layout(
            category_lines,
            category_font,
            path_lines,
            path_font,
            tile_width,
            caption_area_height,
        )

    tile_height = IMAGE_AREA_HEIGHT + CAPTION_GAP + caption_area_height
    page_height = 2 * MARGIN + 2 * tile_height + GAP
    page = Image.new("RGB", (PAGE_WIDTH, page_height), "white")
    draw = ImageDraw.Draw(page)

    for index, (sample, caption_layout) in enumerate(zip(images, caption_layouts)):
        column = index % COLUMNS
        row = index // COLUMNS
        x = MARGIN + column * (tile_width + GAP)
        y = MARGIN + row * (tile_height + GAP)
        thumbnail = ImageOps.contain(
            sample.image, (tile_width, IMAGE_AREA_HEIGHT)
        )
        image_x = x + (tile_width - thumbnail.width) // 2
        image_y = y + (IMAGE_AREA_HEIGHT - thumbnail.height) // 2
        page.paste(thumbnail, (image_x, image_y))

        caption_y = y + IMAGE_AREA_HEIGHT + CAPTION_GAP
        _, _, _, positioned_lines = caption_layout
        for line, font, draw_y in positioned_lines:
            draw.text(
                (x, caption_y + draw_y),
                line,
                fill="black",
                font=font,
            )

    return page


def save_category_pdf(
    output_path: Path, images: list[ReviewImage], category: str
) -> None:
    """Save a category's images as four-image, multi-page PDF sheets."""
    pages = [
        _create_page(images[index : index + 4])
        for index in range(0, len(images), 4)
    ]
    if not pages:
        raise ValueError(f"No readable {category} images are available for a review sheet.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        output_path,
        format="PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=144.0,
    )


def main() -> None:
    for category in CATEGORIES:
        images = load_review_images(SAMPLE_CSV_PATH, DATASET_DIR, category)
        if not images:
            output_path = OUTPUT_PATHS[category]
            old_output_note = (
                f" An older PDF may still exist at {output_path}; it was left untouched."
                if output_path.exists()
                else ""
            )
            print(
                f"Warning: no readable {category} images were found in the sample CSV."
                f" No new PDF was written.{old_output_note}"
            )
            continue
        save_category_pdf(OUTPUT_PATHS[category], images, category)
        print(
            f"Saved {len(images)} listed {category} images across "
            f"{(len(images) + 3) // 4} PDF page(s): {OUTPUT_PATHS[category]}"
        )


if __name__ == "__main__":
    main()
