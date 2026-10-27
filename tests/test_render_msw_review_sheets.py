import csv
from pathlib import Path

import pytest
from PIL import Image, ImageFont

from scripts import render_msw_review_sheets as renderer


def _write_sample_csv(path: Path, source_relative_path: str, label: str = "paper") -> None:
    with path.open("w", encoding="utf-8", newline="") as sample_file:
        writer = csv.DictWriter(
            sample_file, fieldnames=["source_relative_path", "proposed_label"]
        )
        writer.writeheader()
        writer.writerow(
            {"source_relative_path": source_relative_path, "proposed_label": label}
        )


def test_load_review_images_accepts_windows_style_relative_paths(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image_path = dataset / "paper" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), "red").save(image_path)
    sample_csv = tmp_path / "sample.csv"
    _write_sample_csv(sample_csv, "paper\\sample.png")

    samples = renderer.load_review_images(sample_csv, dataset, "paper")

    assert len(samples) == 1
    assert samples[0].source_relative_path == "paper\\sample.png"
    assert samples[0].image.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("unsafe_path", ["../outside.png", "paper/../../outside.png"])
def test_load_review_images_rejects_traversal_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    sample_csv = tmp_path / "sample.csv"
    _write_sample_csv(sample_csv, unsafe_path)

    with pytest.raises(ValueError, match="Invalid source-relative image path"):
        renderer.load_review_images(sample_csv, tmp_path / "dataset", "paper")


@pytest.mark.parametrize(
    "unsafe_path", ["C:\\outside.png", "\\\\server\\share\\image.png"]
)
def test_load_review_images_rejects_absolute_windows_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    sample_csv = tmp_path / "sample.csv"
    _write_sample_csv(sample_csv, unsafe_path)

    with pytest.raises(ValueError, match="Invalid source-relative image path"):
        renderer.load_review_images(sample_csv, tmp_path / "dataset", "paper")


def test_load_review_images_rejects_symlink_escape_when_supported(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    outside_image = tmp_path / "outside.png"
    Image.new("RGB", (8, 6), "blue").save(outside_image)
    link = dataset / "paper.png"
    try:
        link.symlink_to(outside_image)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")
    sample_csv = tmp_path / "sample.csv"
    _write_sample_csv(sample_csv, "paper.png")

    with pytest.raises(ValueError, match="resolves outside the dataset root"):
        renderer.load_review_images(sample_csv, dataset, "paper")


def test_wrap_to_pixel_width_keeps_long_path_lines_within_caption_width() -> None:
    font = ImageFont.load_default()
    caption = (
        "Source path: paper/Household paper products/"
        + ("very long filename segment " * 12)
        + "sample image.png"
    )

    lines = renderer._wrap_to_pixel_width(caption, font, max_width=180)

    assert len(lines) > 1
    assert "".join(lines) == caption
    assert all(renderer._text_width(line, font) <= 180 for line in lines)

    source_path = caption.removeprefix("Source path: ")
    sample = renderer.ReviewImage(
        source_path, "paper", Image.new("RGB", (10, 10))
    )
    page = renderer._create_page([sample])
    assert page.height > renderer.PAGE_HEIGHT


@pytest.mark.parametrize(
    ("prefix", "error_message"),
    [
        ("Proposed category:", "category caption line"),
        ("Source path:", "source path caption line"),
    ],
)
def test_create_page_rejects_caption_lines_wider_than_tile(
    monkeypatch: pytest.MonkeyPatch, prefix: str, error_message: str
) -> None:
    original_wrap = renderer._wrap_to_pixel_width

    def return_overwide_line(
        text: str, font: ImageFont.ImageFont, max_width: int
    ) -> list[str]:
        if text.startswith(prefix):
            return ["W" * (max_width + 1)]
        return original_wrap(text, font, max_width)

    monkeypatch.setattr(renderer, "_wrap_to_pixel_width", return_overwide_line)
    sample = renderer.ReviewImage(
        "paper/example.png", "paper", Image.new("RGB", (10, 10))
    )

    with pytest.raises(ValueError, match=error_message):
        renderer._create_page([sample])


def test_caption_layout_uses_nonzero_font_offsets_and_line_spacing() -> None:
    class OffsetFont:
        def getlength(self, text: str) -> float:
            return float(len(text) * 5)

        def getbbox(self, text: str) -> tuple[int, int, int, int]:
            return (0, -4, len(text) * 5, 12)

    font = OffsetFont()

    with pytest.raises(ValueError, match="53px is required"):
        renderer._validate_caption_layout(
            ["category"],
            font,
            ["source path"],
            font,
            tile_width=1000,
            caption_height=52,
        )

    renderer._validate_caption_layout(
        ["category"],
        font,
        ["source path"],
        font,
        tile_width=1000,
        caption_height=53,
    )


def test_create_page_rejects_caption_line_above_allocated_area(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OffsetFont:
        def getlength(self, text: str) -> float:
            return float(len(text) * 5)

        def getbbox(self, text: str) -> tuple[int, int, int, int]:
            return (0, -4, len(text) * 5, 12)

    font = OffsetFont()
    monkeypatch.setattr(renderer, "_font", lambda size, bold=False: font)
    # Exercise the guard with a deliberately invalid layout: this makes the
    # path line's calculated top precede the caption area's top edge.
    monkeypatch.setattr(renderer, "LINE_SPACING", -10)
    sample = renderer.ReviewImage(
        "paper/example.png", "paper", Image.new("RGB", (10, 10))
    )

    with pytest.raises(ValueError, match="rendered vertical extent -4px"):
        renderer._create_page([sample])


def test_empty_category_warns_that_existing_pdf_is_left_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_csv = tmp_path / "sample.csv"
    _write_sample_csv(sample_csv, "paper/missing.png")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "older.pdf"
    output.write_bytes(b"old pdf stays untouched")
    monkeypatch.setattr(renderer, "SAMPLE_CSV_PATH", sample_csv)
    monkeypatch.setattr(renderer, "DATASET_DIR", dataset)
    monkeypatch.setattr(
        renderer,
        "OUTPUT_PATHS",
        {"paper": output, "cardboard": tmp_path / "c.pdf"},
    )

    renderer.main()

    message = capsys.readouterr().out
    assert "no readable paper images" in message
    assert "older PDF may still exist" in message
    assert output.read_bytes() == b"old pdf stays untouched"


def test_save_category_pdf_writes_all_images_across_multiple_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_calls: list[dict[str, object]] = []

    def record_save(self: Image.Image, output_path: Path, **kwargs: object) -> None:
        save_calls.append(kwargs)

    monkeypatch.setattr(Image.Image, "save", record_save)
    samples = [
        renderer.ReviewImage(
            f"paper/{index}.png", "paper", Image.new("RGB", (10, 10))
        )
        for index in range(5)
    ]
    output = tmp_path / "sheets" / "paper.pdf"

    renderer.save_category_pdf(output, samples, "paper")

    assert output.parent.is_dir()
    assert len(save_calls) == 1
    assert save_calls[0]["save_all"] is True
    assert len(save_calls[0]["append_images"]) == 1
