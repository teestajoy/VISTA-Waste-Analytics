from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def image_input(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from vista.detection import image_input

    return image_input


def test_validate_image_returns_metadata(tmp_path, image_input):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 12)).save(image_path)

    metadata = image_input.validate_image(image_path)

    assert metadata == image_input.ImageMetadata(width=16, height=12, format="PNG")


def test_validate_image_reports_missing_file(tmp_path, image_input):
    image_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        image_input.validate_image(image_path)


def test_validate_image_rejects_corrupt_file(tmp_path, image_input):
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(image_input.ImageInputError, match="not a supported, readable image"):
        image_input.validate_image(image_path)


def test_validate_image_rejects_text_file(tmp_path, image_input):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("This is ordinary text.", encoding="utf-8")

    with pytest.raises(image_input.ImageInputError, match="not a supported, readable image"):
        image_input.validate_image(text_path)
