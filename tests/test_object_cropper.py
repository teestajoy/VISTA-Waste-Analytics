from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def cropper_modules(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from vista.detection import object_cropper, object_detector

    return object_cropper, object_detector


@pytest.fixture
def synthetic_image():
    image = Image.new("RGB", (6, 5))
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), (x, y, 0))
    return image


def make_detection(object_detector, coordinates):
    box = object_detector.BoundingBox(*coordinates)
    return object_detector.Detection(
        box=box,
        class_id=1,
        class_label="sample",
        confidence=0.9,
    )


def test_crop_detection_returns_expected_region_and_pixels(
    synthetic_image, cropper_modules
):
    cropper, object_detector = cropper_modules
    detection = make_detection(object_detector, (1, 1, 4, 3))

    crop = cropper.crop_detection(synthetic_image, detection)

    assert crop.size == (3, 2)
    assert crop.getpixel((0, 0)) == (1, 1, 0)
    assert crop.getpixel((2, 1)) == (3, 2, 0)


def test_crop_detection_rounds_fractional_edges_outward(
    synthetic_image, cropper_modules
):
    cropper, object_detector = cropper_modules
    detection = make_detection(object_detector, (1.2, 1.7, 4.1, 3.1))

    crop = cropper.crop_detection(synthetic_image, detection)

    assert crop.size == (4, 3)
    assert crop.getpixel((0, 0)) == (1, 1, 0)
    assert crop.getpixel((3, 2)) == (4, 3, 0)


def test_crop_detection_clips_box_to_image_boundaries(
    synthetic_image, cropper_modules
):
    cropper, object_detector = cropper_modules
    detection = make_detection(object_detector, (-2, 1, 3.2, 10))

    crop = cropper.crop_detection(synthetic_image, detection)

    assert crop.size == (4, 4)
    assert crop.getpixel((0, 0)) == (0, 1, 0)
    assert crop.getpixel((3, 3)) == (3, 4, 0)


@pytest.mark.parametrize(
    "coordinates",
    [
        (1, 1, 1, 3),
        (1, 1, 4, 1),
        (4, 1, 2, 3),
        (1, 4, 3, 2),
        (float("nan"), 1, 4, 3),
        (1, 1, float("inf"), 3),
        (float("-inf"), 1, 4, 3),
    ],
)
def test_crop_detection_rejects_invalid_coordinates(
    synthetic_image, cropper_modules, coordinates
):
    cropper, object_detector = cropper_modules
    detection = make_detection(object_detector, coordinates)

    with pytest.raises(ValueError):
        cropper.crop_detection(synthetic_image, detection)


def test_crop_detection_rejects_box_outside_image(
    synthetic_image, cropper_modules
):
    cropper, object_detector = cropper_modules
    detection = make_detection(object_detector, (7, 1, 10, 3))

    with pytest.raises(ValueError, match="no visible area"):
        cropper.crop_detection(synthetic_image, detection)


def test_crop_detection_rejects_box_starting_at_right_boundary(
    synthetic_image, cropper_modules
):
    cropper, object_detector = cropper_modules
    detection = make_detection(object_detector, (6, 1, 8, 3))

    with pytest.raises(ValueError, match="no visible area"):
        cropper.crop_detection(synthetic_image, detection)
