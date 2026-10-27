from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def workflow_modules(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from vista.detection import image_workflow, object_detector

    return image_workflow, object_detector


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def detect(self, path):
        self.calls.append(path)
        return self.detections


class FakeClassifier:
    def __init__(self, prediction=None, error=None):
        self.prediction = prediction
        self.error = error
        self.calls = []

    def predict(self, image):
        self.calls.append(image.copy())
        if self.error:
            raise self.error
        return self.prediction


@pytest.fixture
def synthetic_image(tmp_path):
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (6, 5))
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), (x, y, 0))
    image.save(image_path)
    return image_path


def make_detection(object_detector, coordinates, class_id=1, class_label="sample"):
    return object_detector.Detection(
        box=object_detector.BoundingBox(*coordinates),
        class_id=class_id,
        class_label=class_label,
        confidence=0.9,
    )


def test_detect_and_crop_pairs_detection_with_expected_crop(
    synthetic_image, workflow_modules
):
    workflow, object_detector = workflow_modules
    detection = make_detection(object_detector, (1, 1, 4, 3))
    detector = FakeDetector([detection])

    results = workflow.detect_and_crop(synthetic_image, detector)

    assert len(results) == 1
    assert results[0].detection is detection
    assert results[0].crop.size == (3, 2)
    assert results[0].crop.getpixel((0, 0)) == (1, 1, 0)
    assert results[0].crop.getpixel((2, 1)) == (3, 2, 0)
    assert detector.calls == [synthetic_image]


def test_detect_and_crop_preserves_pairing_for_multiple_detections(
    synthetic_image, workflow_modules
):
    workflow, object_detector = workflow_modules
    first = make_detection(object_detector, (0, 0, 2, 2), 1, "first")
    second = make_detection(object_detector, (3, 1, 6, 5), 2, "second")
    detector = FakeDetector([first, second])

    results = workflow.detect_and_crop(synthetic_image, detector)

    assert [result.detection for result in results] == [first, second]
    assert [result.crop.size for result in results] == [(2, 2), (3, 4)]
    assert results[0].crop.getpixel((0, 0)) == (0, 0, 0)
    assert results[1].crop.getpixel((0, 0)) == (3, 1, 0)


def test_detect_and_crop_returns_empty_list_when_there_are_no_detections(
    synthetic_image, workflow_modules
):
    workflow, _ = workflow_modules
    detector = FakeDetector([])

    assert workflow.detect_and_crop(synthetic_image, detector) == []
    assert detector.calls == [synthetic_image]


@pytest.mark.parametrize("file_kind", ["missing", "corrupt"])
def test_detect_and_crop_rejects_invalid_image_before_detector_call(
    tmp_path, workflow_modules, file_kind
):
    workflow, _ = workflow_modules
    image_path = tmp_path / "input.png"
    if file_kind == "corrupt":
        image_path.write_bytes(b"not a valid image")
    detector = FakeDetector([])

    expected_error = FileNotFoundError if file_kind == "missing" else ValueError
    with pytest.raises(expected_error):
        workflow.detect_and_crop(image_path, detector)

    assert detector.calls == []


def test_detect_and_crop_propagates_invalid_bounding_box_error(
    synthetic_image, workflow_modules
):
    workflow, object_detector = workflow_modules
    invalid_detection = make_detection(object_detector, (4, 1, 2, 3))
    detector = FakeDetector([invalid_detection])

    with pytest.raises(ValueError, match="positive width and height"):
        workflow.detect_and_crop(synthetic_image, detector)


def test_classify_detections_pairs_rgb_crops_with_pilot_predictions(
    synthetic_image, workflow_modules
):
    from vista.classification.paper_cardboard_classifier import PaperCardboardPrediction

    workflow, object_detector = workflow_modules
    detection = make_detection(object_detector, (1, 1, 4, 3), class_label="bottle")
    image = Image.open(synthetic_image).convert("RGB")
    prediction = PaperCardboardPrediction("paper", 0.82, 0.70)
    classifier = FakeClassifier(prediction)

    results = workflow.classify_detections(image, [detection], classifier)

    assert len(results) == 1
    assert results[0].detection is detection
    assert results[0].crop.size == (3, 2)
    assert results[0].prediction is prediction
    assert results[0].error is None
    assert classifier.calls[0].mode == "RGB"


def test_classify_detections_handles_empty_and_invalid_crops(
    synthetic_image, workflow_modules
):
    workflow, object_detector = workflow_modules
    image = Image.open(synthetic_image).convert("RGB")
    classifier = FakeClassifier()
    invalid = make_detection(object_detector, (4, 1, 2, 3))

    assert workflow.classify_detections(image, [], classifier) == []
    result = workflow.classify_detections(image, [invalid], classifier)[0]

    assert result.crop is None
    assert result.prediction is None
    assert "positive width and height" in result.error
    assert classifier.calls == []


def test_classify_detections_reports_classifier_inference_error(
    synthetic_image, workflow_modules
):
    workflow, object_detector = workflow_modules
    image = Image.open(synthetic_image).convert("RGB")
    detection = make_detection(object_detector, (0, 0, 2, 2))
    classifier = FakeClassifier(error=RuntimeError("model inference failed"))

    result = workflow.classify_detections(image, [detection], classifier)[0]

    assert result.crop.size == (2, 2)
    assert result.prediction is None
    assert result.error == "model inference failed"
