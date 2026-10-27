from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def detector_module(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from vista.detection import object_detector

    return object_detector


class FakeBoxes:
    def __init__(self, xyxy, class_ids, confidences):
        self.xyxy = xyxy
        self.cls = class_ids
        self.conf = confidences


class FakeResult:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names or {}


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return self.results


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_detector_rejects_out_of_range_threshold(detector_module, threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        detector_module.YoloObjectDetector(
            confidence_threshold=threshold,
            model=FakeModel([]),
        )


@pytest.mark.parametrize(
    "threshold, error_type",
    [
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.5", TypeError),
    ],
)
def test_detector_rejects_non_finite_or_non_numeric_threshold(
    detector_module, threshold, error_type
):
    with pytest.raises(error_type, match="between 0 and 1"):
        detector_module.YoloObjectDetector(
            confidence_threshold=threshold,
            model=FakeModel([]),
        )


def test_detector_converts_model_output_to_plain_values(
    tmp_path, detector_module
):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24)).save(image_path)
    model = FakeModel(
        [
            FakeResult(
                FakeBoxes([[1.5, 2, 20, 18]], [2.0], [0.875]),
                names={2: "bottle"},
            )
        ]
    )
    detector = detector_module.YoloObjectDetector(model=model)

    detections = detector.detect(image_path)

    assert detections == [
        detector_module.Detection(
            box=detector_module.BoundingBox(1.5, 2.0, 20.0, 18.0),
            class_id=2,
            class_label="bottle",
            confidence=0.875,
        )
    ]
    assert model.predict_calls[0]["source"] == str(image_path)


def test_detector_returns_empty_list_when_model_finds_nothing(
    tmp_path, detector_module
):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24)).save(image_path)
    model = FakeModel([FakeResult(FakeBoxes([], [], []))])
    detector = detector_module.YoloObjectDetector(model=model)

    assert detector.detect(image_path) == []


def test_detector_returns_empty_list_when_model_returns_no_results(
    tmp_path, detector_module
):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24)).save(image_path)
    detector = detector_module.YoloObjectDetector(model=FakeModel([]))

    assert detector.detect(image_path) == []


def test_detector_rejects_inconsistent_model_output(detector_module):
    result = FakeResult(FakeBoxes([[1, 2, 20, 18]], [2], []), names={2: "bottle"})

    with pytest.raises(
        detector_module.DetectionOutputError,
        match="different counts for boxes, class IDs, and confidences",
    ):
        detector_module.YoloObjectDetector._convert_result(result)


def test_detector_falls_back_to_class_id_when_label_is_missing(
    tmp_path, detector_module
):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24)).save(image_path)
    model = FakeModel([FakeResult(FakeBoxes([[1, 2, 20, 18]], [7], [0.5]))])
    detector = detector_module.YoloObjectDetector(model=model)

    detections = detector.detect(image_path)

    assert detections[0].class_label == "7"


def test_detector_rejects_invalid_image_before_inference(
    tmp_path, detector_module
):
    model = FakeModel([])
    detector = detector_module.YoloObjectDetector(model=model)

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        detector.detect(tmp_path / "missing.png")

    assert model.predict_calls == []
