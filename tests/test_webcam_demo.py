from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def webcam_demo(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from scripts import webcam_demo

    return webcam_demo


def _detection(webcam_demo, label="bottle", confidence=0.875):
    from vista.detection.object_detector import BoundingBox, Detection

    return Detection(
        box=BoundingBox(10, 10, 30, 30),
        class_id=39,
        class_label=label,
        confidence=confidence,
    )


def test_panel_text_displays_honest_empty_state_and_disclaimers(webcam_demo):
    lines = webcam_demo.panel_text([])

    assert webcam_demo.GENERAL_DETECTION_NOTICE in lines
    assert webcam_demo.EXPLAINABILITY_NOTICE in lines
    assert "No objects detected." in lines


def test_panel_text_lists_generic_detection_labels_and_confidences(webcam_demo):
    detections = [_detection(webcam_demo, "bottle", 0.875)]

    lines = webcam_demo.panel_text(detections)

    assert "bottle — confidence 0.88" in lines
    assert webcam_demo.GENERAL_DETECTION_NOTICE in lines
    assert webcam_demo.EXPLAINABILITY_NOTICE in lines


def test_render_demo_view_draws_boxes_and_keeps_input_frame_unchanged(webcam_demo):
    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    original = frame.copy()

    view = webcam_demo.render_demo_view(frame, [_detection(webcam_demo)])

    assert view.shape[0] >= frame.shape[0]
    assert view.shape[1:] == (80 + webcam_demo.PANEL_WIDTH, 3)
    assert tuple(view[20, 10]) == (0, 255, 0)
    assert np.array_equal(frame, original)
