import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from torchvision import models

from vista.assessment.quality import condition_score, contamination_score
from vista.classification.loader import WasteClassifier
from vista.explainability.gradcam import GradCAM, overlay
from vista.impact.carbon import estimate_impact
from vista.pipeline import analyze, strip_internal


def _disc(size=200, notch=False):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([30, 30, 170, 170], fill=(40, 90, 160))
    if notch:
        d.polygon([(100, 100), (200, 70), (200, 130)], fill="white")
    return img


def test_intact_scores_lower_than_damaged():
    assert condition_score(_disc()).score < condition_score(_disc(notch=True)).score


def test_flat_image_is_clean():
    assert contamination_score(Image.new("RGB", (64, 64), (200, 200, 200))).verdict == "clean"


def test_scores_are_bounded():
    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    for s in (contamination_score(noisy).score, condition_score(noisy).score):
        assert 0 <= s <= 1


def test_impact_lookup():
    e = estimate_impact("PCB")
    assert e.co2e_kg == pytest.approx(6.5 * 0.06) and estimate_impact("textile") is None


def _tiny_classifier():
    net = models.resnet18(weights=None, num_classes=6)
    return WasteClassifier(net, ["plastic", "paper", "cardboard", "glass", "organic", "non-recyclable"])


def test_gradcam_shape_and_range():
    clf = _tiny_classifier()
    cam = GradCAM(clf.model, clf.target_layer).heatmap(clf.prepare(_disc()))
    assert cam.shape == (224, 224) and 0 <= cam.min() and cam.max() <= 1
    assert overlay(_disc(), cam).size == (200, 200)


def test_pipeline_report_without_model_has_quality_only():
    r = strip_internal(analyze(_disc(), None))
    assert r["classification"] is None and "score" in r["contamination"]


def test_pipeline_with_model_builds_full_report():
    r = analyze(_disc(), _tiny_classifier())
    assert r["classification"]["family"] == "MSW" and r["impact"] and r["_cam"].shape == (224, 224)
    assert "_cam" not in strip_internal(r)


def test_checkpoint_roundtrip(tmp_path):
    net = models.resnet18(weights=None, num_classes=7)
    torch.save({"state_dict": net.state_dict()}, tmp_path / "m.pt")
    clf = WasteClassifier.from_checkpoint(tmp_path / "m.pt")
    assert clf.classes[0] == "metal"
    assert clf.predict(_disc()).family == "E_WASTE"
