from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn


@pytest.fixture
def classifier_module(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)
    from vista.classification import paper_cardboard_classifier

    return paper_cardboard_classifier


class FakeModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.fc = nn.Linear(512, 2)
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.detach().cpu())
        return self.logits.unsqueeze(0).expand(inputs.shape[0], -1)


def test_predict_uses_training_resize_rgb_and_imagenet_normalization(classifier_module):
    model = FakeModel([3.0, 0.0])
    classifier = classifier_module.PaperCardboardClassifier(model=model, device="cpu")
    red_rgb_crop = Image.new("RGB", (12, 8), (255, 0, 0))

    result = classifier.predict(red_rgb_crop)

    assert result.label == "paper"
    assert result.confidence == pytest.approx(torch.softmax(torch.tensor([3.0, 0.0]), dim=0)[0].item())
    assert model.inputs[0].shape == (1, 3, 224, 224)
    expected_red = (1.0 - 0.485) / 0.229
    expected_green = (0.0 - 0.456) / 0.224
    assert model.inputs[0][0, 0, 0, 0].item() == pytest.approx(expected_red)
    assert model.inputs[0][0, 1, 0, 0].item() == pytest.approx(expected_green)


def test_prediction_below_provisional_threshold_is_uncertain(classifier_module):
    classifier = classifier_module.PaperCardboardClassifier(
        model=FakeModel([0.1, 0.0]), device="cpu", confidence_threshold=0.70
    )

    result = classifier.predict(Image.new("RGB", (5, 4), (20, 30, 40)))

    assert result.label is None
    assert result.is_uncertain
    assert result.confidence < 0.70
    assert result.threshold == 0.70


def test_prediction_preserves_class_order_for_cardboard(classifier_module):
    classifier = classifier_module.PaperCardboardClassifier(
        model=FakeModel([0.0, 3.0]), device="cpu"
    )

    result = classifier.predict(Image.new("RGB", (2, 2)))

    assert result.label == "cardboard"
    assert classifier_module.CLASS_NAMES == ("paper", "cardboard")


def test_checkpoint_loader_verifies_two_class_resnet18_and_class_order(
    tmp_path, classifier_module, monkeypatch
):
    checkpoint_path = tmp_path / "best_model.pth"
    checkpoint_path.write_bytes(b"test checkpoint placeholder")
    state = {
        "fc.weight": torch.zeros((2, 512)),
        "fc.bias": torch.zeros(2),
    }
    monkeypatch.setattr(
        classifier_module.torch,
        "load",
        lambda *args, **kwargs: {"class_names": ("paper", "cardboard"), "state_dict": state},
    )
    loaded = {}

    class Model:
        fc = SimpleNamespace(in_features=512)

        def load_state_dict(self, loaded_state, strict):
            loaded["state"] = loaded_state
            loaded["strict"] = strict

    model = classifier_module.build_model_from_checkpoint(
        checkpoint_path, model_factory=Model
    )

    assert isinstance(model, Model)
    assert loaded == {"state": state, "strict": True}


def test_checkpoint_loader_rejects_wrong_class_order(
    tmp_path, classifier_module, monkeypatch
):
    checkpoint_path = tmp_path / "best_model.pth"
    checkpoint_path.write_bytes(b"test checkpoint placeholder")
    monkeypatch.setattr(
        classifier_module.torch,
        "load",
        lambda *args, **kwargs: {
            "class_names": ("cardboard", "paper"),
            "state_dict": {},
        },
    )

    with pytest.raises(classifier_module.ClassifierLoadError, match="class order"):
        classifier_module.build_model_from_checkpoint(checkpoint_path)


def test_missing_checkpoint_has_clear_error(tmp_path, classifier_module):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        classifier_module.build_model_from_checkpoint(tmp_path / "missing.pth")


def test_classifier_rejects_model_with_wrong_output_count(classifier_module):
    model = SimpleNamespace(fc=SimpleNamespace(out_features=3))

    with pytest.raises(classifier_module.ClassifierLoadError, match="exactly two"):
        classifier_module.PaperCardboardClassifier(model=model, device="cpu")
