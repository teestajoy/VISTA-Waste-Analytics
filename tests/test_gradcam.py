import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn


@pytest.fixture
def gradcam_module(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from vista.explainability import gradcam

    return gradcam


class TinyClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, image):
        features = self.layer4(image)
        return self.fc(self.pool(features).flatten(1))


class TinyClassifier:
    class_names = ("paper", "cardboard")
    device = torch.device("cpu")

    def __init__(self, model):
        self.model = model

    @staticmethod
    def transform(image):
        array = np.asarray(image.resize((16, 16)), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def test_gradcam_returns_normalized_heatmap_and_overlay_and_restores_model_state(gradcam_module):
    torch.manual_seed(4)
    model = TinyClassifierModel().train()
    classifier = TinyClassifier(model)
    original_states = [module.training for module in model.modules()]
    hooks_before = len(model.layer4._forward_hooks)

    result = gradcam_module.generate_gradcam(
        classifier, Image.new("RGB", (21, 13), (80, 140, 210))
    )

    assert result.target_class in classifier.class_names
    assert result.heatmap.shape == (13, 21)
    assert np.isfinite(result.heatmap).all()
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0
    assert result.overlay.size == (21, 13)
    assert result.overlay.mode == "RGB"
    assert [module.training for module in model.modules()] == original_states
    assert len(model.layer4._forward_hooks) == hooks_before


def test_gradcam_rejects_invalid_input_and_unsupported_architecture(gradcam_module):
    classifier = TinyClassifier(TinyClassifierModel())
    with pytest.raises(ValueError, match="Pillow image"):
        gradcam_module.generate_gradcam(classifier, "not an image")

    unsupported = TinyClassifier(nn.Sequential(nn.Flatten(), nn.Linear(3, 2)))
    with pytest.raises(gradcam_module.GradCAMError, match="layer4"):
        gradcam_module.generate_gradcam(unsupported, Image.new("RGB", (8, 8)))


def test_gradcam_wraps_gradient_errors_and_still_restores_hooks(gradcam_module):
    class NoGradientModel(TinyClassifierModel):
        def forward(self, image):
            self.layer4(image)
            return torch.zeros((1, 2))

    model = NoGradientModel().eval()
    classifier = TinyClassifier(model)
    hooks_before = len(model.layer4._forward_hooks)

    with pytest.raises(gradcam_module.GradCAMError, match="Grad-CAM computation failed"):
        gradcam_module.generate_gradcam(classifier, Image.new("RGB", (8, 8)))

    assert len(model.layer4._forward_hooks) == hooks_before
    assert model.training is False
