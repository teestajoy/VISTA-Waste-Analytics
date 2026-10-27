import json
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture
def classifier_module(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "src"))
    import vista.classification.provisional_waste_family_classifier as module
    return module


def test_missing_provisional_checkpoint_fails_closed(classifier_module, tmp_path):
    with pytest.raises(FileNotFoundError, match="Provisional family checkpoint not found"):
        classifier_module.ProvisionalWasteFamilyClassifier(tmp_path / "missing.pth")


def test_provisional_checkpoint_requires_folder_label_provenance(
    classifier_module, tmp_path, monkeypatch,
):
    module = classifier_module
    checkpoint_path = tmp_path / "best_model.pth"
    checkpoint_path.touch()
    checkpoint = {
        "state_dict": {"fc.weight": torch.zeros((2, 512)), "fc.bias": torch.zeros(2)},
        "class_names": module.CLASS_NAMES,
        "preprocessing": module.EXPECTED_PREPROCESSING,
        "provisional": True,
        "label_provenance": module.LABEL_PROVENANCE,
    }
    metadata = {
        "model_status": "PROVISIONAL",
        "label_provenance": module.LABEL_PROVENANCE,
        "verified_ground_truth": False,
        "class_names": list(module.CLASS_NAMES),
        "architecture": "torchvision ResNet18",
        "preprocessing": module.EXPECTED_PREPROCESSING,
        "test_set_used_only_after_validation_model_selection": True,
    }
    (tmp_path / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "final_test_metrics.json").write_text(json.dumps({
        "evaluation_status": "PROVISIONAL folder-derived labels",
        "per_class": {name: {"support": 1} for name in module.CLASS_NAMES},
    }), encoding="utf-8")
    (tmp_path / "class_mapping.json").write_text(json.dumps({
        "class_names": list(module.CLASS_NAMES),
        "class_to_index": {name: index for index, name in enumerate(module.CLASS_NAMES)},
        "label_provenance": module.LABEL_PROVENANCE,
    }), encoding="utf-8")
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: checkpoint)

    class FakeArchitecture:
        def __init__(self):
            self.fc = torch.nn.Linear(512, 2)

        def load_state_dict(self, state, strict):
            assert state is checkpoint["state_dict"]
            assert strict

    monkeypatch.setattr(module.models, "resnet18", lambda **kwargs: FakeArchitecture())
    model = module.ProvisionalWasteFamilyClassifier._load_model(checkpoint_path)
    assert isinstance(model, FakeArchitecture)

    metadata["verified_ground_truth"] = True
    (tmp_path / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(module.ProvisionalFamilyModelError, match="folder-derived provisional"):
        module.ProvisionalWasteFamilyClassifier._load_model(checkpoint_path)


class FakeLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, inputs):
        return self.logits.unsqueeze(0).expand(inputs.shape[0], -1)


def test_provisional_low_score_is_uncertain_and_high_score_stays_provisional(classifier_module):
    module = classifier_module
    image = Image.new("RGB", (13, 9), "orange")
    low = module.ProvisionalWasteFamilyClassifier(model=FakeLogitModel([0.1, 0.0]), device="cpu")
    low_result = low.predict(image)
    assert low_result["is_uncertain"]
    assert low_result["family"] is None
    assert "manual review" in low_result["status"]

    high = module.ProvisionalWasteFamilyClassifier(model=FakeLogitModel([0.0, 5.0]), device="cpu")
    high_result = high.predict(image)
    assert high_result["family"].value == "E_WASTE"
    assert high_result["score"] > 0.70
    assert "PROVISIONAL" in high_result["status"]
