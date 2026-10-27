import json
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture
def module(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "src"))
    import vista.classification.provisional_waste_category_classifier as category
    return category


def _write_artifacts(module, folder, monkeypatch):
    checkpoint_path = folder / "best_model.pth"
    checkpoint_path.touch()
    mapping = {name: family.value for name, family in module.CATEGORY_TO_FAMILY.items()}
    (folder / "training_metadata.json").write_text(json.dumps({
        "model_status": "PROVISIONAL",
        "label_provenance": module.LABEL_PROVENANCE,
        "verified_ground_truth": False,
        "class_names": list(module.CLASS_NAMES),
        "architecture": "torchvision ResNet18",
        "preprocessing": module.EXPECTED_PREPROCESSING,
        "test_set_used_only_after_validation_model_selection": True,
    }), encoding="utf-8")
    (folder / "class_mapping.json").write_text(json.dumps({
        "class_names": list(module.CLASS_NAMES),
        "class_to_index": {name: index for index, name in enumerate(module.CLASS_NAMES)},
        "category_to_family": mapping,
        "label_provenance": module.LABEL_PROVENANCE,
    }), encoding="utf-8")
    (folder / "final_test_metrics.json").write_text(json.dumps({
        "evaluation_status": "PROVISIONAL folder-derived category labels",
        "per_class": {name: {"support": 1} for name in module.CLASS_NAMES},
    }), encoding="utf-8")
    checkpoint = {
        "state_dict": {
            "fc.weight": torch.zeros((len(module.CLASS_NAMES), 512)),
            "fc.bias": torch.zeros(len(module.CLASS_NAMES)),
        },
        "class_names": module.CLASS_NAMES,
        "category_to_family": mapping,
        "preprocessing": module.EXPECTED_PREPROCESSING,
        "provisional": True,
        "label_provenance": module.LABEL_PROVENANCE,
    }
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: checkpoint)

    class FakeArchitecture:
        def __init__(self):
            self.fc = torch.nn.Linear(512, len(module.CLASS_NAMES))

        def load_state_dict(self, state, strict):
            assert state is checkpoint["state_dict"]
            assert strict

    monkeypatch.setattr(module.models, "resnet18", lambda **kwargs: FakeArchitecture())
    return checkpoint_path, checkpoint, mapping


class FakeLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.fc = torch.nn.Linear(512, 6)
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, inputs):
        return self.logits.unsqueeze(0).expand(inputs.shape[0], -1)


def test_missing_checkpoint_is_reported(module, tmp_path):
    with pytest.raises(FileNotFoundError, match="Provisional category checkpoint not found"):
        module.ProvisionalWasteCategoryClassifier(tmp_path / "missing.pth")


def test_incomplete_model_files_are_reported(module, tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.touch()
    with pytest.raises(module.ProvisionalCategoryModelError, match="artifacts are incomplete"):
        module.ProvisionalWasteCategoryClassifier(checkpoint)


def test_loader_validates_class_order_and_provisional_provenance(module, tmp_path, monkeypatch):
    checkpoint, state, _ = _write_artifacts(module, tmp_path, monkeypatch)
    loaded = module.ProvisionalWasteCategoryClassifier._load_model(checkpoint)
    assert loaded.fc.out_features == 6
    metadata_path = tmp_path / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["verified_ground_truth"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(module.ProvisionalCategoryModelError, match="folder-derived provisional"):
        module.ProvisionalWasteCategoryClassifier._load_model(checkpoint)


def test_loader_rejects_unsupported_class_mapping(module, tmp_path, monkeypatch):
    checkpoint, _, _ = _write_artifacts(module, tmp_path, monkeypatch)
    mapping_path = tmp_path / "class_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["class_names"][0] = "metal"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(module.ProvisionalCategoryModelError, match="folder-derived provisional"):
        module.ProvisionalWasteCategoryClassifier._load_model(checkpoint)


def test_prediction_returns_category_family_and_manual_review_for_low_score(module):
    image = Image.new("RGB", (17, 11), "green")
    confident = module.ProvisionalWasteCategoryClassifier(
        model=FakeLogitModel([0, 0, 0, 0, 0, 7]), device="cpu"
    ).predict(image)
    assert confident.category.value == "battery"
    assert confident.family_mapping.value == "E_WASTE"
    assert confident.score > 0.70
    assert "PROVISIONAL" in confident.status

    uncertain = module.ProvisionalWasteCategoryClassifier(
        model=FakeLogitModel([0, 0, 0, 0, 0, 0]), device="cpu"
    ).predict(image)
    assert uncertain.is_uncertain
    assert uncertain.category is None
    assert uncertain.candidate_category.value == "paper"
    assert "manual review" in uncertain.status


def test_low_confidence_cardboard_candidate_is_never_returned_as_category(module):
    image = Image.new("RGB", (17, 11), "brown")
    prediction = module.ProvisionalWasteCategoryClassifier(
        model=FakeLogitModel([0, 1, 0, 0, 0, 0]), device="cpu"
    ).predict(image)

    assert prediction.candidate_category.value == "cardboard"
    assert prediction.score < 0.70
    assert prediction.category is None
    assert prediction.is_uncertain is True
    assert prediction.top_candidates[0] == (prediction.candidate_category, prediction.score)
    assert len(prediction.top_candidates) == 3
    assert [score for _, score in prediction.top_candidates] == sorted(
        (score for _, score in prediction.top_candidates), reverse=True
    )


def test_classifier_rejects_non_pillow_input(module):
    classifier = module.ProvisionalWasteCategoryClassifier(
        model=FakeLogitModel([1, 0, 0, 0, 0, 0]), device="cpu"
    )
    with pytest.raises(TypeError, match="Pillow image"):
        classifier.predict("paper.png")
