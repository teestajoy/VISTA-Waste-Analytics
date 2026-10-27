from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture
def family_module(monkeypatch):
    source_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(source_path)
    from vista.classification import family_data_policy, waste_family_classifier

    return family_data_policy, waste_family_classifier


def test_unavailable_family_classifier_returns_uncertain_without_a_score(
    family_module,
):
    _, classifier_module = family_module
    classifier = classifier_module.UnavailableWasteFamilyClassifier()

    result = classifier.predict(Image.new("RGB", (8, 6), "white"))

    assert result.family is None
    assert result.score is None
    assert result.is_uncertain is True
    assert "reviewed E-waste" in result.status


def test_family_prediction_schema_accepts_supported_family_scores(family_module):
    _, classifier_module = family_module
    from vista.classification.taxonomy import WasteFamily

    result = classifier_module.WasteFamilyPrediction(
        WasteFamily.MSW, 0.76, False, "Experimental score; not calibrated."
    )

    assert result.family is WasteFamily.MSW
    assert result.score == 0.76
    assert result.is_uncertain is False


@pytest.mark.parametrize(
    "arguments",
    [
        ("MSW", 0.8, False, "invalid enum"),
        (None, 0.8, False, "missing family"),
        (None, float("nan"), True, "invalid score"),
        (None, None, True, ""),
        ("E_WASTE", 0.8, True, "uncertain label"),
    ],
)
def test_family_prediction_schema_rejects_inconsistent_results(
    family_module, arguments
):
    _, classifier_module = family_module
    with pytest.raises(ValueError):
        classifier_module.WasteFamilyPrediction(*arguments)


def test_family_source_policy_requires_review_and_holds_ambiguous_folders(
    family_module,
):
    policy_module, _ = family_module
    from vista.classification.taxonomy import WasteFamily

    policy = policy_module.MWCD_FAMILY_SOURCE_POLICY
    assert len(policy) == 9
    for folder in ("paper", "cardboard", "plastic", "glass", "organic"):
        assert policy[folder].candidate_family is WasteFamily.MSW
        assert (
            policy[folder].disposition
            is policy_module.SourceDisposition.CANDIDATE_REQUIRES_REVIEW
        )
    assert policy["battery"].candidate_family is WasteFamily.E_WASTE
    assert (
        policy["battery"].disposition
        is policy_module.SourceDisposition.CANDIDATE_REQUIRES_REVIEW
    )
    for folder in ("metal", "trash"):
        assert policy[folder].candidate_family is None
        assert policy[folder].disposition is policy_module.SourceDisposition.HOLD_FOR_POLICY
    assert policy["textile"].candidate_family is None
    assert policy["textile"].disposition is policy_module.SourceDisposition.UNSUPPORTED


def test_family_source_policy_never_automatically_approves_images(family_module):
    policy_module, _ = family_module

    assert all(
        item.disposition is not policy_module.SourceDisposition.CANDIDATE_REQUIRES_REVIEW
        or "verify" in item.rationale.lower()
        or "review" in item.rationale.lower()
        for item in policy_module.MWCD_FAMILY_SOURCE_POLICY.values()
    )


class FakeFamilyModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.fc = torch.nn.Linear(512, 2)
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.detach().cpu())
        return self.logits.unsqueeze(0).expand(inputs.shape[0], -1)


def test_family_resnet_preprocessing_and_low_score_uncertainty(family_module):
    _, classifier_module = family_module
    model = FakeFamilyModel([0.2, 0.0])
    classifier = classifier_module.WasteFamilyResNet18Classifier(model=model, device="cpu")
    result = classifier.predict(Image.new("RGB", (11, 8), (255, 0, 0)))

    assert result.family is None
    assert result.score is not None and result.score < 0.70
    assert result.is_uncertain
    assert "not calibrated" in result.status
    assert model.inputs[0].shape == (1, 3, 224, 224)
    assert model.inputs[0][0, 0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229)
    assert classifier_module.CLASS_NAMES == ("MSW", "E_WASTE")


def test_family_resnet_returns_expected_order_above_provisional_threshold(family_module):
    _, classifier_module = family_module
    from vista.classification.taxonomy import WasteFamily

    classifier = classifier_module.WasteFamilyResNet18Classifier(
        model=FakeFamilyModel([0.0, 4.0]), device="cpu"
    )
    result = classifier.predict(Image.new("RGB", (4, 4)))

    assert result.family is WasteFamily.E_WASTE
    assert result.score is not None and result.score > 0.70
    assert not result.is_uncertain


def test_missing_family_checkpoint_fails_closed(family_module, tmp_path):
    _, classifier_module = family_module
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        classifier_module.build_model_from_checkpoint(tmp_path / "missing.pth")


def test_family_checkpoint_checks_class_order_and_preprocessing(
    family_module, tmp_path, monkeypatch
):
    _, classifier_module = family_module
    checkpoint_path = tmp_path / "best_model.pth"
    checkpoint_path.touch()
    (tmp_path / "training_metadata.json").write_text(
        '{"class_names": ["MSW", "E_WASTE"], '
        '"architecture": "torchvision ResNet18", '
        '"preprocessing": {"color_mode": "RGB", "resize": [224, 224], '
        '"tensor": "ToTensor", "normalize_mean": [0.485, 0.456, 0.406], '
        '"normalize_std": [0.229, 0.224, 0.225]}, '
        '"test_set_used_only_after_validation_model_selection": true}', encoding="utf-8"
    )
    (tmp_path / "final_test_metrics.json").write_text(
        '{"per_class": {"MSW": {"support": 1}, "E_WASTE": {"support": 1}}}',
        encoding="utf-8",
    )
    state = {
        "fc.weight": torch.zeros((2, 512)),
        "fc.bias": torch.zeros(2),
    }
    checkpoint = {
        "class_names": ("MSW", "E_WASTE"),
        "preprocessing": classifier_module.EXPECTED_PREPROCESSING,
        "state_dict": state,
    }
    monkeypatch.setattr(classifier_module.torch, "load", lambda *args, **kwargs: checkpoint)

    class FakeArchitecture:
        def __init__(self):
            self.fc = torch.nn.Linear(512, 2)

        def load_state_dict(self, loaded, strict):
            assert loaded is state
            assert strict is True

    model = classifier_module.build_model_from_checkpoint(
        checkpoint_path, model_factory=FakeArchitecture
    )
    assert isinstance(model, FakeArchitecture)

    checkpoint["class_names"] = ("E_WASTE", "MSW")
    with pytest.raises(classifier_module.FamilyClassifierLoadError, match="class order"):
        classifier_module.build_model_from_checkpoint(
            checkpoint_path, model_factory=FakeArchitecture
        )


def test_family_checkpoint_requires_complete_evaluation_artifacts(
    family_module, tmp_path, monkeypatch
):
    _, classifier_module = family_module
    checkpoint_path = tmp_path / "best_model.pth"
    checkpoint_path.touch()
    monkeypatch.setattr(classifier_module.torch, "load", lambda *args, **kwargs: {})
    with pytest.raises(classifier_module.FamilyClassifierLoadError, match="artifacts are incomplete"):
        classifier_module.build_model_from_checkpoint(checkpoint_path)
