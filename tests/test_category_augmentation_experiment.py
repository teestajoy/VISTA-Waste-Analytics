import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def experiment(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "scripts"))
    import run_category_augmentation_experiment_v1 as module
    return module


def _manifest_row(module, category, split):
    path = f"{category}/{split}_{category}.jpg"
    digest = hashlib.sha256(path.encode()).hexdigest()
    return {
        "source_relative_path": path,
        "sha256": digest,
        "source_folder": category,
        "provisional_category": category,
        "label_provenance": module.category_training.PROVENANCE,
        "duplicate_group_id": f"sha256:{digest}",
        "canonical_path": path,
        "group_status": "provisional_include",
        "split": split,
        "exclusion_reason": "",
    }


def test_manifest_selection_uses_only_existing_train_validation_rows(experiment, tmp_path):
    path = tmp_path / "split.csv"
    rows = [_manifest_row(experiment, c, s)
            for s in experiment.TRAIN_VALIDATION for c in experiment.CLASS_NAMES]
    # Deliberately malformed test metadata must be ignored by the train/validation reader.
    rows.append({**_manifest_row(experiment, "paper", "test"), "sha256": "bad", "provisional_category": "invalid"})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = experiment.read_train_validation_rows(path)
    assert len(selected) == 2 * len(experiment.CLASS_NAMES)
    assert {row["split"] for row in selected} == {"train", "validation"}
    assert not any(row["split"] == "test" for row in selected)


def test_manifest_selection_rejects_duplicate_hashes_in_train_validation(experiment, tmp_path):
    path = tmp_path / "split.csv"
    rows = [_manifest_row(experiment, c, s)
            for s in experiment.TRAIN_VALIDATION for c in experiment.CLASS_NAMES]
    rows[-1]["sha256"] = rows[0]["sha256"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Duplicate or blank"):
        experiment.read_train_validation_rows(path)


def test_pending_disagreements_are_rejected_if_in_train_validation(experiment, tmp_path):
    pending = tmp_path / "pending.csv"
    pending.write_text(
        "source_relative_path,review_status\nglass/train_glass.jpg,pending\n",
        encoding="utf-8",
    )
    rows = [{"source_relative_path": "glass/train_glass.jpg"}]
    with pytest.raises(ValueError, match="Pending review cases"):
        experiment.verify_pending_reviews(rows, pending)


def test_output_directory_creation_refuses_to_reuse_an_existing_path(experiment, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment.create_output_directory(output)
    assert marker.read_text(encoding="utf-8") == "preserve"
    with pytest.raises(ValueError, match="under"):
        experiment.resolve_output_path(tmp_path / "new-output")


def test_augmentation_is_train_only_and_validation_preprocessing_is_deterministic(experiment):
    train_transform, validation_transform = experiment.build_transforms()
    assert isinstance(train_transform.transforms[0], experiment.transforms.RandomResizedCrop)
    assert isinstance(train_transform.transforms[1], experiment.transforms.RandomApply)
    assert isinstance(train_transform.transforms[1].transforms[0], experiment.transforms.ColorJitter)
    assert isinstance(validation_transform.transforms[0], experiment.transforms.Resize)
    # Use non-uniform pixels so a stochastic crop/photometric transform cannot
    # pass merely because all pixels have the same value.
    pattern = np.zeros((80, 80, 3), dtype=np.uint8)
    pattern[:, :, 0] = np.arange(80, dtype=np.uint8)[None, :] * 3
    pattern[:, :, 1] = np.arange(80, dtype=np.uint8)[:, None] * 3
    pattern[:, :, 2] = (pattern[:, :, 0] // 2 + pattern[:, :, 1] // 2)
    image = Image.fromarray(pattern, mode="RGB")
    first = validation_transform(image)
    second = validation_transform(image)
    assert np.array_equal(first.numpy(), second.numpy())
    assert all(not type(transform).__name__.startswith("Random")
               for transform in validation_transform.transforms)


def test_validation_metrics_report_directional_errors_and_threshold_coverage(experiment):
    glass = experiment.CLASS_NAMES.index("glass")
    plastic = experiment.CLASS_NAMES.index("plastic")
    targets = [glass, glass, plastic, plastic]
    probabilities = [
        [0.0, 0.0, 0.05, 0.80, 0.10, 0.05],
        [0.0, 0.0, 0.75, 0.10, 0.10, 0.05],
        [0.0, 0.0, 0.65, 0.20, 0.10, 0.05],
        [0.0, 0.0, 0.05, 0.80, 0.10, 0.05],
    ]
    metrics = experiment.metrics_from_probabilities(targets, probabilities)
    assert metrics["glass_to_plastic"] == {
        "top1_count": 1, "accepted_count": 1, "glass_support": 2,
    }
    assert metrics["plastic_to_glass"] == {
        "top1_count": 1, "accepted_count": 1, "plastic_support": 2,
    }
    assert metrics["operating_point"]["coverage"] == pytest.approx(0.75)
    assert metrics["operating_point"]["accepted_accuracy"] == pytest.approx(1 / 3)
    assert metrics["top1"]["per_class"]["glass"]["support"] == 2


def test_prediction_csv_pair_and_saved_metrics_are_reproducible(experiment, tmp_path):
    baseline = [
        {"source_relative_path": "glass/one.jpg", "reviewed_label": "glass", "top_candidate": "glass",
         "confidence": "0.90", "uncertain_at_0_70": "False", "split": "validation"},
        {"source_relative_path": "plastic/two.jpg", "reviewed_label": "plastic", "top_candidate": "glass",
         "confidence": "0.80", "uncertain_at_0_70": "False", "split": "validation"},
        {"source_relative_path": "glass/three.jpg", "reviewed_label": "glass", "top_candidate": "plastic",
         "confidence": "0.60", "uncertain_at_0_70": "True", "split": "validation"},
    ]
    augmented = [dict(row) for row in baseline]
    augmented[1]["top_candidate"] = "plastic"
    augmented[1]["confidence"] = "0.75"
    fields = list(baseline[0])
    paths = []
    for name, rows in (("baseline.csv", baseline), ("augmented.csv", augmented)):
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    baseline_saved = experiment.read_prediction_csv(paths[0])
    augmented_saved = experiment.read_prediction_csv(paths[1])
    experiment.validate_prediction_pair(baseline_saved, augmented_saved)

    baseline_metrics = experiment.metrics_from_prediction_rows(baseline_saved)
    augmented_metrics = experiment.metrics_from_prediction_rows(augmented_saved)
    assert baseline_metrics["top1"]["accuracy"] == pytest.approx(1 / 3)
    assert baseline_metrics["top1"]["macro_f1"] == pytest.approx(
        sum(item["f1"] for item in baseline_metrics["top1"]["per_class"].values()) / 6
    )
    assert baseline_metrics["operating_point"]["coverage"] == pytest.approx(2 / 3)
    assert baseline_metrics["operating_point"]["accepted_accuracy"] == pytest.approx(1 / 2)
    assert augmented_metrics["top1"]["accuracy"] == pytest.approx(2 / 3)

    reordered = [dict(row) for row in augmented_saved]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="mismatch"):
        experiment.validate_prediction_pair(baseline_saved, reordered)


def test_new_checkpoint_metadata_agrees_on_architecture_mapping_and_run_settings(experiment):
    mapping = {name: index for index, name in enumerate(experiment.CLASS_NAMES)}
    preprocessing = experiment.category_training.PREPROCESSING
    weights = {"path": "cached/resnet18.pth", "sha256": "a" * 64, "hash_algorithm": "SHA-256"}
    metadata = {
        "architecture": experiment.ARCHITECTURE,
        "class_count": len(experiment.CLASS_NAMES),
        "class_names": list(experiment.CLASS_NAMES),
        "class_to_index": mapping,
        "seed": experiment.SEED,
        "confidence_threshold": experiment.THRESHOLD,
        "input_preprocessing": preprocessing,
        "pretrained_weights": weights,
        "augmentation": experiment.AUGMENTATION,
    }
    checkpoint = {
        **metadata,
        "state_dict": {
            "fc.weight": np.zeros((len(experiment.CLASS_NAMES), 512), dtype=np.float32),
            "fc.bias": np.zeros((len(experiment.CLASS_NAMES),), dtype=np.float32),
        },
    }
    experiment.validate_checkpoint_metadata(checkpoint, metadata)
    checkpoint["class_to_index"] = {name: index + 1 for index, name in enumerate(experiment.CLASS_NAMES)}
    with pytest.raises(ValueError, match="class mapping"):
        experiment.validate_checkpoint_metadata(checkpoint, metadata)
