import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def training_module(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "scripts"))
    from scripts import train_waste_family

    return train_waste_family


def _manifest_row(path, digest, family, split, *, status="include", group="include"):
    return {
        "source_relative_path": path,
        "sha256": digest,
        "source_folder": "paper" if family == "MSW" else "battery",
        "review_status": status,
        "reviewed_family": family,
        "group_status": group,
        "split": split,
    }


def _write_manifest(path, module, rows):
    fields = (
        "source_relative_path", "sha256", "source_folder", "review_status",
        "reviewed_family", "group_status", "split",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _valid_rows():
    rows = []
    for family in ("MSW", "E_WASTE"):
        for split in ("train", "validation", "test"):
            token = f"{family}-{split}".encode()
            rows.append(_manifest_row(
                f"{family}/{split}.png", hashlib.sha256(token).hexdigest(), family, split
            ))
    return rows


def test_manifest_selects_only_explicit_canonical_train_validation_rows(
    tmp_path, training_module
):
    rows = _valid_rows()
    rows.extend([
        _manifest_row("paper/pending.png", hashlib.sha256(b"pending").hexdigest(), "MSW", "", status="pending", group="pending"),
        _manifest_row("paper/copy.png", hashlib.sha256(b"copy").hexdigest(), "MSW", "", group="duplicate_copy"),
    ])
    path = tmp_path / "manifest.csv"
    _write_manifest(path, training_module, rows)

    all_rows, selected = training_module.read_manifest(path)

    assert len(all_rows) == 8
    assert {record.split for record in selected} == {"train", "validation"}
    assert len(selected) == 4
    assert {record.family for record in selected} == {"MSW", "E_WASTE"}


def test_duplicate_hash_leakage_is_rejected(training_module, tmp_path):
    digest = hashlib.sha256(b"leak").hexdigest()
    rows = [
        _manifest_row("paper/a.png", digest, "MSW", "train"),
        _manifest_row("battery/b.png", digest, "E_WASTE", "test"),
    ]
    path = tmp_path / "leak.csv"
    _write_manifest(path, training_module, rows)
    with pytest.raises(training_module.FamilyDatasetValidationError, match="leakage"):
        training_module.read_manifest(path)


def test_test_labels_are_not_validated_until_final_evaluation(
    tmp_path, training_module
):
    rows = _valid_rows()
    test_row = next(row for row in rows if row["split"] == "test")
    test_row["review_status"] = "pending"
    test_row["reviewed_family"] = "fabricated"
    path = tmp_path / "manifest.csv"
    _write_manifest(path, training_module, rows)

    all_rows, selected = training_module.read_manifest(path)

    assert len(selected) == 4
    with pytest.raises(training_module.FamilyDatasetValidationError, match="test row"):
        training_module.validate_test_records(all_rows)


def test_final_test_validation_requires_both_families(training_module):
    rows = [row for row in _valid_rows() if not (row["split"] == "test" and row["reviewed_family"] == "E_WASTE")]
    with pytest.raises(training_module.FamilyDatasetValidationError, match="E_WASTE"):
        training_module.validate_test_records(rows)


def test_train_and_validation_require_both_families(training_module):
    records = [
        training_module.FamilyImage("paper/a.png", "a" * 64, "MSW", "train"),
        training_module.FamilyImage("paper/b.png", "b" * 64, "MSW", "validation"),
    ]
    with pytest.raises(training_module.FamilyDatasetValidationError, match="E_WASTE"):
        training_module.validate_train_validation_counts(records)


def test_training_class_weights_balance_uneven_reviewed_counts(training_module):
    records = [
        training_module.FamilyImage(f"paper/{index}.png", f"{index:064x}", "MSW", "train")
        for index in range(4)
    ] + [training_module.FamilyImage("battery/one.png", "f" * 64, "E_WASTE", "train")]

    weights = training_module.training_class_weights(records).tolist()

    assert weights == pytest.approx([0.625, 2.5])


def test_paths_hashes_and_image_readability_are_validated(tmp_path, training_module):
    root = tmp_path / "dataset"
    root.mkdir()
    path = root / "paper" / "good.png"
    path.parent.mkdir()
    Image.new("RGB", (10, 8), "green").save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = training_module.FamilyImage("paper/good.png", digest, "MSW", "train")
    assert training_module.validate_image_file(root, record) == path.resolve()
    with pytest.raises(training_module.FamilyDatasetValidationError, match="Unsafe"):
        training_module._safe_source_path(root, "../outside.png")
    with pytest.raises(training_module.FamilyDatasetValidationError, match="missing"):
        training_module.validate_image_file(
            root, training_module.FamilyImage("paper/missing.png", digest, "MSW", "train")
        )
    path.write_bytes(b"not an image")
    bad_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(training_module.FamilyDatasetValidationError, match="unreadable"):
        training_module.validate_image_file(
            root, training_module.FamilyImage("paper/good.png", bad_digest, "MSW", "train")
        )


def test_symlink_escape_is_rejected_when_supported(tmp_path, training_module):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    link = root / "paper.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not supported in this test environment")
    with pytest.raises(training_module.FamilyDatasetValidationError, match="outside|symlink"):
        training_module._safe_source_path(root, "paper.png")


def test_cached_weights_lookup_never_downloads(tmp_path, training_module):
    with pytest.raises(FileNotFoundError, match="No download attempted"):
        training_module.resolve_cached_weights(tmp_path)


def test_resnet_head_has_two_family_outputs(training_module):
    model = training_module.build_model()
    assert model.fc.out_features == 2
    assert training_module.CLASS_NAMES == ("MSW", "E_WASTE")


def test_insufficient_manifest_stops_before_creating_model_output(
    tmp_path, training_module, monkeypatch
):
    rows = [
        _manifest_row("paper/train.png", hashlib.sha256(b"one").hexdigest(), "MSW", "train"),
        _manifest_row("paper/validation.png", hashlib.sha256(b"two").hexdigest(), "MSW", "validation"),
    ]
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, training_module, rows)
    output_dir = tmp_path / "family_model"
    monkeypatch.setattr(training_module, "OUTPUT_DIR", output_dir)

    with pytest.raises(training_module.FamilyDatasetValidationError, match="E_WASTE"):
        training_module.train_family_classifier(
            manifest, tmp_path / "dataset", output_dir=output_dir
        )
    assert not output_dir.exists()


def test_metrics_report_both_classes(training_module):
    import numpy as np

    metrics = training_module.metrics_from_confusion(np.array([[2, 1], [0, 3]]))
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert set(metrics["per_class"]) == {"MSW", "E_WASTE"}
    assert metrics["per_class"]["MSW"]["support"] == 3
