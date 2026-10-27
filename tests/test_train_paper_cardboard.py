import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def trainer(monkeypatch):
    scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
    monkeypatch.syspath_prepend(scripts_path)
    import train_paper_cardboard

    return train_paper_cardboard


def manifest_row(path, label, split, *, status="approved", reason=None, digest=None):
    data = path.read_bytes() if path.exists() else path.as_posix().encode("utf-8")
    return {
        "source_relative_path": path.as_posix(),
        "sha256": digest or hashlib.sha256(data).hexdigest(),
        "source_folder": label,
        "proposed_label": label,
        "reviewed_label": label,
        "review_status": status,
        "duplicate_group_id": f"sha256:{digest or hashlib.sha256(data).hexdigest()}",
        "split": split,
        "inclusion_exclusion_reason": reason or "Included as canonical path. Human reviewed.",
    }


def write_manifest(path, rows):
    fields = (
        "source_relative_path",
        "sha256",
        "source_folder",
        "proposed_label",
        "reviewed_label",
        "review_status",
        "duplicate_group_id",
        "split",
        "inclusion_exclusion_reason",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_image(root, relative, color):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, format="PNG")
    return path


def test_manifest_selects_only_canonical_included_rows(tmp_path, trainer):
    rows = [
        {
            **manifest_row(Path("paper/a.png"), "paper", "train"),
            "source_relative_path": "paper/a.png",
        },
        {
            **manifest_row(Path("paper/b.png"), "paper", "train", status="approved_via_exact_duplicate",
                           reason="Excluded duplicate copy; canonical path is paper/a.png."),
            "source_relative_path": "paper/b.png",
        },
        {
            **manifest_row(Path("cardboard/c.png"), "cardboard", "train", status="unreviewed",
                           reason="Excluded conservatively: not reviewed."),
            "source_relative_path": "cardboard/c.png",
        },
        {
            **manifest_row(Path("cardboard/d.png"), "cardboard", "", status="hold",
                           reason="Held: ambiguous."),
            "source_relative_path": "cardboard/d.png",
        },
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    all_rows, selected = trainer.read_manifest(manifest)

    assert len(all_rows) == 4
    assert [(item.source_relative_path, item.label, item.split) for item in selected] == [
        ("paper/a.png", "paper", "train")
    ]


def test_manifest_rejects_hash_leakage_across_splits(tmp_path, trainer):
    same_hash = "a" * 64
    rows = [
        {**manifest_row(Path("paper/a.png"), "paper", "train", digest=same_hash),
         "source_relative_path": "paper/a.png"},
        {**manifest_row(Path("paper/b.png"), "paper", "test", digest=same_hash,
                        reason="Excluded duplicate copy; canonical path is paper/a.png."),
         "source_relative_path": "paper/b.png", "review_status": "approved_via_exact_duplicate"},
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    with pytest.raises(trainer.DatasetValidationError, match="hash leakage"):
        trainer.read_manifest(manifest)


def test_manifest_rejects_duplicate_canonical_hashes(tmp_path, trainer):
    same_hash = "b" * 64
    rows = [
        {**manifest_row(Path("paper/a.png"), "paper", "train", digest=same_hash),
         "source_relative_path": "paper/a.png"},
        {**manifest_row(Path("paper/b.png"), "paper", "train", digest=same_hash),
         "source_relative_path": "paper/b.png"},
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    with pytest.raises(trainer.DatasetValidationError, match="more than once"):
        trainer.read_manifest(manifest)


def test_hash_mismatch_is_rejected(tmp_path, trainer):
    path = make_image(tmp_path, "paper/sample.png", (12, 34, 56))
    record = trainer.ManifestImage(
        "paper/sample.png", "0" * 64, "paper", "train"
    )

    with pytest.raises(trainer.DatasetValidationError, match="SHA-256 mismatch"):
        trainer.validate_image_file(tmp_path, record)


def test_missing_and_unreadable_images_are_reported(tmp_path, trainer):
    missing = trainer.ManifestImage("paper/missing.png", "0" * 64, "paper", "train")
    with pytest.raises(trainer.DatasetValidationError, match="missing or resolves outside"):
        trainer.validate_image_file(tmp_path, missing)

    bad_path = tmp_path / "paper" / "bad.png"
    bad_path.parent.mkdir()
    bad_path.write_bytes(b"not a real image")
    unreadable = trainer.ManifestImage(
        "paper/bad.png", hashlib.sha256(bad_path.read_bytes()).hexdigest(), "paper", "train"
    )
    with pytest.raises(trainer.DatasetValidationError, match="unreadable or corrupt"):
        trainer.validate_image_file(tmp_path, unreadable)


def test_image_paths_cannot_escape_dataset_root(tmp_path, trainer):
    outside = tmp_path.parent / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    record = trainer.ManifestImage("../outside.png", "0" * 64, "paper", "train")

    with pytest.raises(trainer.DatasetValidationError, match="Unsafe source_relative_path"):
        trainer.validate_image_file(tmp_path, record)


def test_existing_output_directory_is_never_overwritten(tmp_path, trainer, monkeypatch):
    output = tmp_path / "models" / "paper_cardboard_resnet18_pilot"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(trainer, "OUTPUT_DIR", output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trainer.train_pilot(manifest_path=tmp_path / "missing.csv")

    assert sentinel.read_text(encoding="utf-8") == "existing"


def test_model_configuration_is_two_class_resnet18_with_frozen_backbone(trainer):
    model = trainer.build_model()

    assert model.fc.in_features == 512
    assert model.fc.out_features == 2
    assert model.fc.weight.requires_grad
    assert not model.conv1.weight.requires_grad


def test_cached_weights_check_does_not_download(tmp_path, trainer):
    with pytest.raises(FileNotFoundError, match="No download was attempted"):
        trainer.resolve_cached_weights(tmp_path)


def test_class_counts_use_manifest_assignments(trainer):
    records = [
        trainer.ManifestImage("paper/a", "a" * 64, "paper", "train"),
        trainer.ManifestImage("paper/b", "b" * 64, "paper", "validation"),
        trainer.ManifestImage("paper/c", "c" * 64, "paper", "test"),
        trainer.ManifestImage("cardboard/a", "d" * 64, "cardboard", "train"),
        trainer.ManifestImage("cardboard/b", "e" * 64, "cardboard", "validation"),
        trainer.ManifestImage("cardboard/c", "f" * 64, "cardboard", "test"),
    ]

    counts = trainer.validate_split_class_counts(records)

    assert counts["train"] == {"paper": 1, "cardboard": 1}
    assert counts["validation"] == {"paper": 1, "cardboard": 1}
    assert "test" not in counts


def test_test_labels_and_counts_are_deferred_until_final_evaluation(tmp_path, trainer):
    rows = [
        {
            **manifest_row(Path("paper/train.png"), "paper", "train"),
            "source_relative_path": "paper/train.png",
        },
        {
            **manifest_row(Path("cardboard/train.png"), "cardboard", "train"),
            "source_relative_path": "cardboard/train.png",
        },
        {
            **manifest_row(Path("paper/validation.png"), "paper", "validation"),
            "source_relative_path": "paper/validation.png",
        },
        {
            **manifest_row(Path("cardboard/validation.png"), "cardboard", "validation"),
            "source_relative_path": "cardboard/validation.png",
        },
        {
            **manifest_row(Path("paper/test.png"), "paper", "test"),
            "source_relative_path": "paper/test.png",
            "reviewed_label": "not-a-label",
        },
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    manifest_rows, training_and_validation = trainer.read_manifest(manifest)
    counts = trainer.validate_split_class_counts(training_and_validation)

    assert set(counts) == {"train", "validation"}
    assert len(training_and_validation) == 4
    # The malformed test label does not affect pre-selection parsing or counts.
    with pytest.raises(trainer.DatasetValidationError, match="test reviewed_label"):
        trainer.validate_test_records(manifest_rows)


def test_final_test_validation_requires_both_classes(trainer):
    rows = [
        {
            "source_relative_path": "paper/test.png",
            "sha256": "a" * 64,
            "review_status": "approved",
            "reviewed_label": "paper",
            "split": "test",
            "inclusion_exclusion_reason": "Included as canonical path. Human reviewed.",
        }
    ]

    with pytest.raises(
        trainer.DatasetValidationError,
        match="After checkpoint selection.*missing: cardboard",
    ):
        trainer.validate_test_records(rows)
