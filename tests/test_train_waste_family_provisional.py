import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def training_module(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "scripts"))
    import prepare_provisional_family
    import train_waste_family_provisional
    return prepare_provisional_family, train_waste_family_provisional


def _manifest_rows(preparation, tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = []
    for folder_index, folder in enumerate(("paper", "battery")):
        for index in range(6):
            path = root / folder / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 9), (folder_index * 100, index * 20, 30)).save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            inventory.append({
                "source_relative_path": f"{folder}/{index}.png",
                "sha256": digest,
                "source_folder": folder,
                "proposed_review_route": preparation.workflow.REVIEW_ROUTES[folder],
                "reviewed_family": "",
                "review_status": "pending",
                "duplicate_group_id": f"sha256:{digest}",
                "inventory_error": "",
            })
    manifest = preparation.build_provisional_manifest(
        inventory, root, msw_folders=("paper",), e_waste_folders=("battery",),
        max_per_folder=None, seed=10,
    )
    return root, manifest


def _write_manifest(path, preparation, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=preparation.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_provisional_training_manifest_uses_only_canonical_provisional_rows(
    tmp_path, training_module,
):
    preparation, trainer = training_module
    _, rows = _manifest_rows(preparation, tmp_path)
    path = tmp_path / "manifest.csv"
    _write_manifest(path, preparation, rows)
    all_rows, train_validation = trainer.read_provisional_manifest(path)

    assert len(all_rows) == 12
    assert {record.family for record in train_validation} == {"MSW", "E_WASTE"}
    assert {record.split for record in train_validation} == {"train", "validation"}
    assert all(rows_by_source["label_provenance"] == preparation.PROVENANCE
               for rows_by_source in all_rows if rows_by_source["group_status"] == "provisional_include")


def test_provisional_training_rejects_duplicate_hash_leakage(tmp_path, training_module):
    preparation, trainer = training_module
    _, rows = _manifest_rows(preparation, tmp_path)
    first = next(row for row in rows if row["group_status"] == "provisional_include" and row["split"] == "train")
    second = next(row for row in rows if row["group_status"] == "provisional_include" and row["split"] == "test")
    second["sha256"] = first["sha256"]
    path = tmp_path / "leak.csv"
    _write_manifest(path, preparation, rows)
    with pytest.raises(trainer.ProvisionalTrainingError, match="leakage"):
        trainer.read_provisional_manifest(path)


def test_test_labels_are_validated_only_after_checkpoint_selection_stage(training_module, tmp_path):
    preparation, trainer = training_module
    _, rows = _manifest_rows(preparation, tmp_path)
    test_row = next(row for row in rows if row["split"] == "test" and row["group_status"] == "provisional_include")
    test_row["provisional_family"] = "invented"
    path = tmp_path / "deferred_test.csv"
    _write_manifest(path, preparation, rows)
    _, train_validation = trainer.read_provisional_manifest(path)
    assert train_validation
    with pytest.raises(trainer.ProvisionalTrainingError, match="Invalid provisional test"):
        trainer.validate_test_records(rows)


def test_provisional_training_verifies_current_hash_and_image(tmp_path, training_module):
    _, trainer = training_module
    root = tmp_path / "dataset"
    root.mkdir()
    path = root / "paper" / "one.png"
    path.parent.mkdir()
    Image.new("RGB", (5, 5), "red").save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = trainer.ProvisionalImage("paper/one.png", digest, "MSW", "train", "paper")
    assert trainer.validate_image_file(root, record) == path.resolve()
    path.write_bytes(b"changed")
    with pytest.raises(trainer.ProvisionalTrainingError, match="SHA-256 changed"):
        trainer.validate_image_file(root, record)


def test_provisional_output_directory_is_never_overwritten(tmp_path, training_module, monkeypatch):
    _, trainer = training_module
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(trainer, "OUTPUT_DIR", output.resolve())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trainer.train_provisional_classifier(tmp_path / "unused.csv", tmp_path, output)
