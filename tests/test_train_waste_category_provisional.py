import csv
import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def category_modules(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "scripts"))
    import prepare_provisional_family
    import train_waste_category_provisional
    return prepare_provisional_family, train_waste_category_provisional


def _source_manifest(preparation, path, *, repeat=2):
    rows = []
    for category in preparation.DEFAULT_MSW_FOLDERS + preparation.DEFAULT_E_WASTE_FOLDERS:
        family = "E_WASTE" if category == "battery" else "MSW"
        for split in ("train", "validation", "test"):
            for index in range(repeat):
                digest = hashlib.sha256(f"{category}-{split}-{index}".encode()).hexdigest()
                relative = f"{category}/{split}-{index}.png"
                rows.append({
                    "source_relative_path": relative,
                    "sha256": digest,
                    "source_folder": category,
                    "provisional_family": family,
                    "label_provenance": preparation.PROVENANCE,
                    "source_folder_mapping": f"{category} -> {family}",
                    "human_review_status": "pending",
                    "human_reviewed_family": "",
                    "human_review_reason": "",
                    "duplicate_group_id": f"sha256:{digest}",
                    "canonical_path": relative,
                    "group_status": "provisional_include",
                    "split": split,
                    "exclusion_reason": "",
                })
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=preparation.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_category_mapping_is_explicit_and_provisional(category_modules):
    _, module = category_modules
    assert module.CLASS_NAMES == ("paper", "cardboard", "plastic", "glass", "organic", "battery")
    assert module.CATEGORY_TO_FAMILY["battery"] == "E_WASTE"
    assert set(module.CATEGORY_TO_FAMILY) == set(module.CLASS_NAMES)
    assert module.PROVENANCE == "folder_derived_provisional"


def test_category_manifest_reuses_deterministic_splits_and_all_classes(category_modules, tmp_path):
    preparation, module = category_modules
    source = tmp_path / "family.csv"
    _source_manifest(preparation, source)
    first_rows, first_images = module.build_category_manifest(source)
    second_rows, second_images = module.build_category_manifest(source)
    assert first_rows == second_rows
    assert first_images == second_images
    counts = module._validate_class_counts(first_images)
    assert all(counts[split][category] == 2 for split in ("train", "validation") for category in module.CLASS_NAMES)
    test_records = module._test_records_after_selection(first_rows)
    assert all(sum(record.category == category for record in test_records) == 2 for category in module.CLASS_NAMES)
    assert all(row["label_provenance"] == module.PROVENANCE for row in first_rows)


def test_category_manifest_excludes_cross_category_hash_conflicts(category_modules, tmp_path):
    preparation, module = category_modules
    source = tmp_path / "family.csv"
    rows = _source_manifest(preparation, source)
    digest = hashlib.sha256(b"cross-category conflict").hexdigest()
    for category, family in (("paper", "MSW"), ("battery", "E_WASTE")):
        relative = f"{category}/conflict.png"
        rows.append({
            "source_relative_path": relative, "sha256": digest, "source_folder": category,
            "provisional_family": family, "label_provenance": preparation.PROVENANCE,
            "source_folder_mapping": f"{category} -> {family}", "human_review_status": "pending",
            "human_reviewed_family": "", "human_review_reason": "",
            "duplicate_group_id": f"sha256:{digest}", "canonical_path": relative,
            "group_status": "provisional_include", "split": "train", "exclusion_reason": "",
        })
    with source.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=preparation.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest_rows, images = module.build_category_manifest(source)
    assert digest not in {image.sha256 for image in images}
    conflicts = [row for row in manifest_rows if row["sha256"] == digest]
    assert len(conflicts) == 2
    assert all(row["group_status"] == "provisional_excluded" and not row["split"] for row in conflicts)
    assert all("conflict" in row["exclusion_reason"] for row in conflicts)


def test_category_manifest_ignores_unsupported_source_folders(category_modules, tmp_path):
    preparation, module = category_modules
    source = tmp_path / "family.csv"
    rows = _source_manifest(preparation, source)
    rows.append({
        "source_relative_path": "metal/x.png", "sha256": hashlib.sha256(b"x").hexdigest(),
        "source_folder": "metal", "provisional_family": "", "label_provenance": preparation.PROVENANCE,
        "source_folder_mapping": "", "human_review_status": "pending", "human_reviewed_family": "",
        "human_review_reason": "", "duplicate_group_id": "", "canonical_path": "",
        "group_status": "provisional_excluded", "split": "", "exclusion_reason": "unmapped",
    })
    with source.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=preparation.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest_rows, _ = module.build_category_manifest(source)
    assert all(row["source_folder"] != "metal" for row in manifest_rows)


def test_conflict_with_unsupported_folder_excludes_supported_category_hash(category_modules, tmp_path):
    preparation, module = category_modules
    source = tmp_path / "family.csv"
    rows = _source_manifest(preparation, source)
    digest = hashlib.sha256(b"metal-paper exact duplicate").hexdigest()
    rows.extend([
        {
            "source_relative_path": "paper/conflict.png", "sha256": digest, "source_folder": "paper",
            "provisional_family": "MSW", "label_provenance": preparation.PROVENANCE,
            "source_folder_mapping": "paper -> MSW", "human_review_status": "pending",
            "human_reviewed_family": "", "human_review_reason": "", "duplicate_group_id": digest,
            "canonical_path": "paper/conflict.png", "group_status": "provisional_include",
            "split": "train", "exclusion_reason": "",
        },
        {
            "source_relative_path": "metal/conflict.png", "sha256": digest, "source_folder": "metal",
            "provisional_family": "", "label_provenance": preparation.PROVENANCE,
            "source_folder_mapping": "", "human_review_status": "pending",
            "human_reviewed_family": "", "human_review_reason": "", "duplicate_group_id": digest,
            "canonical_path": "", "group_status": "provisional_excluded",
            "split": "", "exclusion_reason": "unmapped source folder",
        },
    ])
    with source.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=preparation.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest_rows, images = module.build_category_manifest(source)
    assert digest not in {image.sha256 for image in images}
    paper_conflict = next(row for row in manifest_rows if row["source_relative_path"] == "paper/conflict.png")
    assert paper_conflict["group_status"] == "provisional_excluded"
    assert "cross-category exact-hash conflict" in paper_conflict["exclusion_reason"]


def test_category_training_rejects_existing_output_directory(category_modules, tmp_path, monkeypatch):
    _, module = category_modules
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(module, "OUTPUT_DIR", output.resolve())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.train_category_classifier(tmp_path / "missing.csv", tmp_path, output)


def test_test_categories_are_validated_only_after_selection(category_modules, tmp_path):
    preparation, module = category_modules
    source = tmp_path / "family.csv"
    _source_manifest(preparation, source)
    rows, images = module.build_category_manifest(source)
    test_row = next(row for row in rows if row["split"] == "test" and row["group_status"] == "provisional_include")
    test_row["provisional_category"] = "metal"
    train_validation = [image for image in images if image.split != "test"]
    assert module._validate_class_counts(train_validation)
    with pytest.raises(module.CategoryTrainingError, match="Invalid provisional test row"):
        module._test_records_after_selection(rows)
