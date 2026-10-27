import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def provisional_module(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project / "scripts"))
    import prepare_provisional_family
    return prepare_provisional_family


def _image_row(module, root: Path, folder: str, name: str, color):
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_relative_path": f"{folder}/{name}",
        "sha256": digest,
        "source_folder": folder,
        "proposed_review_route": module.workflow.REVIEW_ROUTES[folder],
        "reviewed_family": "",
        "review_status": "pending",
        "duplicate_group_id": f"sha256:{digest}",
        "inventory_error": "",
    }


def _usable_inventory(module, root):
    rows = []
    for folder_index, folder in enumerate(("paper", "cardboard", "plastic", "glass", "organic", "battery")):
        for index in range(4):
            rows.append(_image_row(
                module, root, folder, f"item_{index}.png",
                (folder_index * 27 + index + 1, index * 31 + 2, folder_index * 19 + 3),
            ))
    return rows


def test_provisional_mapping_is_explicit_configurable_and_rejects_unsupported_folders(
    provisional_module,
):
    module = provisional_module
    assert module.parse_folder_mapping() == {
        "paper": "MSW", "cardboard": "MSW", "plastic": "MSW",
        "glass": "MSW", "organic": "MSW", "battery": "E_WASTE",
    }
    assert module.parse_folder_mapping(("paper", "plastic"), ("battery",)) == {
        "paper": "MSW", "plastic": "MSW", "battery": "E_WASTE",
    }
    with pytest.raises(module.ProvisionalDataError, match="metal.*excluded"):
        module.parse_folder_mapping(("paper", "metal"), ("battery",))


def test_provisional_manifest_is_deterministic_and_prevents_duplicate_leakage(
    tmp_path, provisional_module,
):
    module = provisional_module
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = _usable_inventory(module, root)
    # Same-folder duplicate gets one canonical row; cross-folder duplicate is excluded.
    source = root / "paper" / "item_0.png"
    duplicate = root / "paper" / "exact_copy.png"
    duplicate.write_bytes(source.read_bytes())
    duplicate_digest = hashlib.sha256(duplicate.read_bytes()).hexdigest()
    inventory.append({
        **inventory[0], "source_relative_path": "paper/exact_copy.png",
        "sha256": duplicate_digest, "duplicate_group_id": f"sha256:{duplicate_digest}",
    })
    cross_folder_path = root / "battery" / "cross_folder.png"
    cross_folder_path.write_bytes(source.read_bytes())
    inventory.append({
        **inventory[0],
        "source_relative_path": "battery/cross_folder.png",
        "source_folder": "battery",
        "proposed_review_route": module.workflow.REVIEW_ROUTES["battery"],
        "sha256": duplicate_digest,
        "duplicate_group_id": f"sha256:{duplicate_digest}",
    })

    first = module.build_provisional_manifest(inventory, root, seed=17, max_per_folder=4)
    second = module.build_provisional_manifest(inventory, root, seed=17, max_per_folder=4)
    assert first == second
    included = [row for row in first if row["group_status"] == "provisional_include"]
    assert {row["label_provenance"] for row in included} == {module.PROVENANCE}
    assert all(row["human_review_status"] == "pending" for row in included)
    split_by_hash = {}
    for row in included:
        split_by_hash.setdefault(row["sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in split_by_hash.values())
    assert all(
        row["group_status"] == "excluded"
        and "spans source folders" in row["exclusion_reason"]
        for row in first
        if row["sha256"] == duplicate_digest
    )
    counts = {family: {split: 0 for split in module.SPLITS} for family in ("MSW", "E_WASTE")}
    for row in included:
        counts[row["provisional_family"]][row["split"]] += 1
    assert all(counts[family][split] for family in counts for split in module.SPLITS)


def test_provisional_manifest_respects_holds_and_excludes_currently_unreadable_sources(
    tmp_path, provisional_module,
):
    module = provisional_module
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = _usable_inventory(module, root)
    held_path = inventory[0]["source_relative_path"]
    held = module.build_provisional_manifest(
        inventory, root,
        decisions={held_path: {"review_status": "hold", "reviewed_family": "", "reason": "unclear"}},
        max_per_folder=4,
    )
    assert next(row for row in held if row["source_relative_path"] == held_path)["split"] == ""
    assert "HOLD" in next(row for row in held if row["source_relative_path"] == held_path)["exclusion_reason"]

    changed = root / inventory[1]["source_relative_path"]
    changed.write_bytes(b"changed after inventory")
    manifest = module.build_provisional_manifest(inventory, root, max_per_folder=4)
    changed_row = next(row for row in manifest if row["source_relative_path"] == inventory[1]["source_relative_path"])
    assert changed_row["split"] == ""
    assert "SHA-256 differs" in changed_row["exclusion_reason"]


def test_insufficient_provisional_class_fails_before_manifest_output(tmp_path, provisional_module):
    module = provisional_module
    root = tmp_path / "dataset"
    root.mkdir()
    inventory = []
    for index in range(4):
        inventory.append(_image_row(module, root, "paper", f"p{index}.png", (index + 1, 0, 0)))
    inventory.append(_image_row(module, root, "battery", "one.png", (0, 0, 255)))
    with pytest.raises(module.ProvisionalDataError, match="Insufficient distinct eligible hashes for E_WASTE"):
        module.build_provisional_manifest(inventory, root, max_per_folder=None)


def test_manifest_output_refuses_protected_and_existing_files(tmp_path, provisional_module, monkeypatch):
    module = provisional_module
    root = tmp_path / "dataset"
    root.mkdir()
    protected = tmp_path / "family_review_decisions.csv"
    monkeypatch.setattr(module.workflow, "DECISIONS_PATH", protected)
    with pytest.raises(module.ProvisionalDataError, match="protected"):
        module.write_manifest(protected, [], root)
    existing = tmp_path / "already.csv"
    existing.touch()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.write_manifest(existing, [], root)
