import csv
import hashlib
import sys
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def batch_modules(monkeypatch):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    import create_msw_review_batch
    import family_dataset_workflow
    import generate_family_candidates

    return create_msw_review_batch, family_dataset_workflow, generate_family_candidates


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (20, 15), color)
    image.save(path, format="PNG")
    image.close()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def batch_data(tmp_path, batch_modules):
    batch, workflow, candidates_module = batch_modules
    dataset = tmp_path / "dataset"
    for folder in workflow.SOURCE_FOLDERS:
        (dataset / folder).mkdir(parents=True, exist_ok=True)
    for index in range(24):
        _write_image(dataset / "paper" / f"paper_{index:02}.png", (index + 1, 20, 30))
        _write_image(dataset / "cardboard" / f"cardboard_{index:02}.png", (30, index + 1, 20))
        _write_image(dataset / "plastic" / f"plastic_{index:02}.png", (20, 30, index + 1))
    # Same-folder duplicate copies are not separate candidates.
    (dataset / "paper" / "paper_00_copy.png").write_bytes(
        (dataset / "paper" / "paper_00.png").read_bytes()
    )
    # Cross-folder identical bytes are a conflict and must not enter the batch.
    (dataset / "battery" / "same_as_paper.png").write_bytes(
        (dataset / "paper" / "paper_01.png").read_bytes()
    )

    inventory_path = tmp_path / "inventory.csv"
    inventory = workflow.write_inventory(dataset, inventory_path)
    queue_path = tmp_path / "original_queue.csv"
    decisions_path = tmp_path / "original_decisions.csv"
    workflow.create_review_queue(
        inventory_path, dataset, per_folder=40, queue_path=queue_path,
        decisions_path=decisions_path, sheets_dir=tmp_path / "original_sheets",
    )
    queue = workflow._load_queue(queue_path, inventory)
    decisions = workflow._load_decisions(decisions_path, inventory, queue)
    candidate_rows = candidates_module.build_candidate_rows(inventory, queue, decisions)
    candidate_audit_path = tmp_path / "candidate_audit.csv"
    _write_csv(candidate_audit_path, candidates_module.FIELDS, candidate_rows)
    return {
        "batch": batch,
        "workflow": workflow,
        "dataset": dataset,
        "inventory_path": inventory_path,
        "inventory": inventory,
        "queue_path": queue_path,
        "queue": queue,
        "decisions_path": decisions_path,
        "candidate_audit_path": candidate_audit_path,
    }


def test_batch_selection_is_deterministic_and_excludes_duplicate_groups(batch_data):
    batch = batch_data["batch"]
    args = (
        batch_data["inventory"],
        batch._load_candidate_audit(batch_data["candidate_audit_path"], batch_data["inventory"]),
        batch_data["queue"],
        {},
        batch_data["dataset"],
    )
    first = batch.select_review_batch(*args, per_folder=7, seed=73)
    second = batch.select_review_batch(*args, per_folder=7, seed=73)
    assert [row["source_relative_path"] for row in first] == [
        row["source_relative_path"] for row in second
    ]
    assert {row["source_folder"] for row in first} == {"paper", "cardboard", "plastic"}
    assert all(sum(row["source_folder"] == folder for row in first) == 7
               for folder in ("paper", "cardboard", "plastic"))
    paths = {row["source_relative_path"] for row in first}
    assert "paper/paper_00_copy.png" not in paths
    assert "paper/paper_01.png" not in paths
    assert "battery/same_as_paper.png" not in paths
    assert len({row["sha256"] for row in first}) == len(first)


def test_create_batch_writes_only_separate_pending_outputs_and_sheets(batch_data, tmp_path):
    batch = batch_data["batch"]
    workflow = batch_data["workflow"]
    original_queue = batch_data["queue_path"].read_bytes()
    original_decisions = batch_data["decisions_path"].read_bytes()
    source_hashes = {
        path.relative_to(batch_data["dataset"]).as_posix(): workflow.sha256_file(path)
        for path in batch_data["dataset"].rglob("*.png")
    }
    output_queue = tmp_path / "new" / "msw_batch_queue.csv"
    output_decisions = tmp_path / "new" / "msw_batch_decisions.csv"
    sheets = tmp_path / "new" / "msw_batch_sheets"

    rows = batch.create_batch(
        dataset_root=batch_data["dataset"],
        inventory_path=batch_data["inventory_path"],
        candidate_audit_path=batch_data["candidate_audit_path"],
        existing_queue_path=batch_data["queue_path"],
        existing_decisions_path=batch_data["decisions_path"],
        output_queue_path=output_queue,
        output_decisions_path=output_decisions,
        sheets_dir=sheets,
        per_folder=4,
        seed=9,
    )

    assert len(rows) == 12
    assert {row["source_folder"] for row in rows} == {"paper", "cardboard", "plastic"}
    decisions = _read_csv(output_decisions)
    assert len(decisions) == len(rows)
    assert all(row["review_status"] == "pending" for row in decisions)
    assert all(not row["reviewed_family"] and not row["reason"] for row in decisions)
    assert {path.stem for path in sheets.glob("*.pdf")} == {"paper", "cardboard", "plastic"}
    assert batch_data["queue_path"].read_bytes() == original_queue
    assert batch_data["decisions_path"].read_bytes() == original_decisions
    assert source_hashes == {
        path.relative_to(batch_data["dataset"]).as_posix(): workflow.sha256_file(path)
        for path in batch_data["dataset"].rglob("*.png")
    }


def test_batch_creation_refuses_to_overwrite_any_existing_output(batch_data, tmp_path):
    batch = batch_data["batch"]
    existing_queue = tmp_path / "already_exists.csv"
    existing_queue.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        batch.create_batch(
            dataset_root=batch_data["dataset"],
            inventory_path=batch_data["inventory_path"],
            candidate_audit_path=batch_data["candidate_audit_path"],
            existing_queue_path=batch_data["queue_path"],
            existing_decisions_path=batch_data["decisions_path"],
            output_queue_path=existing_queue,
            output_decisions_path=tmp_path / "new_decisions.csv",
            sheets_dir=tmp_path / "new_sheets",
        )
    assert existing_queue.read_text(encoding="utf-8") == "keep"


def test_candidate_audit_mismatch_fails_before_output(batch_data, tmp_path):
    batch = batch_data["batch"]
    rows = _read_csv(batch_data["candidate_audit_path"])
    rows[0]["sha256"] = "0" * 64
    _write_csv(batch_data["candidate_audit_path"], tuple(rows[0].keys()), rows)
    output_queue = tmp_path / "queue.csv"

    with pytest.raises(ValueError, match="is stale for"):
        batch.create_batch(
            dataset_root=batch_data["dataset"],
            inventory_path=batch_data["inventory_path"],
            candidate_audit_path=batch_data["candidate_audit_path"],
            existing_queue_path=batch_data["queue_path"],
            existing_decisions_path=batch_data["decisions_path"],
            output_queue_path=output_queue,
            output_decisions_path=tmp_path / "decisions.csv",
            sheets_dir=tmp_path / "sheets",
        )
    assert not output_queue.exists()


def test_corrupt_but_rehashed_candidate_fails_image_validation(batch_data, tmp_path):
    batch = batch_data["batch"]
    workflow = batch_data["workflow"]
    corrupt_relative = "paper/paper_00.png"
    corrupt_path = batch_data["dataset"] / corrupt_relative
    corrupt_path.write_bytes(b"not a decodable image")
    new_hash = hashlib.sha256(corrupt_path.read_bytes()).hexdigest()

    inventory_rows = _read_csv(batch_data["inventory_path"])
    inventory_row = next(row for row in inventory_rows if row["source_relative_path"] == corrupt_relative)
    inventory_row.update(sha256=new_hash, duplicate_group_id=f"sha256:{new_hash}")
    _write_csv(batch_data["inventory_path"], workflow.INVENTORY_FIELDS, inventory_rows)

    queue_rows = _read_csv(batch_data["queue_path"])
    queue_row = next(row for row in queue_rows if row["source_relative_path"] == corrupt_relative)
    queue_row.update(sha256=new_hash, duplicate_group_id=f"sha256:{new_hash}")
    _write_csv(batch_data["queue_path"], workflow.QUEUE_FIELDS, queue_rows)
    inventory = workflow._load_inventory(batch_data["inventory_path"])
    queue = workflow._load_queue(batch_data["queue_path"], inventory)
    decisions = workflow._load_decisions(batch_data["decisions_path"], inventory, queue)
    candidates_module = __import__("generate_family_candidates")
    candidate_rows = candidates_module.build_candidate_rows(inventory, queue, decisions)
    _write_csv(
        batch_data["candidate_audit_path"], candidates_module.FIELDS, candidate_rows
    )
    output_queue = tmp_path / "must_not_be_created.csv"

    with pytest.raises(ValueError, match="Review candidate is unreadable"):
        batch.create_batch(
            dataset_root=batch_data["dataset"],
            inventory_path=batch_data["inventory_path"],
            candidate_audit_path=batch_data["candidate_audit_path"],
            existing_queue_path=batch_data["queue_path"],
            existing_decisions_path=batch_data["decisions_path"],
            output_queue_path=output_queue,
            output_decisions_path=tmp_path / "must_not_be_created_decisions.csv",
            sheets_dir=tmp_path / "must_not_be_created_sheets",
            per_folder=24,
        )
    assert not output_queue.exists()
