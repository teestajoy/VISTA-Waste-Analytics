import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "create_reviewed_test_review_batch.py"
SPEC = importlib.util.spec_from_file_location("create_reviewed_test_review_batch", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def _make_image(root: Path, relative: str, color=(45, 120, 90)):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color).save(path, format="PNG")
    return path


def _manifest(relative, digest, split="test", canonical=None, status="provisional_include"):
    return {
        "source_relative_path": relative,
        "sha256": digest,
        "source_folder": relative.split("/", 1)[0],
        "canonical_path": canonical or relative,
        "group_status": status,
        "split": split,
    }


def _queue(relative, digest):
    return {"source_relative_path": relative, "sha256": digest}


def test_selection_is_reproducible_balanced_and_leaves_decisions_blank(tmp_path):
    files = {
        "battery/queued.png": _make_image(tmp_path, "battery/queued.png"),
        "battery/supplement.png": _make_image(tmp_path, "battery/supplement.png", (80, 30, 90)),
        "cardboard/review.png": _make_image(tmp_path, "cardboard/review.png", (140, 90, 40)),
        "paper/train.png": _make_image(tmp_path, "paper/train.png", (220, 220, 220)),
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    family_manifest = [
        _manifest("battery/queued.png", hashes["battery/queued.png"]),
        _manifest("battery/supplement.png", hashes["battery/supplement.png"]),
        _manifest("cardboard/review.png", hashes["cardboard/review.png"]),
        _manifest("paper/train.png", hashes["paper/train.png"], split="train"),
    ]
    category_manifest = [dict(row) for row in family_manifest]
    kwargs = (
        tmp_path,
        [_queue("battery/queued.png", hashes["battery/queued.png"])],
        [{"source_relative_path": "battery/queued.png", "review_status": "pending"}],
        [], [], [], family_manifest, category_manifest,
    )

    first, _ = batch.select_candidates(*kwargs, per_folder=2, seed=17)
    second, _ = batch.select_candidates(*kwargs, per_folder=2, seed=17)

    assert [row["source_relative_path"] for row in first] == [
        row["source_relative_path"] for row in second
    ]
    assert {row["source_folder"] for row in first} == {"battery", "cardboard"}
    battery_rows = [row for row in first if row["source_folder"] == "battery"]
    assert len(battery_rows) == 2
    assert battery_rows[0]["selection_source"] == "family review queue"
    assert battery_rows[1]["selection_source"] == "held-out test manifest supplement"
    for row in first:
        assert row["split"] == "test"
        assert row["review_status"] == ""
        assert row["reviewed_family"] == ""
        assert row["decision"] == ""
        assert row["reviewed_label"] == ""
        assert row["reason"] == ""
    assert "paper/train.png" not in {row["source_relative_path"] for row in first}


def test_duplicate_hashes_at_distinct_canonical_test_paths_are_excluded(tmp_path):
    first = _make_image(tmp_path, "battery/one.png")
    second = _make_image(tmp_path, "cardboard/two.png")
    second.write_bytes(first.read_bytes())
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    manifests = [_manifest("battery/one.png", digest), _manifest("cardboard/two.png", digest)]

    selected, exclusions = batch.select_candidates(
        tmp_path, [], [], [], [], [], manifests, [dict(row) for row in manifests], per_folder=1
    )

    assert selected == []
    assert exclusions["hash is shared by multiple canonical test paths"] >= 1


def test_hash_in_train_or_validation_excludes_test_candidate(tmp_path):
    test_file = _make_image(tmp_path, "plastic/test.png")
    train_file = _make_image(tmp_path, "plastic/train.png")
    train_file.write_bytes(test_file.read_bytes())
    digest = hashlib.sha256(test_file.read_bytes()).hexdigest()
    family_manifest = [
        _manifest("plastic/test.png", digest),
        _manifest("plastic/train.png", digest, split="train"),
    ]

    selected, exclusions = batch.select_candidates(
        tmp_path, [], [], [], [], [], family_manifest, [dict(row) for row in family_manifest]
    )

    assert selected == []
    assert exclusions["hash appears in training or validation data"] >= 1


def test_held_or_previously_decided_images_are_not_selected(tmp_path):
    held = _make_image(tmp_path, "battery/held.png")
    previously_included = _make_image(tmp_path, "cardboard/done.png")
    held_hash = hashlib.sha256(held.read_bytes()).hexdigest()
    included_hash = hashlib.sha256(previously_included.read_bytes()).hexdigest()
    manifests = [
        _manifest("battery/held.png", held_hash),
        _manifest("cardboard/done.png", included_hash),
    ]

    selected, exclusions = batch.select_candidates(
        tmp_path,
        [_queue("battery/held.png", held_hash), _queue("cardboard/done.png", included_hash)],
        [{"source_relative_path": "battery/held.png", "review_status": "pending"}],
        [], [],
        [{"source_relative_path": "battery/held.png", "decision": "hold"},
         {"source_relative_path": "cardboard/done.png", "decision": "include"}],
        manifests, [dict(row) for row in manifests],
    )

    assert selected == []
    assert exclusions["already has an explicit human decision"] == 2


def test_paths_in_previous_pilot_test_decisions_are_not_selected_again(tmp_path):
    reviewed = _make_image(tmp_path, "paper/reviewed.png")
    fresh = _make_image(tmp_path, "paper/fresh.png", (200, 80, 30))
    reviewed_hash = hashlib.sha256(reviewed.read_bytes()).hexdigest()
    fresh_hash = hashlib.sha256(fresh.read_bytes()).hexdigest()
    manifests = [
        _manifest("paper/reviewed.png", reviewed_hash),
        _manifest("paper/fresh.png", fresh_hash),
    ]

    selected, exclusions = batch.select_candidates(
        tmp_path, [], [], [], [], [], manifests, [dict(row) for row in manifests],
        per_folder=2,
        test_batch_decisions=[{
            "source_relative_path": "paper/reviewed.png",
            "review_type": "family",
            "decision": "include",
        }],
    )

    assert [row["source_relative_path"] for row in selected] == ["paper/fresh.png"]
    assert exclusions["already has an explicit human decision"] == 1


def test_multiple_prior_test_batches_are_excluded_from_new_candidates(tmp_path):
    first = _make_image(tmp_path, "glass/first.png")
    second = _make_image(tmp_path, "glass/second.png", (40, 80, 130))
    fresh = _make_image(tmp_path, "glass/fresh.png", (180, 30, 60))
    paths = ("glass/first.png", "glass/second.png", "glass/fresh.png")
    digests = {path: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() for path in paths}
    manifests = [_manifest(path, digests[path]) for path in paths]

    selected, _ = batch.select_candidates(
        tmp_path, [], [], [], [], [], manifests, [dict(row) for row in manifests],
        per_folder=5,
        test_batch_decisions=[{
            "source_relative_path": "glass/first.png", "decision": "include",
        }, {
            "source_relative_path": "glass/second.png", "decision": "hold",
        }],
    )

    assert [row["source_relative_path"] for row in selected] == ["glass/fresh.png"]


def test_written_batch_has_blank_review_fields_and_contact_sheet(tmp_path):
    source = _make_image(tmp_path, "paper/example.png")
    row = {
        "source_relative_path": "paper/example.png",
        "source_folder": "paper",
        "split": "test",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection_source": "family review queue",
        "_image_path": str(source),
    }
    csv_output = tmp_path / "batch.csv"
    sheet_output = tmp_path / "batch.jpg"

    batch.write_batch([row], csv_output, sheet_output)

    saved = next(csv.DictReader(csv_output.open("r", encoding="utf-8-sig", newline="")))
    assert saved["review_status"] == saved["reviewed_family"] == ""
    assert saved["decision"] == saved["reviewed_label"] == saved["reason"] == ""
    assert Image.open(sheet_output).size[0] > 0
    with pytest.raises(FileExistsError, match="Refusing to overwrite|already exists"):
        batch.write_batch([row], csv_output, sheet_output)
