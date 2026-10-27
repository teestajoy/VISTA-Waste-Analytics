import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "test_batch_review.py"
SPEC = importlib.util.spec_from_file_location("vista_test_batch_review", MODULE_PATH)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def _make_queue(tmp_path, image_paths=("cardboard/box.png", "battery/cell.png")):
    dataset = tmp_path / "dataset"
    batch_path = tmp_path / "batch.csv"
    rows = []
    for relative in image_paths:
        image_path = dataset / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), "green").save(image_path, format="PNG")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        rows.append({
            "source_relative_path": relative,
            "sha256": digest,
            "source_folder": relative.split("/")[0],
            "split": "test",
            "selection_source": "test",
            "review_status": "",
            "reviewed_family": "",
            "decision": "",
            "reviewed_label": "",
            "reason": "",
        })
    with batch_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return dataset, batch_path, review.load_batch(batch_path)


def test_load_batch_checks_test_paths_and_blank_decisions(tmp_path):
    _, batch_path, rows = _make_queue(tmp_path)
    assert len(rows) == 2
    assert all(row["split"] == "test" for row in rows)
    batch_path.write_text(
        "source_relative_path,sha256,source_folder,split,reviewed_label\n"
        "../outside.png," + "0" * 64 + ",..,test,MSW\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsafe source-relative"):
        review.load_batch(batch_path)


def test_decisions_persist_separately_and_require_confirmation_to_replace(tmp_path):
    dataset, batch_path, rows = _make_queue(tmp_path)
    output = tmp_path / "decisions.csv"
    original_batch = batch_path.read_bytes()
    decisions = review.load_decisions(output, rows)

    decisions = review.save_decision(
        output, batch_path, dataset, rows, decisions,
        "cardboard/box.png", "category", "cardboard",
    )
    decisions = review.save_decision(
        output, batch_path, dataset, rows, decisions,
        "battery/cell.png", "family", "E_WASTE",
    )

    loaded = review.load_decisions(output, rows)
    assert loaded[("cardboard/box.png", "category")]["reviewed_label"] == "cardboard"
    assert loaded[("battery/cell.png", "family")]["reviewed_label"] == "E_WASTE"
    assert batch_path.read_bytes() == original_batch
    with pytest.raises(ValueError, match="confirm replacement"):
        review.save_decision(
            output, batch_path, dataset, rows, loaded,
            "cardboard/box.png", "category", "paper",
        )
    replaced = review.save_decision(
        output, batch_path, dataset, rows, loaded,
        "cardboard/box.png", "category", "paper", confirm_replace=True,
    )
    assert replaced[("cardboard/box.png", "category")]["reviewed_label"] == "paper"
    assert len(review.load_decisions(output, rows)) == 2


def test_category_mode_is_default_for_category_batch_and_saves_alongside_family_decision(tmp_path):
    dataset, batch_path, rows = _make_queue(tmp_path)
    category_batch = tmp_path / "reviewed_test_category_review_batch.csv"
    category_batch.write_bytes(batch_path.read_bytes())
    output = tmp_path / "decisions.csv"
    decisions = review.load_decisions(output, rows)

    assert review.default_review_type(category_batch) == "category"
    assert review.default_review_type(batch_path) == "family"

    relative = rows[0]["source_relative_path"]
    decisions = review.save_decision(
        output, category_batch, dataset, rows, decisions,
        relative, "family", "MSW",
    )
    decisions = review.save_decision(
        output, category_batch, dataset, rows, decisions,
        relative, "category", "cardboard",
    )
    reloaded = review.load_decisions(output, rows)

    assert reloaded[(relative, "family")]["reviewed_label"] == "MSW"
    assert reloaded[(relative, "category")]["review_type"] == "category"
    assert reloaded[(relative, "category")]["reviewed_label"] == "cardboard"
    assert review.progress_counts(rows, reloaded, "category")["included"] == 1
    assert review.progress_counts(rows, reloaded, "family")["included"] == 1


def test_blind_review_identity_hides_folder_and_label_bearing_filename(tmp_path):
    row = {
        "source_relative_path": "plastic/transparent_bottle.jpg",
        "source_folder": "plastic",
        "sha256": "a" * 64,
    }

    title, identity, image_caption = review.review_display_identity(row, 2, blind=True)

    assert title == "Review image 3"
    assert identity == "Blind review ID: " + "a" * 16
    assert "plastic" not in f"{title} {identity} {image_caption}".lower()
    assert "transparent_bottle.jpg" not in f"{title} {identity} {image_caption}"
    assert review.review_display_identity(row, 2, blind=False)[0] == "transparent_bottle.jpg"
    confirmation = {"path": row["source_relative_path"], "image_id": "a" * 16}
    assert review.review_confirmation_target(confirmation, blind=True) == "review image ID " + "a" * 16
    assert review.review_confirmation_target(confirmation, blind=False) == row["source_relative_path"]
    assert "plastic" not in review.review_error_message(ValueError(row["source_relative_path"]), blind=True)


def test_hold_is_explicit_and_reasoned_but_skip_is_not_a_saved_decision(tmp_path):
    dataset, batch_path, rows = _make_queue(tmp_path)
    output = tmp_path / "decisions.csv"
    decisions = review.load_decisions(output, rows)
    decisions = review.save_decision(
        output, batch_path, dataset, rows, decisions,
        "cardboard/box.png", "category", "HOLD",
    )

    saved = decisions[("cardboard/box.png", "category")]
    assert saved["decision"] == "hold"
    assert saved["reviewed_label"] == ""
    assert saved["reason"] == review.HOLD_REASON
    assert review.shortcut_action("s", "category") == "skip"
    assert ("battery/cell.png", "category") not in decisions
    assert review.shortcut_action("h", "family") == "hold"


def test_progress_and_resume_count_only_the_selected_review_type(tmp_path):
    _, _, rows = _make_queue(tmp_path)
    decisions = {
        ("cardboard/box.png", "category"): {"decision": "include"},
        ("battery/cell.png", "category"): {"decision": "hold"},
        ("battery/cell.png", "family"): {"decision": "include"},
    }
    assert review.progress_counts(rows, decisions, "category") == {
        "reviewed": 2, "included": 1, "held": 1, "remaining": 0,
    }
    assert review.progress_counts(rows, decisions, "family")["remaining"] == 1
    assert review.next_pending_index(rows, decisions, "family", -1) == 0
    assert review.next_pending_index(rows, decisions, "category", -1) is None


def test_shortcut_mapping_is_task_specific_and_navigation_is_available():
    assert review.shortcut_action("1", "family") == "MSW"
    assert review.shortcut_action("2", "family") == "E_WASTE"
    assert review.shortcut_action("1", "category") == "paper"
    assert review.shortcut_action("6", "category") == "battery"
    assert review.shortcut_action("ArrowLeft", "family") == "previous"
    assert review.shortcut_action("ArrowRight", "category") == "next"


def test_symlink_escape_and_protected_output_are_rejected(tmp_path):
    dataset, batch_path, rows = _make_queue(tmp_path, ("cardboard/link.png",))
    outside = tmp_path / "outside.png"
    Image.new("RGB", (3, 3), "red").save(outside)
    link = dataset / "cardboard" / "link.png"
    try:
        link.unlink()
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment.")

    with pytest.raises(ValueError, match="outside the dataset root"):
        review.resolve_source_image(dataset, "cardboard/link.png")
    with pytest.raises(ValueError, match="separate output"):
        review._validate_output_path(batch_path, batch_path, dataset)
