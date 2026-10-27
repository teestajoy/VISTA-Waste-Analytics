import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def review_module(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root))
    from dashboard import family_review

    monkeypatch.setattr(family_review, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        family_review, "AUDIT_PATH",
        tmp_path / "data" / "processed" / "family_review_audit.csv",
    )

    return family_review


def _make_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 8), color)
    image.save(path, format="PNG")
    image.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _queue_row(family_review, root: Path, folder: str, filename: str, color):
    relative = f"{folder}/{filename}"
    digest = _make_image(root / relative, color)
    return {
        "source_relative_path": relative,
        "sha256": digest,
        "source_folder": folder,
        "proposed_review_route": family_review.family_workflow.REVIEW_ROUTES[folder],
        "duplicate_group_id": f"sha256:{digest}",
        "inventory_error": "",
    }


def _write_queue(path: Path, family_review, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=family_review.family_workflow.QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


@pytest.fixture
def review_setup(tmp_path, monkeypatch, review_module):
    root = tmp_path / "dataset"
    root.mkdir()
    rows = [
        _queue_row(review_module, root, "paper", "paper.png", (220, 30, 10)),
        _queue_row(review_module, root, "cardboard", "box.png", (10, 120, 30)),
    ]
    queue_path = tmp_path / "family_review_queue.csv"
    _write_queue(queue_path, review_module, rows)
    decisions_path = tmp_path / "family_review_decisions.csv"
    monkeypatch.setattr(review_module, "DECISIONS_PATH", decisions_path)
    return root, rows, queue_path, decisions_path


def test_queue_loading_validates_schema_and_safe_paths(review_setup, review_module):
    root, rows, queue_path, _ = review_setup
    assert review_module.load_review_queue(queue_path, root) == rows

    rows[0]["source_relative_path"] = "../outside.png"
    _write_queue(queue_path, review_module, rows)
    with pytest.raises(ValueError, match="Unsafe source_relative_path"):
        review_module.load_review_queue(queue_path, root)


def test_missing_or_unreadable_images_are_reported_safely(review_setup, review_module):
    root, rows, _, decisions_path = review_setup
    (root / "paper" / "paper.png").unlink()
    with pytest.raises(FileNotFoundError, match="Review image is missing"):
        review_module.load_review_image(root, rows[0]["source_relative_path"])

    (root / "paper" / "paper.png").write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read review image"):
        review_module.load_review_image(root, rows[0]["source_relative_path"])
    corrupt_hash = hashlib.sha256((root / "paper" / "paper.png").read_bytes()).hexdigest()
    rows[0]["sha256"] = corrupt_hash
    rows[0]["duplicate_group_id"] = f"sha256:{corrupt_hash}"
    with pytest.raises(ValueError, match="Could not read review image"):
        review_module.save_review_decision(
            rows, decisions_path, rows[0]["source_relative_path"], "MSW",
            dataset_root=root,
        )
    assert not decisions_path.exists()


def test_image_hash_must_match_the_review_queue(review_setup, review_module):
    root, rows, _, _ = review_setup
    (root / "paper" / "paper.png").write_bytes(b"different source bytes")
    with pytest.raises(ValueError, match="Image changed since the review queue"):
        review_module.load_review_image(
            root, rows[0]["source_relative_path"], expected_sha256=rows[0]["sha256"]
        )


def test_save_persists_explicit_choices_and_preserves_previous_decisions(
    review_setup, review_module
):
    root, rows, _, decisions_path = review_setup
    first = review_module.save_review_decision(
        rows, decisions_path, rows[0]["source_relative_path"], "MSW", dataset_root=root
    )
    assert first == {
        "source_relative_path": "paper/paper.png",
        "review_status": "include",
        "reviewed_family": "MSW",
        "reason": "",
    }
    saved = _read_csv(decisions_path)
    assert tuple(saved[0]) == review_module.family_workflow.DECISION_FIELDS
    assert saved[0]["review_status"] == "include"
    assert saved[0]["reviewed_family"] == "MSW"
    assert saved[1]["review_status"] == "pending"

    review_module.save_review_decision(
        rows, decisions_path, rows[1]["source_relative_path"], "HOLD",
        "Material is unclear.", dataset_root=root,
    )
    loaded = review_module.load_saved_decisions(decisions_path, rows)
    assert loaded[rows[0]["source_relative_path"]]["reviewed_family"] == "MSW"
    assert loaded[rows[1]["source_relative_path"]] == {
        "review_status": "hold",
        "reviewed_family": "",
        "reason": "Material is unclear.",
    }


def test_decision_changes_are_appended_to_a_separate_audit_log(
    review_setup, review_module
):
    root, rows, _, decisions_path = review_setup
    path = rows[0]["source_relative_path"]
    review_module.save_review_decision(
        rows, decisions_path, path, "MSW", dataset_root=root
    )
    review_module.save_review_decision(
        rows, decisions_path, path, "EXCLUDE", "Image is not waste.",
        dataset_root=root, confirm_replace=True,
    )

    with review_module.AUDIT_PATH.open(encoding="utf-8-sig", newline="") as source:
        audit_rows = list(csv.DictReader(source))
    assert tuple(audit_rows[0]) == review_module.AUDIT_FIELDS
    assert len(audit_rows) == 2
    assert audit_rows[0]["previous_review_status"] == "pending"
    assert audit_rows[0]["new_reviewed_family"] == "MSW"
    assert audit_rows[1]["previous_review_status"] == "include"
    assert audit_rows[1]["new_review_status"] == "exclude"
    assert audit_rows[1]["new_reason"] == "Image is not waste."


@pytest.mark.parametrize("choice", ["HOLD", "EXCLUDE"])
@pytest.mark.parametrize("reason", ["", "   "])
def test_hold_and_exclude_require_nonempty_reason(
    review_setup, review_module, choice, reason
):
    root, rows, _, decisions_path = review_setup
    with pytest.raises(ValueError, match="requires a reason"):
        review_module.save_review_decision(
            rows, decisions_path, rows[0]["source_relative_path"], choice,
            reason, dataset_root=root,
        )
    assert not decisions_path.exists()


@pytest.mark.parametrize("choice", ["unknown", "include", "battery"])
def test_invalid_choices_are_rejected(review_setup, review_module, choice):
    root, rows, _, decisions_path = review_setup
    with pytest.raises(ValueError, match="Choose one of"):
        review_module.save_review_decision(
            rows, decisions_path, rows[0]["source_relative_path"], choice,
            dataset_root=root,
        )


def test_organic_and_battery_approval_needs_content_confirmation_and_reason(
    tmp_path, review_module, monkeypatch
):
    root = tmp_path / "dataset"
    root.mkdir()
    rows = [
        _queue_row(review_module, root, "organic", "food.png", (90, 130, 20)),
        _queue_row(review_module, root, "battery", "cell.png", (40, 50, 60)),
    ]
    decisions_path = tmp_path / "family_review_decisions.csv"
    monkeypatch.setattr(review_module, "DECISIONS_PATH", decisions_path)
    with pytest.raises(ValueError, match="Confirm visible food waste"):
        review_module.save_review_decision(
            rows, decisions_path, rows[0]["source_relative_path"], "MSW",
            "Looks like food.", dataset_root=root,
        )
    with pytest.raises(ValueError, match="needs a reason"):
        review_module.save_review_decision(
            rows, decisions_path, rows[0]["source_relative_path"], "MSW",
            confirm_content=True, dataset_root=root,
        )
    with pytest.raises(ValueError, match="Confirm visible battery/electronic"):
        review_module.save_review_decision(
            rows, decisions_path, rows[1]["source_relative_path"], "E_WASTE",
            "Battery visible.", dataset_root=root,
        )


def test_existing_decision_requires_confirmation_to_replace(review_setup, review_module):
    root, rows, _, decisions_path = review_setup
    path = rows[0]["source_relative_path"]
    review_module.save_review_decision(rows, decisions_path, path, "MSW", dataset_root=root)

    with pytest.raises(ValueError, match="confirm replacement"):
        review_module.save_review_decision(
            rows, decisions_path, path, "EXCLUDE", "Not a waste item.", dataset_root=root
        )
    review_module.save_review_decision(
        rows, decisions_path, path, "EXCLUDE", "Not a waste item.",
        dataset_root=root, confirm_replace=True,
    )
    loaded = review_module.load_saved_decisions(decisions_path, rows)
    assert loaded[path]["review_status"] == "exclude"
    assert loaded[path]["reason"] == "Not a waste item."


def test_explicit_pending_choice_clears_previous_decision_after_confirmation(
    review_setup, review_module
):
    root, rows, _, decisions_path = review_setup
    path = rows[0]["source_relative_path"]
    review_module.save_review_decision(
        rows, decisions_path, path, "HOLD", "Need closer inspection.", dataset_root=root
    )
    with pytest.raises(ValueError, match="confirm replacement"):
        review_module.save_review_decision(
            rows, decisions_path, path, "PENDING", "Old hold reason", dataset_root=root
        )
    review_module.save_review_decision(
        rows, decisions_path, path, "PENDING", "Old hold reason",
        dataset_root=root, confirm_replace=True,
    )
    assert review_module.load_saved_decisions(decisions_path, rows)[path] == {
        "review_status": "pending", "reviewed_family": "", "reason": "",
    }


def test_invalid_saved_decision_is_rejected(review_setup, review_module):
    _, rows, _, decisions_path = review_setup
    with decisions_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=review_module.family_workflow.DECISION_FIELDS)
        writer.writeheader()
        writer.writerow({
            "source_relative_path": rows[0]["source_relative_path"],
            "review_status": "approved",
            "reviewed_family": "MSW",
            "reason": "",
        })
    with pytest.raises(ValueError, match="must be pending, include, hold, or exclude"):
        review_module.load_saved_decisions(decisions_path, rows)


def test_decision_output_is_confined_to_configured_family_review_csv(
    review_setup, review_module
):
    root, rows, _, decisions_path = review_setup
    with pytest.raises(ValueError, match="only be written to"):
        review_module.save_review_decision(
            rows, decisions_path.with_name("paper_cardboard_pilot_review.csv"),
            rows[0]["source_relative_path"], "MSW", dataset_root=root,
        )


def test_batch_requires_confirmation_and_preserves_unselected_decisions(
    tmp_path, review_module, monkeypatch
):
    root = tmp_path / "dataset"
    root.mkdir()
    rows = [
        _queue_row(review_module, root, "paper", "one.png", (200, 20, 10)),
        _queue_row(review_module, root, "cardboard", "two.png", (10, 200, 20)),
        _queue_row(review_module, root, "glass", "three.png", (10, 20, 200)),
    ]
    decisions_path = tmp_path / "family_review_decisions.csv"
    monkeypatch.setattr(review_module, "DECISIONS_PATH", decisions_path)
    review_module.save_review_decision(
        rows, decisions_path, rows[0]["source_relative_path"], "MSW", dataset_root=root
    )

    preview = review_module.preview_batch_decision(
        rows, decisions_path, [rows[1]["source_relative_path"]], "EXCLUDE",
        "Not a waste item.", dataset_root=root,
    )
    assert preview["count"] == 1
    assert preview["folders"] == {"cardboard": 1}
    assert "excluding each selected image" in preview["policy"]
    assert decisions_path.is_file()  # preview does not write or reset the saved file
    with pytest.raises(ValueError, match="Confirm the batch summary"):
        review_module.save_review_batch(
            rows, decisions_path, preview["paths"], "EXCLUDE", "Not a waste item.",
            dataset_root=root,
        )

    review_module.save_review_batch(
        rows, decisions_path, preview["paths"], "EXCLUDE", "Not a waste item.",
        dataset_root=root, confirmed=True,
    )
    loaded = review_module.load_saved_decisions(decisions_path, rows)
    assert loaded[rows[0]["source_relative_path"]]["reviewed_family"] == "MSW"
    assert loaded[rows[1]["source_relative_path"]]["review_status"] == "exclude"
    assert loaded[rows[2]["source_relative_path"]]["review_status"] == "pending"
    with review_module.AUDIT_PATH.open(encoding="utf-8-sig", newline="") as source:
        audit_rows = list(csv.DictReader(source))
    assert len(audit_rows) == 2
    assert {row["source_relative_path"] for row in audit_rows} == {
        rows[0]["source_relative_path"], rows[1]["source_relative_path"]
    }


def test_batch_refuses_conflicting_cross_folder_duplicates(tmp_path, review_module, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    paper = _queue_row(review_module, root, "paper", "same.png", (130, 30, 10))
    duplicate_path = root / "battery" / "same.png"
    duplicate_path.parent.mkdir(parents=True)
    duplicate_path.write_bytes((root / "paper" / "same.png").read_bytes())
    battery = {
        **paper,
        "source_relative_path": "battery/same.png",
        "source_folder": "battery",
        "proposed_review_route": review_module.family_workflow.REVIEW_ROUTES["battery"],
    }
    rows = [paper, battery]
    decisions_path = tmp_path / "family_review_decisions.csv"
    monkeypatch.setattr(review_module, "DECISIONS_PATH", decisions_path)

    conflicts = review_module.duplicate_conflicts(rows, {})
    assert set(conflicts) == {"paper/same.png", "battery/same.png"}
    assert review_module.suggested_family_for_row(paper, conflicts) == (
        "Conflicting duplicate — manual review"
    )
    with pytest.raises(ValueError, match="conflicting duplicate group"):
        review_module.preview_batch_decision(
            rows, decisions_path, ["paper/same.png"], "MSW", dataset_root=root
        )
    assert not decisions_path.exists()


def test_batch_organic_approval_requires_specific_confirmation_and_reason(
    tmp_path, review_module, monkeypatch
):
    root = tmp_path / "dataset"
    root.mkdir()
    rows = [_queue_row(review_module, root, "organic", "food.png", (150, 100, 20))]
    decisions_path = tmp_path / "family_review_decisions.csv"
    monkeypatch.setattr(review_module, "DECISIONS_PATH", decisions_path)
    with pytest.raises(ValueError, match="Confirm visible food waste"):
        review_module.preview_batch_decision(
            rows, decisions_path, [rows[0]["source_relative_path"]], "MSW",
            "Looks like food.", dataset_root=root,
        )
    preview = review_module.preview_batch_decision(
        rows, decisions_path, [rows[0]["source_relative_path"]], "MSW",
        "Visible banana and food scraps.", dataset_root=root, confirm_content=True,
    )
    assert preview["count"] == 1
    assert not decisions_path.exists()


def test_batch_rejects_unsafe_paths_and_unconfirmed_replacements(review_setup, review_module):
    root, rows, _, decisions_path = review_setup
    with pytest.raises(ValueError, match="Unsafe source_relative_path"):
        review_module.preview_batch_decision(
            rows, decisions_path, ["../outside.png"], "HOLD", "Unclear.", dataset_root=root
        )
    path = rows[0]["source_relative_path"]
    review_module.save_review_decision(rows, decisions_path, path, "MSW", dataset_root=root)
    with pytest.raises(ValueError, match="confirm replacement"):
        review_module.preview_batch_decision(
            rows, decisions_path, [path], "EXCLUDE", "Not waste.", dataset_root=root
        )


def test_suggestions_are_not_written_as_human_decisions(review_setup, review_module):
    _, rows, _, decisions_path = review_setup
    decisions = review_module.load_saved_decisions(decisions_path, rows)
    assert review_module.suggested_family_for_row(rows[0], {}) == "MSW candidate"
    assert decisions == {}
    assert review_module.review_summary_rows(rows, decisions) == [
        {"source_folder": "cardboard", "decision": "PENDING", "count": 1},
        {"source_folder": "paper", "decision": "PENDING", "count": 1},
    ]


def test_bulk_select_only_returns_pending_canonical_candidates_without_saving(
    review_setup, review_module
):
    root, rows, _, decisions_path = review_setup
    # Add an exact same-folder copy. It should not be suggested twice in bulk review.
    duplicate = {
        **rows[0],
        "source_relative_path": "paper/paper_copy.png",
    }
    duplicate_path = root / duplicate["source_relative_path"]
    duplicate_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_path.write_bytes((root / rows[0]["source_relative_path"]).read_bytes())
    rows_with_duplicate = [*rows, duplicate]

    selected = review_module.batch_selectable_paths(
        rows_with_duplicate, rows_with_duplicate, {}, {}, "MSW"
    )
    assert selected == [
        rows[0]["source_relative_path"], rows[1]["source_relative_path"]
    ]
    assert not decisions_path.exists()

    # A saved decision, a conflicting hash, and an unreadable record are not
    # returned by the selection helper.
    decisions = {
        rows[1]["source_relative_path"]: {
            "review_status": "include", "reviewed_family": "MSW", "reason": "",
        }
    }
    conflicts = {rows[0]["source_relative_path"]: (rows[0]["source_relative_path"],)}
    rows_with_duplicate[2]["inventory_error"] = "Unreadable in inventory."
    assert review_module.batch_selectable_paths(
        rows_with_duplicate, rows_with_duplicate, decisions, conflicts, "MSW"
    ) == []


def test_bulk_selection_requires_a_matching_family_candidate(review_setup, review_module):
    _, rows, _, _ = review_setup
    assert review_module.batch_selectable_paths(rows, rows, {}, {}, "HOLD") == []
    assert review_module.batch_selectable_paths(rows, rows, {}, {}, "E_WASTE") == []
    assert review_module.batch_selectable_paths(
        rows, rows, {}, {}, "MSW", {rows[0]["source_relative_path"]}
    ) == [rows[0]["source_relative_path"]]


def test_batch_policy_summary_requires_content_based_family_review(review_module):
    summary = review_module.batch_policy_summary("MSW", {"paper", "organic"})
    assert "visible contents of every selected image" in summary
    assert "not ground truth" in summary
    assert "visible food waste is confirmed" in summary


def test_compact_batch_view_offers_all_60_images_in_a_dense_grid(review_module):
    assert review_module.batch_page_sizes(60) == [20, 30, 40, 60]
    assert review_module.batch_grid_columns(60) == 6
    assert review_module.batch_grid_columns(40) == 4


def test_verified_msw_group_count_uses_duplicate_policy_and_current_files(
    tmp_path, review_module
):
    root = tmp_path / "dataset"
    root.mkdir()
    paper = _queue_row(review_module, root, "paper", "one.png", (220, 30, 10))
    duplicate_path = root / "cardboard" / "copy.png"
    duplicate_path.parent.mkdir(parents=True)
    duplicate_path.write_bytes((root / "paper" / "one.png").read_bytes())
    duplicate = {
        **paper,
        "source_relative_path": "cardboard/copy.png",
        "source_folder": "cardboard",
        "proposed_review_route": review_module.family_workflow.REVIEW_ROUTES["cardboard"],
    }
    plastic = _queue_row(review_module, root, "plastic", "two.png", (10, 210, 40))
    inventory = [paper, duplicate, plastic]
    decisions = {
        row["source_relative_path"]: {
            "review_status": "include", "reviewed_family": "MSW", "reason": "",
        }
        for row in inventory
    }

    assert review_module.verified_msw_hash_group_count(inventory, decisions, root) == (2, 0)

    # Differing reviewed labels block the complete shared-hash group.
    decisions[duplicate["source_relative_path"]] = {
        "review_status": "include", "reviewed_family": "E_WASTE", "reason": "",
    }
    assert review_module.verified_msw_hash_group_count(inventory, decisions, root) == (1, 0)

    # A changed source is no longer counted even though its saved label remains.
    (root / plastic["source_relative_path"]).write_bytes(b"changed bytes")
    assert review_module.verified_msw_hash_group_count(inventory, decisions, root) == (0, 1)


def test_projected_batch_count_requires_an_explicit_choice(review_setup, review_module):
    root, rows, _, _ = review_setup
    inventory = [
        {
            **row,
            "reviewed_family": "",
            "review_status": "pending",
        }
        for row in rows
    ]
    existing = {}
    projected = review_module.projected_msw_decisions(
        existing, rows, [rows[0]["source_relative_path"]], "MSW"
    )

    assert existing == {}
    assert projected[rows[0]["source_relative_path"]]["reviewed_family"] == "MSW"
    assert review_module.verified_msw_hash_group_count(inventory, projected, root) == (1, 0)
    assert review_module.verified_msw_hash_group_count(inventory, existing, root) == (0, 0)


def test_review_queue_environment_paths_resolve_from_project_root(review_module, monkeypatch):
    monkeypatch.setenv("VISTA_FAMILY_REVIEW_QUEUE", "data/processed/compact_queue.csv")
    path = review_module._configured_path(
        "VISTA_FAMILY_REVIEW_QUEUE", review_module.family_workflow.QUEUE_PATH
    )
    assert path == review_module.PROJECT_ROOT / "data" / "processed" / "compact_queue.csv"


@pytest.mark.parametrize(
    "folder, expected",
    [
        ("paper", "MSW candidate"),
        ("cardboard", "MSW candidate"),
        ("plastic", "MSW candidate"),
        ("glass", "MSW candidate"),
        ("organic", "MSW candidate — confirm food waste"),
        ("battery", "E_WASTE candidate — verify content"),
        ("metal", "No suggestion — hold for inspection"),
        ("trash", "No suggestion — hold for inspection"),
        ("textile", "Unsupported"),
    ],
)
def test_folder_suggestions_are_display_only(folder, expected, review_module):
    assert review_module.suggested_family_for_row(
        {"source_relative_path": f"{folder}/item.png", "source_folder": folder}
    ) == expected
