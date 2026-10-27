import csv
from pathlib import Path

import pytest
from PIL import Image, ImageFont


@pytest.fixture
def family_workflow(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import family_dataset_workflow

    return family_dataset_workflow


def _write_image(path: Path, color: tuple[int, int, int]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (12, 9), color)
    image.save(path, format="PNG")
    image.close()
    return path.read_bytes()


def _dataset(root: Path, family_workflow, *, duplicates: bool = False):
    for folder in family_workflow.SOURCE_FOLDERS:
        (root / folder).mkdir(parents=True, exist_ok=True)
    for index, color in enumerate(((220, 0, 0), (0, 220, 0), (0, 0, 220))):
        _write_image(root / "paper" / f"paper_{index}.png", color)
        battery_color = tuple(255 - channel for channel in color)
        _write_image(root / "battery" / f"battery_{index}.png", battery_color)
    if duplicates:
        original = root / "paper" / "paper_0.png"
        (root / "paper" / "paper_0_copy.png").write_bytes(original.read_bytes())
    return root


def _workflow_outputs(tmp_path: Path, family_workflow, dataset: Path):
    inventory = tmp_path / "family_inventory.csv"
    family_workflow.write_inventory(dataset, inventory)
    queue = tmp_path / "family_queue.csv"
    decisions = tmp_path / "family_decisions.csv"
    sheets = tmp_path / "sheets"
    family_workflow.create_review_queue(
        inventory,
        dataset,
        per_folder=40,
        queue_path=queue,
        decisions_path=decisions,
        sheets_dir=sheets,
    )
    return inventory, queue, decisions, sheets


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _write_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _approve_queue(decisions_path: Path, family_workflow, queue_rows: list[dict[str, str]]) -> None:
    rows = [
        {
            "source_relative_path": row["source_relative_path"],
            "review_status": "include",
            "reviewed_family": "MSW" if row["source_folder"] == "paper" else "E_WASTE",
            "reason": (
                "Manual inspection confirmed battery/electronic context."
                if row["source_folder"] == "battery" else ""
            ),
        }
        for row in queue_rows
    ]
    _write_rows(decisions_path, family_workflow.DECISION_FIELDS, rows)


def test_inventory_records_routes_hashes_and_pending_only(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    rows = family_workflow.discover_inventory(dataset)
    by_folder = {row["source_folder"]: row for row in rows}

    assert len(rows) == 6
    assert by_folder["paper"]["proposed_review_route"] == "candidate_msw"
    assert family_workflow.REVIEW_ROUTES["organic"] == "candidate_msw_food_waste_only"
    assert family_workflow.REVIEW_ROUTES["battery"] == "candidate_e_waste_verify_content"
    assert family_workflow.REVIEW_ROUTES["metal"] == "hold_manual_review"
    assert family_workflow.REVIEW_ROUTES["textile"] == "unsupported"
    assert all(row["review_status"] == "pending" and not row["reviewed_family"] for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_queue_is_reproducible_deduplicated_and_has_blank_decision_template(
    tmp_path, family_workflow
):
    dataset = _dataset(tmp_path / "source", family_workflow, duplicates=True)
    inventory = tmp_path / "inventory.csv"
    family_workflow.write_inventory(dataset, inventory)
    outputs = []
    for suffix in ("first", "second"):
        result = (
            tmp_path / f"{suffix}.csv",
            tmp_path / f"{suffix}_decisions.csv",
            tmp_path / f"{suffix}_sheets",
        )
        rows = family_workflow.create_review_queue(
            inventory,
            dataset,
            seed=17,
            queue_path=result[0],
            decisions_path=result[1],
            sheets_dir=result[2],
        )
        outputs.append((rows, result))

    assert [row["source_relative_path"] for row in outputs[0][0]] == [
        row["source_relative_path"] for row in outputs[1][0]
    ]
    selected = {row["source_relative_path"] for row in outputs[0][0]}
    assert "paper/paper_0.png" in selected
    assert "paper/paper_0_copy.png" not in selected
    template_rows = _read_rows(outputs[0][1][1])
    assert all(row["review_status"] == "pending" for row in template_rows)
    assert all(row["reviewed_family"] == "" and row["reason"] == "" for row in template_rows)
    assert (outputs[0][1][2] / "paper.pdf").is_file()
    assert (outputs[0][1][2] / "battery.pdf").is_file()


def test_caption_wrapping_preserves_full_source_path(family_workflow):
    path = "paper/a folder/very_long_filename_with_spaces " + "segment " * 15 + ".png"
    font = ImageFont.load_default()
    lines = family_workflow._wrap_pixel_width(path, font, 90)

    assert "".join(lines) == path
    assert all(font.getlength(line) <= 90 for line in lines)


def test_unreviewed_images_never_enter_a_split(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    inventory, queue, decisions, sheets = _workflow_outputs(tmp_path, family_workflow, dataset)
    manifest = tmp_path / "family_manifest.csv"
    rows = family_workflow.prepare_family_manifest(
        inventory, decisions, dataset, manifest, queue_path=queue
    )

    assert manifest.is_file()
    assert all(row["review_status"] == "pending" for row in rows)
    assert all(row["split"] == "" for row in rows)
    assert not any(row["reviewed_family"] for row in rows)
    assert sheets.is_dir()


def test_explicit_approvals_create_deterministic_group_level_splits(
    tmp_path, family_workflow
):
    dataset = _dataset(tmp_path / "source", family_workflow, duplicates=True)
    inventory, queue, decisions, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    _approve_queue(decisions, family_workflow, _read_rows(queue))

    first = tmp_path / "first_manifest.csv"
    second = tmp_path / "second_manifest.csv"
    rows = family_workflow.prepare_family_manifest(
        inventory, decisions, dataset, first, queue_path=queue, seed=81
    )
    family_workflow.prepare_family_manifest(
        inventory, decisions, dataset, second, queue_path=queue, seed=81
    )

    assert first.read_bytes() == second.read_bytes()
    duplicate_hash = family_workflow.sha256_file(dataset / "paper" / "paper_0.png")
    canonical_duplicate_rows = [row for row in rows if row["sha256"] == duplicate_hash]
    assert len(canonical_duplicate_rows) == 2
    assert sum(bool(row["split"]) for row in canonical_duplicate_rows) == 1
    assert {row["split"] for row in canonical_duplicate_rows if row["split"]} <= {
        "train", "validation", "test"
    }
    for family in ("MSW", "E_WASTE"):
        family_rows = [
            row for row in rows
            if row["reviewed_family"] == family and row["group_status"] == "include"
        ]
        assert {row["split"] for row in family_rows} == {"train", "validation", "test"}
    assert all(row["review_status"] == "include" for row in rows if row["split"])

    splits_by_hash: dict[str, set[str]] = {}
    for row in rows:
        if row["split"]:
            splits_by_hash.setdefault(row["sha256"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_hash.values())


def test_readiness_counts_match_canonical_groups_and_require_both_families(
    tmp_path, family_workflow
):
    dataset = _dataset(tmp_path / "source", family_workflow, duplicates=True)
    inventory, queue, decisions_path, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    queue_rows = _read_rows(queue)
    _approve_queue(decisions_path, family_workflow, queue_rows)
    inventory_rows = family_workflow._load_inventory(inventory)
    decisions = family_workflow._load_decisions(
        decisions_path, inventory_rows, queue_rows
    )

    counts = family_workflow.family_readiness_counts(inventory_rows, decisions)
    assert counts == {"E_WASTE": 3, "MSW": 3}

    # Removing one family approval makes the readiness report refuse a split.
    decisions.pop("paper/paper_0.png")
    assert family_workflow.family_readiness_counts(inventory_rows, decisions) == {
        "E_WASTE": 3, "MSW": 2,
    }


def test_readiness_excludes_duplicate_groups_with_conflicting_approved_labels(
    tmp_path, family_workflow
):
    dataset = _dataset(tmp_path / "source", family_workflow)
    shared_bytes = (dataset / "paper" / "paper_0.png").read_bytes()
    (dataset / "battery" / "same.png").write_bytes(shared_bytes)
    inventory, queue, decisions_path, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    inventory_rows = family_workflow._load_inventory(inventory)
    queue_rows = _read_rows(queue)
    decisions = {
        row["source_relative_path"]: {
            "review_status": "include",
            "reviewed_family": "E_WASTE" if row["source_folder"] == "battery" else "MSW",
            "reason": "Human-reviewed visible content.",
        }
        for row in queue_rows
    }

    counts = family_workflow.family_readiness_counts(inventory_rows, decisions)
    # The hash shared between folders is blocked rather than double-counted.
    assert counts["MSW"] == 2
    assert counts["E_WASTE"] == 3


def test_readiness_excludes_held_and_excluded_hash_groups(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    inventory, queue, decisions_path, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    queue_rows = _read_rows(queue)
    _approve_queue(decisions_path, family_workflow, queue_rows)
    decision_rows = _read_rows(decisions_path)
    decision_rows[0].update(review_status="hold", reviewed_family="", reason="Ambiguous view.")
    decision_rows[-1].update(review_status="exclude", reviewed_family="", reason="Not a target item.")
    _write_rows(decisions_path, family_workflow.DECISION_FIELDS, decision_rows)
    inventory_rows = family_workflow._load_inventory(inventory)
    decisions = family_workflow._load_decisions(
        decisions_path, inventory_rows, queue_rows
    )

    counts = family_workflow.family_readiness_counts(inventory_rows, decisions)
    assert counts == {"E_WASTE": 2, "MSW": 2}


def test_readiness_cli_reports_counts_and_split_limitations(
    tmp_path, family_workflow, capsys
):
    dataset = _dataset(tmp_path / "source", family_workflow)
    inventory, queue, decisions, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    _approve_queue(decisions, family_workflow, _read_rows(queue))

    family_workflow.main([
        "readiness", "--inventory", str(inventory), "--queue", str(queue),
        "--review-csv", str(decisions),
    ])
    output = capsys.readouterr().out
    assert "MSW distinct approved hash groups: 3" in output
    assert "E_WASTE distinct approved hash groups: 3" in output
    assert "Three-way split possible (at least 3 groups per family): yes" in output
    assert "does not establish statistically reliable evaluation" in output


def test_preparation_uses_same_sorted_canonical_for_duplicate_group(
    tmp_path, family_workflow
):
    dataset = _dataset(tmp_path / "source", family_workflow, duplicates=True)
    inventory, queue, decisions, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    _approve_queue(decisions, family_workflow, _read_rows(queue))
    inventory_rows = _read_rows(inventory)
    _write_rows(inventory, family_workflow.INVENTORY_FIELDS, list(reversed(inventory_rows)))

    rows = family_workflow.prepare_family_manifest(
        inventory, decisions, dataset, tmp_path / "manifest.csv", queue_path=queue
    )
    duplicate_hash = family_workflow.sha256_file(dataset / "paper" / "paper_0.png")
    group = [row for row in rows if row["sha256"] == duplicate_hash]
    canonical = next(row for row in group if row["group_status"] == "include")
    copy = next(row for row in group if row["group_status"] == "duplicate_copy")
    assert canonical["source_relative_path"] == "paper/paper_0.png"
    assert copy["source_relative_path"] == "paper/paper_0_copy.png"


def test_batch_decision_merge_requires_confirmation_and_preserves_originals(
    tmp_path, family_workflow
):
    dataset = _dataset(tmp_path / "source", family_workflow)
    inventory, queue, decisions_path, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    original_queue_bytes = queue.read_bytes()
    original_decisions = _read_rows(decisions_path)
    battery_path = next(row["source_relative_path"] for row in _read_rows(queue) if row["source_folder"] == "battery")
    battery_decision = next(row for row in original_decisions if row["source_relative_path"] == battery_path)
    battery_decision.update(
        review_status="include",
        reviewed_family="E_WASTE",
        reason="Visible battery product.",
    )
    _write_rows(decisions_path, family_workflow.DECISION_FIELDS, original_decisions)
    original_decision_bytes = decisions_path.read_bytes()

    queue_rows = _read_rows(queue)
    paper_batch = [row for row in queue_rows if row["source_folder"] == "paper"][:1]
    batch_queue = tmp_path / "batch_queue.csv"
    batch_decisions = tmp_path / "batch_decisions.csv"
    _write_rows(batch_queue, family_workflow.QUEUE_FIELDS, paper_batch)
    _write_rows(batch_decisions, family_workflow.DECISION_FIELDS, [{
        "source_relative_path": paper_batch[0]["source_relative_path"],
        "review_status": "include", "reviewed_family": "MSW",
        "reason": "Human visually confirmed municipal paper waste.",
    }])
    output = tmp_path / "merged_decisions.csv"

    with pytest.raises(ValueError, match="Explicitly confirm"):
        family_workflow.merge_review_batch_decisions(
            inventory, queue, decisions_path, batch_queue, batch_decisions,
            dataset, output,
        )
    assert not output.exists()

    merged = family_workflow.merge_review_batch_decisions(
        inventory, queue, decisions_path, batch_queue, batch_decisions,
        dataset, output, confirmed=True,
    )
    assert queue.read_bytes() == original_queue_bytes
    assert decisions_path.read_bytes() == original_decision_bytes
    merged_by_path = {row["source_relative_path"]: row for row in merged}
    assert merged_by_path[battery_path]["reviewed_family"] == "E_WASTE"
    assert merged_by_path[paper_batch[0]["source_relative_path"]]["reviewed_family"] == "MSW"


def test_batch_decision_merge_rejects_conflicting_duplicate_labels(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    duplicate_path = dataset / "battery" / "same_as_paper.png"
    duplicate_path.write_bytes((dataset / "paper" / "paper_0.png").read_bytes())
    inventory, queue, decisions_path, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    queue_rows = _read_rows(queue)
    battery_row = next(row for row in queue_rows if row["source_relative_path"] == "battery/same_as_paper.png")
    paper_row = next(row for row in queue_rows if row["source_relative_path"] == "paper/paper_0.png")
    decisions = _read_rows(decisions_path)
    battery_decision = next(row for row in decisions if row["source_relative_path"] == battery_row["source_relative_path"])
    battery_decision.update(
        review_status="include", reviewed_family="E_WASTE",
        reason="Visible battery product.",
    )
    _write_rows(decisions_path, family_workflow.DECISION_FIELDS, decisions)
    batch_queue = tmp_path / "batch_queue.csv"
    batch_decisions = tmp_path / "batch_decisions.csv"
    _write_rows(batch_queue, family_workflow.QUEUE_FIELDS, [paper_row])
    _write_rows(batch_decisions, family_workflow.DECISION_FIELDS, [{
        "source_relative_path": paper_row["source_relative_path"],
        "review_status": "include", "reviewed_family": "MSW", "reason": "Visible paper waste.",
    }])
    output = tmp_path / "merged_decisions.csv"

    with pytest.raises(ValueError, match="not a canonical, conflict-free hash group"):
        family_workflow.merge_review_batch_decisions(
            inventory, queue, decisions_path, batch_queue, batch_decisions,
            dataset, output, confirmed=True,
        )
    assert not output.exists()


def test_cross_folder_conflicting_family_labels_hold_the_whole_hash_group(
    tmp_path, family_workflow
):
    dataset = _dataset(tmp_path / "source", family_workflow)
    conflict_bytes = (dataset / "paper" / "paper_0.png").read_bytes()
    (dataset / "cardboard" / "same_as_paper.png").write_bytes(conflict_bytes)
    inventory = tmp_path / "inventory.csv"
    family_workflow.write_inventory(dataset, inventory)
    queue = tmp_path / "queue.csv"
    decisions = tmp_path / "decisions.csv"
    family_workflow.create_review_queue(
        inventory, dataset, queue_path=queue, decisions_path=decisions,
        sheets_dir=tmp_path / "sheets",
    )
    queued_paths = {row["source_relative_path"] for row in _read_rows(queue)}
    assert {"paper/paper_0.png", "cardboard/same_as_paper.png"} <= queued_paths
    decision_rows = []
    for row in _read_rows(queue):
        family = "MSW" if row["source_folder"] == "paper" else "E_WASTE"
        decision_rows.append({
            "source_relative_path": row["source_relative_path"],
            "review_status": "include",
            "reviewed_family": family,
            "reason": "Manual inspection recorded a family decision.",
        })
    _write_rows(decisions, family_workflow.DECISION_FIELDS, decision_rows)
    manifest = tmp_path / "manifest.csv"
    rows = family_workflow.prepare_family_manifest(
        inventory, decisions, dataset, manifest, queue_path=queue
    )
    conflict_hash = family_workflow.sha256_file(dataset / "paper" / "paper_0.png")
    group = [row for row in rows if row["sha256"] == conflict_hash]

    assert len(group) == 2
    assert {row["group_status"] for row in group} == {"hold"}
    assert all(row["split"] == "" for row in group)
    assert all("conflicting" in row["review_exclusion_reason"].lower() for row in group)


@pytest.mark.parametrize(
    "decision, family, reason, message",
    [
        ("approved", "MSW", "", "must be pending, include"),
        ("include", "", "", "needs reviewed_family"),
        ("include", "garbage", "", "MSW or E_WASTE"),
        ("hold", "", "  ", "requires a reason"),
        ("exclude", "", "", "requires a reason"),
        ("pending", "MSW", "", "Only included"),
    ],
)
def test_invalid_review_states_are_rejected(
    tmp_path, family_workflow, decision, family, reason, message
):
    dataset = _dataset(tmp_path / "source", family_workflow)
    inventory, queue, decisions, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    rows = _read_rows(decisions)
    rows[0].update(review_status=decision, reviewed_family=family, reason=reason)
    _write_rows(decisions, family_workflow.DECISION_FIELDS, rows)

    with pytest.raises(ValueError, match=message):
        family_workflow.prepare_family_manifest(
            inventory, decisions, dataset, tmp_path / "manifest.csv", queue_path=queue
        )


def test_decision_for_path_outside_the_review_queue_is_rejected(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    inventory, queue, decisions, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    rows = _read_rows(decisions)
    rows.append({
        "source_relative_path": "paper/not_in_queue.png",
        "review_status": "include", "reviewed_family": "MSW", "reason": "",
    })
    _write_rows(decisions, family_workflow.DECISION_FIELDS, rows)

    with pytest.raises(ValueError, match="unknown path"):
        family_workflow.prepare_family_manifest(
            inventory, decisions, dataset, tmp_path / "manifest.csv", queue_path=queue
        )


def test_traversal_and_symlink_paths_are_rejected(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    with pytest.raises(ValueError, match="Unsafe source_relative_path"):
        family_workflow.safe_source_path(dataset, "..\\outside.png")

    outside = tmp_path / "outside.png"
    _write_image(outside, (1, 2, 3))
    link = dataset / "paper" / "outside.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not supported in this test environment")
    with pytest.raises(ValueError, match="symlink"):
        family_workflow.safe_source_path(dataset, "paper/outside.png")


def test_source_hash_change_fails_before_manifest_creation(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    inventory, queue, decisions, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    changed = dataset / "paper" / "paper_0.png"
    _write_image(changed, (99, 98, 97))
    output = tmp_path / "manifest.csv"

    with pytest.raises(ValueError, match="SHA-256 changed"):
        family_workflow.prepare_family_manifest(
            inventory, decisions, dataset, output, queue_path=queue
        )
    assert not output.exists()


def test_unreadable_image_cannot_be_included(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    corrupt_path = dataset / "paper" / "corrupt.png"
    corrupt_path.write_bytes(b"not an image")
    inventory = tmp_path / "inventory.csv"
    family_workflow.write_inventory(dataset, inventory)
    queue = tmp_path / "queue.csv"
    decisions = tmp_path / "decisions.csv"
    family_workflow.create_review_queue(
        inventory, dataset, queue_path=queue, decisions_path=decisions,
        sheets_dir=tmp_path / "sheets",
    )
    queue_rows = _read_rows(queue)
    inventory_row = next(row for row in _read_rows(inventory) if row["source_relative_path"] == "paper/corrupt.png")
    queue_rows.append({field: inventory_row[field] for field in family_workflow.QUEUE_FIELDS})
    _write_rows(queue, family_workflow.QUEUE_FIELDS, queue_rows)
    decision_rows = _read_rows(decisions)
    decision_rows.append({
        "source_relative_path": "paper/corrupt.png",
        "review_status": "include", "reviewed_family": "MSW", "reason": "Reviewed.",
    })
    _write_rows(decisions, family_workflow.DECISION_FIELDS, decision_rows)

    with pytest.raises(ValueError, match="Unreadable or unhashed"):
        family_workflow.prepare_family_manifest(
            inventory, decisions, dataset, tmp_path / "manifest.csv", queue_path=queue
        )


def test_outputs_cannot_enter_existing_pilot_model_directory(tmp_path, family_workflow):
    target = (
        family_workflow.PROJECT_ROOT / "models" / "paper_cardboard_resnet18_pilot"
        / "family_manifest.csv"
    )
    with pytest.raises(ValueError, match="paper/cardboard model directory"):
        family_workflow._validate_new_output(target, tmp_path / "dataset")


def test_textile_cannot_be_included_under_unsupported_route(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    _write_image(dataset / "textile" / "cloth.png", (55, 55, 55))
    inventory = tmp_path / "inventory.csv"
    family_workflow.write_inventory(dataset, inventory)
    queue = tmp_path / "queue.csv"
    decisions = tmp_path / "decisions.csv"
    family_workflow.create_review_queue(
        inventory, dataset, queue_path=queue, decisions_path=decisions,
        sheets_dir=tmp_path / "sheets",
    )
    queue_rows = _read_rows(queue)
    textile = next(row for row in _read_rows(inventory) if row["source_folder"] == "textile")
    queue_rows.append({field: textile[field] for field in family_workflow.QUEUE_FIELDS})
    _write_rows(queue, family_workflow.QUEUE_FIELDS, queue_rows)
    decision_rows = _read_rows(decisions)
    decision_rows.append({
        "source_relative_path": textile["source_relative_path"],
        "review_status": "include", "reviewed_family": "MSW", "reason": "",
    })
    _write_rows(decisions, family_workflow.DECISION_FIELDS, decision_rows)

    with pytest.raises(ValueError, match="unsupported"):
        family_workflow.prepare_family_manifest(
            inventory, decisions, dataset, tmp_path / "manifest.csv", queue_path=queue
        )


def test_valid_hold_and_exclude_decisions_are_preserved(tmp_path, family_workflow):
    dataset = _dataset(tmp_path / "source", family_workflow)
    inventory, queue, decisions, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    rows = _read_rows(decisions)
    rows[0].update(review_status="hold", reason="Image context is ambiguous.")
    rows[1].update(review_status="exclude", reason="Not a waste item.")
    _write_rows(decisions, family_workflow.DECISION_FIELDS, rows)

    manifest_rows = family_workflow.prepare_family_manifest(
        inventory, decisions, dataset, tmp_path / "manifest.csv", queue_path=queue
    )
    statuses = {row["review_status"] for row in manifest_rows}
    assert {"hold", "exclude"} <= statuses
    assert next(row for row in manifest_rows if row["review_status"] == "hold")["review_reason"] == "Image context is ambiguous."
    assert next(row for row in manifest_rows if row["review_status"] == "exclude")["review_reason"] == "Not a waste item."
    assert all(row["split"] == "" for row in manifest_rows)


@pytest.mark.parametrize(
    "folder, family, reason, message",
    [
        ("organic", "MSW", "", "confirming visible food waste"),
        ("organic", "E_WASTE", "food-like", "only be included as MSW"),
        ("battery", "E_WASTE", "", "confirming visible battery/electronic"),
        ("battery", "MSW", "battery visible", "only be included as E_WASTE"),
    ],
)
def test_special_review_routes_require_explicit_content_confirmation(
    tmp_path, family_workflow, folder, family, reason, message
):
    dataset = _dataset(tmp_path / "source", family_workflow)
    _write_image(dataset / folder / "special.png", (101, 44, 22))
    inventory, queue, decisions, _ = _workflow_outputs(tmp_path, family_workflow, dataset)
    decision_rows = _read_rows(decisions)
    decision = next(row for row in decision_rows if row["source_relative_path"] == f"{folder}/special.png")
    decision.update(review_status="include", reviewed_family=family, reason=reason)
    _write_rows(decisions, family_workflow.DECISION_FIELDS, decision_rows)

    with pytest.raises(ValueError, match=message):
        family_workflow.prepare_family_manifest(
            inventory, decisions, dataset, tmp_path / "manifest.csv", queue_path=queue
        )
