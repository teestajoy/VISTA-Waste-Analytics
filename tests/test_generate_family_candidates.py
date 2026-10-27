import csv
import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def candidate_module(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "scripts"))
    monkeypatch.syspath_prepend(str(project_root / "src"))
    from scripts import generate_family_candidates

    return generate_family_candidates


def _row(path, folder, digest, route, error=""):
    return {
        "source_relative_path": path,
        "sha256": digest,
        "source_folder": folder,
        "proposed_review_route": route,
        "reviewed_family": "",
        "review_status": "pending",
        "duplicate_group_id": f"sha256:{digest}" if digest else "",
        "inventory_error": error,
    }


def test_candidates_are_suggestions_not_approvals(candidate_module):
    policy = candidate_module.MWCD_FAMILY_SOURCE_POLICY
    rows = [
        _row(f"{folder}/x.png", folder, hashlib.sha256(folder.encode()).hexdigest(), candidate_module.workflow.REVIEW_ROUTES[folder])
        for folder in ("paper", "organic", "battery", "metal", "trash", "textile")
    ]
    output = candidate_module.build_candidate_rows(rows, [], {})
    by_folder = {row["source_folder"]: row for row in output}

    assert by_folder["paper"]["suggested_family"] == "MSW"
    assert by_folder["paper"]["candidate_status"] == "pending_human_review"
    assert "food waste" in by_folder["organic"]["review_requirement"]
    assert by_folder["battery"]["suggested_family"] == "E_WASTE"
    assert "confirms" in by_folder["battery"]["review_requirement"]
    assert by_folder["metal"]["suggested_family"] == ""
    assert by_folder["trash"]["suggested_family"] == ""
    assert by_folder["textile"]["candidate_status"] == "unsupported"
    assert all(row["human_review_status"] == "pending" for row in output)


def test_cross_folder_exact_duplicate_is_held_without_candidate(candidate_module):
    digest = hashlib.sha256(b"same image bytes").hexdigest()
    policy = candidate_module.MWCD_FAMILY_SOURCE_POLICY
    rows = [
        _row("paper/a.png", "paper", digest, policy["paper"].disposition.value),
        _row("battery/b.png", "battery", digest, policy["battery"].disposition.value),
    ]

    output = candidate_module.build_candidate_rows(rows, [], {})

    assert all(row["candidate_status"] == "cross_folder_duplicate_hold" for row in output)
    assert all(row["suggested_family"] == "" for row in output)
    assert all(row["duplicate_group_paths"] == "battery/b.png | paper/a.png" for row in output)


def test_same_folder_duplicate_uses_deterministic_canonical_path(candidate_module):
    digest = hashlib.sha256(b"same bytes").hexdigest()
    route = candidate_module.workflow.REVIEW_ROUTES["paper"]
    rows = [
        _row("paper/z.png", "paper", digest, route),
        _row("paper/a.png", "paper", digest, route),
    ]

    output = candidate_module.build_candidate_rows(rows, [], {})

    assert output[0]["source_relative_path"] == "paper/a.png"
    assert output[0]["candidate_status"] == "pending_human_review"
    assert output[1]["candidate_status"] == "same_folder_duplicate_copy"
    assert "paper/a.png" in output[1]["review_requirement"]


def test_unreadable_and_unhashed_rows_never_get_suggestions(candidate_module):
    policy = candidate_module.MWCD_FAMILY_SOURCE_POLICY["paper"]
    rows = [
        _row("paper/bad.png", "paper", "", policy.disposition.value, "decode error"),
    ]

    result = candidate_module.build_candidate_rows(rows, [], {})[0]

    assert result["candidate_status"] == "unreadable_or_unhashed"
    assert result["suggested_family"] == ""


def test_summary_groups_reasons_without_per_path_output(candidate_module):
    rows = [
        {
            "source_folder": "paper",
            "suggested_family": "MSW",
            "candidate_status": "same_folder_duplicate_copy",
            "review_requirement": f"canonical path {index}",
            "human_review_status": "pending",
        }
        for index in range(100)
    ]

    summary = candidate_module.summarize(rows)

    assert summary["review_requirement"] == {
        "Same-folder duplicate copy; use the canonical path": 100
    }


def test_generate_report_refuses_existing_output_and_preserves_decisions(
    tmp_path, candidate_module
):
    inventory_path = tmp_path / "inventory.csv"
    queue_path = tmp_path / "queue.csv"
    decisions_path = tmp_path / "decisions.csv"
    output_path = tmp_path / "candidate.csv"
    digest = hashlib.sha256(b"image").hexdigest()
    inventory_row = _row("paper/a.png", "paper", digest, candidate_module.workflow.REVIEW_ROUTES["paper"])
    queue_row = {
        field: inventory_row[field]
        for field in candidate_module.workflow.QUEUE_FIELDS
    }
    for path, fields, rows in (
        (inventory_path, candidate_module.workflow.INVENTORY_FIELDS, [inventory_row]),
        (queue_path, candidate_module.workflow.QUEUE_FIELDS, [queue_row]),
        (
            decisions_path,
            candidate_module.workflow.DECISION_FIELDS,
            [{"source_relative_path": "paper/a.png", "review_status": "include", "reviewed_family": "MSW", "reason": "Human inspected."}],
        ),
    ):
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    report, _ = candidate_module.generate_report(
        inventory_path, queue_path, decisions_path, output_path, tmp_path / "dataset"
    )

    assert report[0]["suggested_family"] == "MSW"
    assert report[0]["human_review_status"] == "include"
    assert report[0]["human_reviewed_family"] == "MSW"
    assert "human_reviewed_family" not in candidate_module.FIELDS[:5]
    original = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        candidate_module.generate_report(
            inventory_path, queue_path, decisions_path, output_path, tmp_path / "dataset"
        )
    assert output_path.read_bytes() == original
