import csv
from pathlib import Path

import pytest


@pytest.fixture
def preparer(monkeypatch):
    scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
    monkeypatch.syspath_prepend(scripts_path)

    import prepare_msw_dataset

    return prepare_msw_dataset


def make_dataset(root):
    paper = root / "paper"
    cardboard = root / "cardboard"
    paper.mkdir(parents=True)
    cardboard.mkdir()
    return paper, cardboard


def write_reviews(path, rows):
    with path.open("w", encoding="utf-8", newline="") as review_file:
        writer = csv.DictWriter(
            review_file,
            fieldnames=("source_relative_path", "decision", "reviewed_label", "reason"),
        )
        writer.writeheader()
        writer.writerows(rows)


def approved_rows(paths, label_by_folder=None):
    label_by_folder = label_by_folder or {"paper": "paper", "cardboard": "cardboard"}
    return [
        {
            "source_relative_path": path.relative_to(path.parents[1]).as_posix(),
            "decision": "include",
            "reviewed_label": label_by_folder[path.parts[-2]],
            "reason": "Reviewed sample.",
        }
        for path in paths
    ]


def read_manifest(path):
    with path.open("r", encoding="utf-8", newline="") as manifest_file:
        return list(csv.DictReader(manifest_file))


def test_without_review_csv_writes_conservative_audit_and_preserves_sources(
    tmp_path, preparer
):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    paper_image = paper / "paper_a.jpg"
    cardboard_image = cardboard / "cardboard_a.png"
    paper_image.write_bytes(b"paper image bytes")
    cardboard_image.write_bytes(b"cardboard image bytes")
    ignored = dataset / "plastic"
    ignored.mkdir()
    (ignored / "not_in_scope.jpg").write_bytes(b"not in scope")
    original_bytes = {path: path.read_bytes() for path in (paper_image, cardboard_image)}

    output = tmp_path / "manifest.csv"
    records = preparer.prepare_manifest(dataset, output)

    rows = read_manifest(output)
    assert len(records) == len(rows) == 2
    assert {row["source_relative_path"] for row in rows} == {
        "paper/paper_a.jpg",
        "cardboard/cardboard_a.png",
    }
    assert all(row["review_status"] == "unreviewed" for row in rows)
    assert all(row["split"] == "" for row in rows)
    assert all("excluded conservatively" in row["inclusion_exclusion_reason"].lower() for row in rows)
    assert all(row["sha256"] and row["duplicate_group_id"].startswith("sha256:") for row in rows)
    assert {path: path.read_bytes() for path in original_bytes} == original_bytes


def test_reviewed_split_is_reproducible_and_duplicate_files_stay_together(
    tmp_path, preparer
):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    paths = [
        paper / "a_canonical.jpg",
        paper / "z_duplicate.jpg",
        paper / "paper_b.jpg",
        paper / "paper_c.jpg",
        cardboard / "cardboard_a.jpg",
        cardboard / "cardboard_b.jpg",
        cardboard / "cardboard_c.jpg",
    ]
    shared_bytes = b"identical reviewed image bytes"
    for path in paths:
        path.write_bytes(shared_bytes if path.name in {"a_canonical.jpg", "z_duplicate.jpg"} else path.name.encode())

    review_csv = tmp_path / "review.csv"
    write_reviews(review_csv, approved_rows(paths))
    first_output = tmp_path / "first.csv"
    second_output = tmp_path / "second.csv"

    preparer.prepare_manifest(dataset, first_output, review_csv=review_csv, seed=73)
    preparer.prepare_manifest(dataset, second_output, review_csv=review_csv, seed=73)

    first_rows = read_manifest(first_output)
    second_rows = read_manifest(second_output)
    assert first_output.read_bytes() == second_output.read_bytes()

    duplicate_rows = [
        row
        for row in first_rows
        if row["sha256"] == preparer.sha256_file(paths[0])
    ]
    assert {row["source_relative_path"] for row in duplicate_rows} == {
        "paper/a_canonical.jpg",
        "paper/z_duplicate.jpg",
    }
    assert len({row["split"] for row in duplicate_rows}) == 1
    included = [row for row in duplicate_rows if row["inclusion_exclusion_reason"].startswith("Included")]
    copies = [row for row in duplicate_rows if "Excluded duplicate copy" in row["inclusion_exclusion_reason"]]
    assert [row["source_relative_path"] for row in included] == ["paper/a_canonical.jpg"]
    assert [row["source_relative_path"] for row in copies] == ["paper/z_duplicate.jpg"]

    for label in ("paper", "cardboard"):
        label_rows = [
            row for row in first_rows
            if row["reviewed_label"] == label
            and row["inclusion_exclusion_reason"].startswith("Included")
        ]
        assert {row["split"] for row in label_rows} == {"train", "validation", "test"}


def test_cross_folder_duplicate_is_held_without_consistent_review(tmp_path, preparer):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    conflict_paper = paper / "same.jpg"
    conflict_cardboard = cardboard / "same.jpg"
    conflict_paper.write_bytes(b"same bytes with conflicting folder labels")
    conflict_cardboard.write_bytes(b"same bytes with conflicting folder labels")
    paper_paths = [paper / f"paper_{index}.jpg" for index in range(3)]
    cardboard_paths = [cardboard / f"cardboard_{index}.jpg" for index in range(3)]
    for index, path in enumerate(paper_paths + cardboard_paths):
        path.write_bytes(f"unique-{index}".encode())

    review_csv = tmp_path / "review.csv"
    reviewed_paths = paper_paths + cardboard_paths
    write_reviews(review_csv, approved_rows(reviewed_paths))
    output = tmp_path / "manifest.csv"

    preparer.prepare_manifest(dataset, output, review_csv=review_csv)

    rows = read_manifest(output)
    conflict_rows = [row for row in rows if row["sha256"] == preparer.sha256_file(conflict_paper)]
    assert len(conflict_rows) == 2
    assert {row["review_status"] for row in conflict_rows} == {"held_by_duplicate_group"}
    assert all(row["split"] == "" for row in conflict_rows)
    assert all("both source folders" in row["inclusion_exclusion_reason"] for row in conflict_rows)


def test_review_can_resolve_cross_folder_hash_to_one_label(tmp_path, preparer):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    conflict_paper = paper / "same.jpg"
    conflict_cardboard = cardboard / "same.jpg"
    conflict_paper.write_bytes(b"same bytes reviewed as paper")
    conflict_cardboard.write_bytes(b"same bytes reviewed as paper")
    paper_paths = [paper / f"paper_{index}.jpg" for index in range(3)]
    cardboard_paths = [cardboard / f"cardboard_{index}.jpg" for index in range(3)]
    for index, path in enumerate(paper_paths + cardboard_paths):
        path.write_bytes(f"unique-{index}".encode())

    review_csv = tmp_path / "review.csv"
    rows = approved_rows(paper_paths + cardboard_paths)
    rows.extend(approved_rows([conflict_paper, conflict_cardboard], {"paper": "paper", "cardboard": "paper"}))
    write_reviews(review_csv, rows)
    output = tmp_path / "manifest.csv"

    preparer.prepare_manifest(dataset, output, review_csv=review_csv)

    manifest_rows = read_manifest(output)
    conflict_rows = [row for row in manifest_rows if row["sha256"] == preparer.sha256_file(conflict_paper)]
    assert {row["reviewed_label"] for row in conflict_rows} == {"paper"}
    assert len({row["split"] for row in conflict_rows}) == 1
    assert conflict_rows[0]["split"] in {"train", "validation", "test"}
    assert sum(row["inclusion_exclusion_reason"].startswith("Included") for row in conflict_rows) == 1
    assert sum("Excluded duplicate copy" in row["inclusion_exclusion_reason"] for row in conflict_rows) == 1


def test_split_fails_when_a_class_has_too_few_unique_reviewed_groups(tmp_path, preparer):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    paper_paths = [paper / f"paper_{index}.jpg" for index in range(2)]
    cardboard_paths = [cardboard / f"cardboard_{index}.jpg" for index in range(3)]
    for index, path in enumerate(paper_paths + cardboard_paths):
        path.write_bytes(f"unique-{index}".encode())
    review_csv = tmp_path / "review.csv"
    write_reviews(review_csv, approved_rows(paper_paths + cardboard_paths))
    output = tmp_path / "manifest.csv"

    with pytest.raises(ValueError, match="at least three unique approved hash groups per class"):
        preparer.prepare_manifest(dataset, output, review_csv=review_csv)

    assert not output.exists()


def test_missing_category_and_output_inside_dataset_fail_clearly(tmp_path, preparer):
    dataset = tmp_path / "dataset"
    (dataset / "paper").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Required category folder not found"):
        preparer.discover_images(dataset)

    with pytest.raises(ValueError, match="outside the original dataset tree"):
        preparer.validate_output_path(dataset, dataset / "manifest.csv")


def test_duplicate_group_exclusion_uses_one_group_level_reason(tmp_path, preparer):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    duplicate_paths = [paper / "a.jpg", paper / "z.jpg"]
    for path in duplicate_paths:
        path.write_bytes(b"same excluded group")
    other_paths = [paper / f"paper_{i}.jpg" for i in range(3)]
    other_paths += [cardboard / f"cardboard_{i}.jpg" for i in range(3)]
    for index, path in enumerate(other_paths):
        path.write_bytes(f"unique-{index}".encode())

    rows = approved_rows(duplicate_paths + other_paths)
    excluded = next(row for row in rows if row["source_relative_path"] == "paper/z.jpg")
    excluded.update(decision="exclude", reviewed_label="", reason="Reviewer found a bad label.")
    review_csv = tmp_path / "review.csv"
    write_reviews(review_csv, rows)
    output = tmp_path / "manifest.csv"

    preparer.prepare_manifest(dataset, output, review_csv=review_csv)

    duplicate_rows = [
        row for row in read_manifest(output) if row["sha256"] == preparer.sha256_file(duplicate_paths[0])
    ]
    reasons = {row["inclusion_exclusion_reason"] for row in duplicate_rows}
    assert len(reasons) == 1
    reason = reasons.pop()
    assert "At least one exact-duplicate path was explicitly excluded" in reason
    assert "paper/z.jpg: Reviewer found a bad label" in reason
    assert "paper/a.jpg: Reviewed sample." not in reason
    assert all(not row["split"] for row in duplicate_rows)


def test_review_and_manifest_cannot_be_the_same_file(tmp_path, preparer):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    (paper / "paper.jpg").write_bytes(b"paper")
    (cardboard / "cardboard.jpg").write_bytes(b"cardboard")
    review_csv = tmp_path / "review.csv"
    write_reviews(review_csv, [])
    original_review = review_csv.read_bytes()

    with pytest.raises(ValueError, match="Manifest output and review CSV refer to the same file"):
        preparer.prepare_manifest(dataset, review_csv, review_csv=review_csv)

    assert review_csv.read_bytes() == original_review


def test_nonempty_review_row_with_blank_path_is_rejected(tmp_path, preparer):
    dataset = tmp_path / "dataset"
    paper, _ = make_dataset(dataset)
    (paper / "paper.jpg").write_bytes(b"paper")
    review_csv = tmp_path / "review.csv"
    write_reviews(
        review_csv,
        [{"source_relative_path": "", "decision": "include", "reviewed_label": "paper", "reason": "Reviewed."}],
    )

    records = preparer.discover_images(dataset)
    with pytest.raises(ValueError, match="line 2 has a blank source_relative_path"):
        preparer.load_review_csv(review_csv, records)


def test_fully_empty_delimited_review_row_is_ignored(tmp_path, preparer):
    review_csv = tmp_path / "review.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as review_file:
        writer = csv.DictWriter(
            review_file,
            fieldnames=("source_relative_path", "decision", "reviewed_label", "reason"),
        )
        writer.writeheader()
        writer.writerow({})

    assert preparer.load_review_csv(review_csv, []) == {}


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        ("hold", ""),
        ("hold", "   \t  "),
        ("exclude", ""),
        ("exclude", "  \t "),
    ],
)
def test_hold_and_exclude_require_nonblank_reason_for_source_path(
    tmp_path, preparer, decision, reason
):
    dataset = tmp_path / "dataset"
    paper, _ = make_dataset(dataset)
    (paper / "sample.jpg").write_bytes(b"synthetic image bytes")
    review_csv = tmp_path / "review.csv"
    write_reviews(
        review_csv,
        [
            {
                "source_relative_path": "paper/sample.jpg",
                "decision": decision,
                "reviewed_label": "",
                "reason": reason,
            }
        ],
    )
    records = preparer.discover_images(dataset)

    with pytest.raises(ValueError, match="requires a non-empty reason") as error:
        preparer.load_review_csv(review_csv, records)

    assert "paper/sample.jpg" in str(error.value)


@pytest.mark.parametrize("reason", ["", "   "])
def test_include_does_not_require_a_reason(tmp_path, preparer, reason):
    dataset = tmp_path / "dataset"
    paper, _ = make_dataset(dataset)
    (paper / "sample.jpg").write_bytes(b"synthetic image bytes")
    review_csv = tmp_path / "review.csv"
    write_reviews(
        review_csv,
        [
            {
                "source_relative_path": "paper/sample.jpg",
                "decision": "include",
                "reviewed_label": "paper",
                "reason": reason,
            }
        ],
    )
    records = preparer.discover_images(dataset)

    reviews = preparer.load_review_csv(review_csv, records)

    assert reviews["paper/sample.jpg"] == {
        "decision": "include",
        "reviewed_label": "paper",
        "reason": "",
    }


def test_explicit_hold_and_exclude_decisions_are_recorded(tmp_path, preparer):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    approved = [paper / f"paper_{index}.jpg" for index in range(3)]
    approved += [cardboard / f"cardboard_{index}.jpg" for index in range(3)]
    held = paper / "held.jpg"
    excluded = cardboard / "excluded.jpg"
    for index, path in enumerate(approved + [held, excluded]):
        path.write_bytes(f"image-{index}".encode())
    rows = approved_rows(approved)
    rows.extend(
        [
            {"source_relative_path": "paper/held.jpg", "decision": "hold", "reviewed_label": "", "reason": "Mixed material."},
            {"source_relative_path": "cardboard/excluded.jpg", "decision": "exclude", "reviewed_label": "", "reason": "Out of scope."},
        ]
    )
    review_csv = tmp_path / "review.csv"
    write_reviews(review_csv, rows)
    output = tmp_path / "manifest.csv"

    preparer.prepare_manifest(dataset, output, review_csv=review_csv)

    rows_by_path = {row["source_relative_path"]: row for row in read_manifest(output)}
    assert rows_by_path["paper/held.jpg"]["review_status"] == "hold"
    assert rows_by_path["paper/held.jpg"]["split"] == ""
    assert "Mixed material" in rows_by_path["paper/held.jpg"]["inclusion_exclusion_reason"]
    assert rows_by_path["cardboard/excluded.jpg"]["review_status"] == "excluded"
    assert rows_by_path["cardboard/excluded.jpg"]["split"] == ""
    assert "Out of scope" in rows_by_path["cardboard/excluded.jpg"]["inclusion_exclusion_reason"]


def test_hash_read_error_is_reported_and_other_files_continue(tmp_path, preparer, monkeypatch):
    dataset = tmp_path / "dataset"
    paper, cardboard = make_dataset(dataset)
    bad = paper / "unreadable.jpg"
    good = cardboard / "readable.jpg"
    bad.write_bytes(b"cannot read")
    good.write_bytes(b"readable")
    original_hash = preparer.sha256_file

    def simulated_hash(path):
        if path == bad:
            raise PermissionError("simulated access denied")
        return original_hash(path)

    monkeypatch.setattr(preparer, "sha256_file", simulated_hash)
    output = tmp_path / "manifest.csv"

    preparer.prepare_manifest(dataset, output)

    rows = {row["source_relative_path"]: row for row in read_manifest(output)}
    assert rows["paper/unreadable.jpg"]["review_status"] == "unreadable"
    assert "simulated access denied" in rows["paper/unreadable.jpg"]["inclusion_exclusion_reason"]
    assert rows["paper/unreadable.jpg"]["sha256"] == ""
    assert rows["cardboard/readable.jpg"]["sha256"] == original_hash(good)
