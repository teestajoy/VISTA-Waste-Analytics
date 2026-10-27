import csv
import hashlib
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def sampler(monkeypatch):
    scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
    monkeypatch.syspath_prepend(scripts_path)

    import sample_msw_review

    return sample_msw_review


def write_image(path: Path, color: tuple[int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), color).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "source_relative_path",
        "sha256",
        "source_folder",
        "proposed_label",
        "reviewed_label",
        "review_status",
        "duplicate_group_id",
        "split",
        "inclusion_exclusion_reason",
    )
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_manifest_image(rows, dataset_root, category, name, color):
    relative_path = f"{category}/{name}.png"
    image_hash = write_image(dataset_root / relative_path, color)
    rows.append(
        {
            "source_relative_path": relative_path,
            "sha256": image_hash,
            "source_folder": category,
            "proposed_label": category,
            "reviewed_label": "",
            "review_status": "unreviewed",
            "duplicate_group_id": f"sha256:{image_hash}",
            "split": "",
            "inclusion_exclusion_reason": "Reference sample only.",
        }
    )
    return image_hash


def test_sampling_is_reproducible_and_covers_both_classes(tmp_path, sampler):
    dataset_root = tmp_path / "dataset"
    rows = []
    for category, base_color in (
        ("paper", (210, 220, 230)),
        ("cardboard", (180, 150, 120)),
    ):
        for index in range(24):
            color = (base_color[0] - index, base_color[1], base_color[2])
            add_manifest_image(rows, dataset_root, category, f"item_{index:02d}", color)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    first = sampler.select_samples(manifest, dataset_root, seed=17, per_class=20)
    second = sampler.select_samples(manifest, dataset_root, seed=17, per_class=20)

    first_paths = [sample.source_relative_path for sample in first]
    assert first_paths == [sample.source_relative_path for sample in second]
    assert sum(sample.proposed_label == "paper" for sample in first) == 20
    assert sum(sample.proposed_label == "cardboard" for sample in first) == 20


def test_cross_category_hash_conflicts_are_excluded_and_reported(
    tmp_path, sampler, capsys
):
    dataset_root = tmp_path / "dataset"
    rows = []
    paper_image = dataset_root / "paper" / "same.png"
    duplicate_hash = write_image(paper_image, (200, 80, 50))
    cardboard_image = dataset_root / "cardboard" / "same.png"
    cardboard_image.parent.mkdir(parents=True)
    cardboard_image.write_bytes(paper_image.read_bytes())
    rows.extend(
        [
            {
                "source_relative_path": "paper/same.png",
                "sha256": duplicate_hash,
                "source_folder": "paper",
                "proposed_label": "paper",
            },
            {
                "source_relative_path": "cardboard/same.png",
                "sha256": duplicate_hash,
                "source_folder": "cardboard",
                "proposed_label": "cardboard",
            },
        ]
    )
    add_manifest_image(rows, dataset_root, "paper", "eligible", (100, 150, 200))
    add_manifest_image(rows, dataset_root, "cardboard", "eligible", (200, 150, 100))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=4, per_class=20)

    assert {sample.source_relative_path for sample in samples} == {
        "paper/eligible.png",
        "cardboard/eligible.png",
    }
    assert len({sample.sha256 for sample in samples}) == len(samples)
    warning = capsys.readouterr().out
    assert "Warning: excluding cross-category SHA-256" in warning
    assert "paper/same.png" in warning
    assert "cardboard/same.png" in warning


def test_same_category_duplicates_use_lexically_first_canonical_path(tmp_path, sampler):
    dataset_root = tmp_path / "dataset"
    rows = []
    canonical = dataset_root / "paper" / "a_original.png"
    duplicate_hash = write_image(canonical, (200, 80, 50))
    duplicate = dataset_root / "paper" / "z_copy.png"
    duplicate.write_bytes(canonical.read_bytes())
    for relative_path in ("paper/a_original.png", "paper/z_copy.png"):
        rows.append(
            {
                "source_relative_path": relative_path,
                "sha256": duplicate_hash,
                "source_folder": "paper",
                "proposed_label": "paper",
            }
        )
    add_manifest_image(rows, dataset_root, "cardboard", "one_cardboard", (100, 150, 200))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=4, per_class=20)

    paths = {sample.source_relative_path for sample in samples}
    assert "paper/a_original.png" in paths
    assert "paper/z_copy.png" not in paths
    assert len({sample.sha256 for sample in samples}) == len(samples)


@pytest.mark.parametrize("other_category", ["textile", ""])
def test_non_target_category_cannot_suppress_target_canonical_path(
    tmp_path, sampler, other_category
):
    dataset_root = tmp_path / "dataset"
    rows = []
    image_hash = add_manifest_image(
        rows, dataset_root, "paper", "eligible_paper", (210, 220, 230)
    )
    rows.append(
        {
            "source_relative_path": f"{other_category}/earlier_path.png",
            "sha256": image_hash,
            "source_folder": other_category,
            "proposed_label": other_category,
        }
    )
    add_manifest_image(rows, dataset_root, "cardboard", "eligible_cardboard", (180, 150, 120))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=42, per_class=20)

    assert {sample.source_relative_path for sample in samples} == {
        "paper/eligible_paper.png",
        "cardboard/eligible_cardboard.png",
    }


def test_explicitly_excluded_group_is_skipped_but_unreviewed_groups_remain(
    tmp_path, sampler
):
    dataset_root = tmp_path / "dataset"
    rows = []
    excluded_hash = add_manifest_image(
        rows, dataset_root, "paper", "excluded_paper", (210, 220, 230)
    )
    rows[-1]["review_status"] = "excluded"
    rows[-1]["inclusion_exclusion_reason"] = "Excluded: reviewer decision."
    add_manifest_image(rows, dataset_root, "paper", "unreviewed_paper", (210, 210, 230))
    add_manifest_image(rows, dataset_root, "cardboard", "unreviewed_cardboard", (180, 150, 120))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=42, per_class=20)

    sample_paths = {sample.source_relative_path for sample in samples}
    assert "paper/excluded_paper.png" not in sample_paths
    assert "paper/unreviewed_paper.png" in sample_paths
    assert "cardboard/unreviewed_cardboard.png" in sample_paths
    assert all(sample.sha256 != excluded_hash for sample in samples)


def test_excluded_member_skips_entire_duplicate_group_and_keeps_unreviewed_sample(
    tmp_path, sampler
):
    dataset_root = tmp_path / "dataset"
    rows = []
    excluded_hash = add_manifest_image(
        rows, dataset_root, "paper", "a_excluded", (210, 220, 230)
    )
    rows[0]["review_status"] = "excluded"
    rows[0]["inclusion_exclusion_reason"] = "Excluded: reviewer decision."

    approved_copy = dataset_root / "paper" / "b_approved.png"
    approved_copy.write_bytes((dataset_root / "paper" / "a_excluded.png").read_bytes())
    rows.append(
        {
            **rows[0],
            "source_relative_path": "paper/b_approved.png",
            "review_status": "approved",
            "inclusion_exclusion_reason": "Included via duplicate group.",
        }
    )
    add_manifest_image(
        rows, dataset_root, "cardboard", "unreviewed_cardboard", (180, 150, 120)
    )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=42, per_class=20)

    sample_paths = {sample.source_relative_path for sample in samples}
    assert "paper/a_excluded.png" not in sample_paths
    assert "paper/b_approved.png" not in sample_paths
    assert sample_paths == {"cardboard/unreviewed_cardboard.png"}
    assert all(sample.sha256 != excluded_hash for sample in samples)


def test_unreviewed_images_are_eligible_for_manual_review(tmp_path, sampler):
    dataset_root = tmp_path / "dataset"
    rows = []
    add_manifest_image(rows, dataset_root, "paper", "unreviewed_paper", (210, 220, 230))
    add_manifest_image(rows, dataset_root, "cardboard", "unreviewed_cardboard", (180, 150, 120))
    assert all(row["review_status"] == "unreviewed" for row in rows)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=42, per_class=20)

    assert {sample.source_relative_path for sample in samples} == {
        "paper/unreviewed_paper.png",
        "cardboard/unreviewed_cardboard.png",
    }


def test_unsafe_manifest_paths_are_skipped(tmp_path, sampler, capsys):
    dataset_root = tmp_path / "dataset"
    outside_image = tmp_path / "outside.png"
    image_hash = write_image(outside_image, (10, 20, 30))
    rows = [
        {
            "source_relative_path": "paper/../../outside.png",
            "sha256": image_hash,
            "source_folder": "paper",
            "proposed_label": "paper",
        }
    ]
    add_manifest_image(rows, dataset_root, "cardboard", "safe", (30, 20, 10))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=42, per_class=20)

    assert [sample.source_relative_path for sample in samples] == ["cardboard/safe.png"]
    assert "Invalid source-relative image path" in capsys.readouterr().out


def test_symlink_outside_dataset_root_is_skipped_when_supported(
    tmp_path, sampler, capsys
):
    dataset_root = tmp_path / "dataset"
    outside_image = tmp_path / "outside.png"
    image_hash = write_image(outside_image, (10, 20, 30))
    paper_folder = dataset_root / "paper"
    paper_folder.mkdir(parents=True)
    link = paper_folder / "outside_link.png"
    try:
        link.symlink_to(outside_image)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")

    rows = [
        {
            "source_relative_path": "paper/outside_link.png",
            "sha256": image_hash,
            "source_folder": "paper",
            "proposed_label": "paper",
        }
    ]
    add_manifest_image(rows, dataset_root, "cardboard", "safe", (30, 20, 10))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=42, per_class=20)

    assert [sample.source_relative_path for sample in samples] == ["cardboard/safe.png"]
    assert "resolves outside the dataset root" in capsys.readouterr().out


def test_paper_and_cardboard_contact_sheets_are_separate(tmp_path, sampler):
    dataset_root = tmp_path / "dataset"
    rows = []
    add_manifest_image(rows, dataset_root, "paper", "paper_sample", (210, 220, 230))
    add_manifest_image(rows, dataset_root, "cardboard", "cardboard_sample", (180, 150, 120))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    samples = sampler.select_samples(manifest, dataset_root, per_class=1)

    outputs = {
        "paper": tmp_path / "msw_review_sample_paper.jpg",
        "cardboard": tmp_path / "msw_review_sample_cardboard.jpg",
    }
    for category, output in outputs.items():
        sampler.save_contact_sheet(
            output,
            [sample for sample in samples if sample.proposed_label == category],
        )

    assert outputs["paper"].is_file()
    assert outputs["cardboard"].is_file()
    assert outputs["paper"] != outputs["cardboard"]
    with Image.open(outputs["paper"]) as paper_sheet:
        assert paper_sheet.width > 0 and paper_sheet.height > 0
    with Image.open(outputs["cardboard"]) as cardboard_sheet:
        assert cardboard_sheet.width > 0 and cardboard_sheet.height > 0


def test_main_uses_configured_output_paths_without_writing_artifacts(
    tmp_path, sampler, monkeypatch
):
    samples = [
        sampler.ReviewSample(
            source_relative_path="paper/example.png",
            sha256="paper-hash",
            source_folder="paper",
            proposed_label="paper",
            thumbnail=Image.new("RGB", (8, 8), "white"),
        ),
        sampler.ReviewSample(
            source_relative_path="cardboard/example.png",
            sha256="cardboard-hash",
            source_folder="cardboard",
            proposed_label="cardboard",
            thumbnail=Image.new("RGB", (8, 8), "white"),
        ),
    ]
    manifest_path = tmp_path / "input_manifest.csv"
    dataset_root = tmp_path / "dataset"
    sample_csv_path = tmp_path / "reference.csv"
    contact_sheet_paths = {
        "paper": tmp_path / "paper.jpg",
        "cardboard": tmp_path / "cardboard.jpg",
    }
    calls = {}
    assert sampler.SAMPLE_CSV_PATH == sampler.PROJECT_ROOT / "data" / "processed" / "msw_review_sample.csv"
    assert sampler.CONTACT_SHEET_PATHS == {
        "paper": sampler.PROJECT_ROOT / "test-data" / "msw_review_sample_paper.jpg",
        "cardboard": sampler.PROJECT_ROOT / "test-data" / "msw_review_sample_cardboard.jpg",
    }
    monkeypatch.setattr(sampler, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(sampler, "DATASET_DIR", dataset_root)
    def fake_select_samples(manifest, root, **kwargs):
        calls["selection"] = (manifest, root, kwargs)
        return samples

    monkeypatch.setattr(sampler, "select_samples", fake_select_samples)
    monkeypatch.setattr(
        sampler,
        "save_sample_csv",
        lambda path, selected: calls.setdefault("csv", (path, selected)),
    )
    monkeypatch.setattr(
        sampler,
        "save_contact_sheet",
        lambda path, selected: calls.setdefault(
            f"sheet_{selected[0].proposed_label}", (path, selected)
        ),
    )

    preserved_sample_path = tmp_path / "preserved_sample.csv"
    sampler.main(
        [
            "--manifest",
            str(manifest_path),
            "--dataset-root",
            str(dataset_root),
            "--preserve-sample",
            str(preserved_sample_path),
            "--sample-csv",
            str(sample_csv_path),
            "--paper-sheet",
            str(contact_sheet_paths["paper"]),
            "--cardboard-sheet",
            str(contact_sheet_paths["cardboard"]),
        ]
    )

    assert calls["selection"] == (
        manifest_path,
        dataset_root,
        {
            "seed": sampler.RANDOM_SEED,
            "per_class": sampler.SAMPLE_COUNT_PER_CLASS,
            "preserve_sample_path": preserved_sample_path,
        },
    )
    assert calls["csv"] == (sample_csv_path, samples)
    assert calls["sheet_paper"][0] == contact_sheet_paths["paper"]
    assert calls["sheet_cardboard"][0] == contact_sheet_paths["cardboard"]
    assert not sample_csv_path.exists()
    assert all(not path.exists() for path in contact_sheet_paths.values())


def test_insufficient_unique_images_are_reported_without_inventing_rows(
    tmp_path, sampler, capsys
):
    dataset_root = tmp_path / "dataset"
    rows = []
    add_manifest_image(rows, dataset_root, "paper", "only_paper", (210, 220, 230))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)

    samples = sampler.select_samples(manifest, dataset_root, seed=42, per_class=20)

    assert [sample.source_relative_path for sample in samples] == ["paper/only_paper.png"]
    output = capsys.readouterr().out
    assert "Only 1 readable canonical paper images were available" in output
    assert "Only 0 readable canonical cardboard images were available" in output
