from pathlib import Path

import pytest


@pytest.fixture
def duplicate_checker(monkeypatch):
    scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
    monkeypatch.syspath_prepend(scripts_path)

    import check_dataset_duplicates

    return check_dataset_duplicates


def make_dataset(root):
    plastic_dir = root / "plastic"
    paper_dir = root / "paper"
    glass_dir = root / "glass"
    plastic_dir.mkdir(parents=True)
    paper_dir.mkdir()
    glass_dir.mkdir()

    (plastic_dir / "plastic_copy_a.jpg").write_bytes(b"shared image bytes")
    (plastic_dir / "plastic_copy_b.jpg").write_bytes(b"shared image bytes")
    (plastic_dir / "plastic_unique.jpg").write_bytes(b"plastic unique bytes")
    (paper_dir / "paper_copy.jpg").write_bytes(b"shared image bytes")
    (paper_dir / "paper_unique.jpg").write_bytes(b"paper unique bytes")
    (paper_dir / "paper_duplicate_a.jpg").write_bytes(b"second duplicate group")
    (paper_dir / "paper_duplicate_b.jpg").write_bytes(b"second duplicate group")
    (glass_dir / "unreadable.png").write_bytes(b"unreadable test bytes")


def test_summary_counts_duplicate_groups_files_and_unique_hashes(
    tmp_path, monkeypatch, capsys, duplicate_checker
):
    make_dataset(tmp_path)
    monkeypatch.setattr(duplicate_checker, "DATASET_DIR", tmp_path)
    original_hash = duplicate_checker.sha256_file

    def hash_with_one_unreadable_file(path):
        if path.name == "unreadable.png":
            raise PermissionError("simulated read failure")
        return original_hash(path)

    monkeypatch.setattr(duplicate_checker, "sha256_file", hash_with_one_unreadable_file)

    duplicate_checker.main([])

    output = capsys.readouterr().out
    assert "plastic: 3" in output
    assert "paper: 4" in output
    assert "glass: 0" in output
    assert "Unreadable files: 1" in output
    assert "plastic: 1" in output
    assert "paper: 1" in output
    assert "Cross-category duplicate groups: 1" in output
    assert "Total files involved in duplicate groups: 5" in output
    assert "Unique images after exact duplicates: 4" in output
    assert "Total duplicate groups: 2" in output
    assert "Missing category folder; skipping:" in output
    assert "Exact duplicate image groups:" not in output


def test_details_option_prints_duplicate_paths_and_relationships(
    tmp_path, monkeypatch, capsys, duplicate_checker
):
    make_dataset(tmp_path)
    monkeypatch.setattr(duplicate_checker, "DATASET_DIR", tmp_path)

    duplicate_checker.main(["--details"])

    output = capsys.readouterr().out
    assert "Exact duplicate image groups:" in output
    assert "[plastic] plastic\\plastic_copy_a.jpg" in output
    assert "[plastic] plastic\\plastic_copy_b.jpg" in output
    assert "[paper] paper\\paper_copy.jpg" in output
    assert "[paper] paper\\paper_duplicate_a.jpg" in output
    assert "[paper] paper\\paper_duplicate_b.jpg" in output
    assert "Duplicate within category: plastic" in output
    assert "Duplicate within category: paper" in output
    assert "Duplicate across categories: paper and plastic" in output


def test_missing_dataset_prints_zero_summary(
    tmp_path, monkeypatch, capsys, duplicate_checker
):
    missing_dataset = tmp_path / "missing"
    monkeypatch.setattr(duplicate_checker, "DATASET_DIR", missing_dataset)

    duplicate_checker.main([])

    output = capsys.readouterr().out
    assert f"Dataset folder not found: {missing_dataset}" in output
    assert "plastic: 0" in output
    assert "Total files involved in duplicate groups: 0" in output
    assert "Unique images after exact duplicates: 0" in output
