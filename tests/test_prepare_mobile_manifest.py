from __future__ import annotations

import copy
import csv
import hashlib
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import prepare_mobile_manifest as mobile  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_repository_protected_manifests(monkeypatch):
    """Keep unit fixtures isolated; dedicated tests add protected defaults explicitly."""
    monkeypatch.setattr(mobile, "DEFAULT_PROTECTED_MANIFESTS", ())


def make_review_row(relative: str, digest: str, device: str, **overrides) -> dict[str, str]:
    row = {field: "" for field in mobile.REQUIRED_FIELDS}
    row.update({
        "source_relative_path": relative,
        "sha256": digest,
        "proposed_category": "mobile",
        "proposed_family": "E_WASTE",
        "reviewed_category": "mobile",
        "reviewed_family": "E_WASTE",
        "category_decision": "include",
        "family_decision": "include",
        "review_status": "complete",
        "reviewer_id": "reviewer-1",
        "reviewed_utc": "2026-10-27T12:00:00Z",
        "physical_device_group_id": device,
        "capture_session_id": f"session-{device}",
        "image_provenance": "documented controlled image collection",
    })
    row.update(overrides)
    return row


def write_csv(path: Path, rows: list[dict[str, str]], fields=mobile.REQUIRED_FIELDS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_images(root: Path, count: int = 60) -> list[dict[str, str]]:
    rows = []
    for index in range(count):
        relative = f"phones/device_{index:03}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), color=(index, (index * 7) % 256, (index * 11) % 256)).save(path, format="PNG")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(make_review_row(relative, digest, f"device-{index:03}"))
    return rows


def make_protected_manifests(root: Path) -> tuple[Path, ...]:
    manifests = []
    for index in range(4):
        path = root / f"protected_{index}.csv"
        write_csv(path, [], ("source_relative_path", "sha256", "split"))
        manifests.append(path)
    return tuple(manifests)


def write_unresolved(root: Path) -> Path:
    path = root / "unresolved.csv"
    write_csv(path, [], ("source_relative_path", "sha256", "review_status"))
    return path


def test_header_only_intake_is_valid_but_reports_shortage_without_output(tmp_path):
    dataset = tmp_path / "images"
    dataset.mkdir()
    intake = write_csv(tmp_path / "intake.csv", [], mobile.REQUIRED_FIELDS)
    output = tmp_path / "out" / "mobile_split.csv"
    with pytest.raises(mobile.InsufficientDeviceGroupsError, match="0 distinct.*short by 60"):
        mobile.prepare_manifest(
            intake, dataset, output,
            protected_manifests=make_protected_manifests(tmp_path),
            unresolved_review_path=write_unresolved(tmp_path),
        )
    assert not output.exists()


def test_valid_reviewed_rows_create_exact_group_aware_pilot_split(tmp_path):
    dataset = tmp_path / "images"
    intake_rows = make_images(dataset)
    intake = write_csv(tmp_path / "intake.csv", intake_rows)
    output = tmp_path / "out" / "mobile_split.csv"
    summary = mobile.prepare_manifest(
        intake, dataset, output,
        protected_manifests=make_protected_manifests(tmp_path),
        unresolved_review_path=write_unresolved(tmp_path),
    )
    assert summary["split_device_counts"] == {"train": 30, "validation": 10, "test": 20}
    output_rows = mobile.read_csv(output)
    assert len(output_rows) == 60
    assert all(row["review_status"] == "complete" for row in output_rows)
    assert all(row["reviewed_category"] == "mobile" and row["reviewed_family"] == "E_WASTE" for row in output_rows)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sha256": "bad"}, "invalid SHA-256"),
        ({"proposed_category": "battery"}, "proposed labels"),
        ({"reviewed_family": "MSW"}, "reviewed labels"),
        ({"review_status": "complete", "reviewer_id": "", "reviewed_utc": ""}, "reviewer_id"),
        ({"category_decision": ""}, "complete review decisions"),
        ({"reviewed_category": "", "category_decision": "", "review_status": "pending"}, "pending rows"),
        ({"physical_device_group_id": ""}, "physical_device_group_id"),
        ({"reviewed_category": "mobile", "reviewed_utc": "not-a-date"}, "ISO-8601"),
        ({"reviewed_category": "made-up-category"}, "invalid reviewed category"),
    ],
)
def test_rejects_malformed_review_rows(tmp_path, changes, message):
    dataset = tmp_path / "images"
    row = make_images(dataset, 1)[0]
    row.update(changes)
    with pytest.raises(mobile.MobileManifestError, match=message):
        mobile.validate_intake_rows(
            [row], dataset,
            existing_splits=mobile.ExistingSplitRecords(), unresolved_paths=set(), unresolved_hashes=set(),
        )


def test_pending_and_held_rows_never_receive_splits(tmp_path):
    dataset = tmp_path / "images"
    included = make_images(dataset, 60)
    pending = make_images(dataset, 61)[-1]
    pending.update({
        "reviewed_category": "", "reviewed_family": "", "category_decision": "",
        "family_decision": "", "review_status": "pending", "reviewer_id": "",
        "reviewed_utc": "", "physical_device_group_id": "",
        "image_provenance": "",
    })
    held = make_images(dataset, 62)[-1]
    held.update({
        "reviewed_category": "", "reviewed_family": "", "category_decision": "hold",
        "family_decision": "hold", "review_status": "held",
    })
    rows = mobile.validate_intake_rows(
        included + [pending, held], dataset,
        existing_splits=mobile.ExistingSplitRecords(), unresolved_paths=set(), unresolved_hashes=set(),
    )
    mobile.assign_fixed_splits(rows)
    assert rows[-2]["split"] == "" and "pending review" in rows[-2]["exclusion_reason"]
    assert rows[-1]["split"] == "" and "held" in rows[-1]["exclusion_reason"]


def test_rejects_duplicate_paths_and_duplicate_hashes(tmp_path):
    dataset = tmp_path / "images"
    rows = make_images(dataset, 2)
    same_path = dict(rows[0])
    same_path["physical_device_group_id"] = "different-device"
    with pytest.raises(mobile.MobileManifestError, match="duplicate image path"):
        mobile.validate_intake_rows(
            [rows[0], same_path], dataset,
            existing_splits=mobile.ExistingSplitRecords(), unresolved_paths=set(), unresolved_hashes=set(),
        )

    duplicate_hash = dict(rows[1])
    duplicate_hash["sha256"] = rows[0]["sha256"]
    with pytest.raises(mobile.MobileManifestError, match="duplicate image hash"):
        mobile.validate_intake_rows(
            [rows[0], duplicate_hash], dataset,
            existing_splits=mobile.ExistingSplitRecords(), unresolved_paths=set(), unresolved_hashes=set(),
        )


@pytest.mark.parametrize("unsafe_path", ["../secret.jpg", "..\\outside.png", "C:/private.jpg", "/absolute.jpg", "phones//bad.jpg"])
def test_rejects_unsafe_paths(tmp_path, unsafe_path):
    row = make_review_row(unsafe_path, "a" * 64, "device-1")
    with pytest.raises(mobile.MobileManifestError, match="Unsafe source_relative_path"):
        mobile.validate_intake_rows(
            [row], tmp_path,
            existing_splits=mobile.ExistingSplitRecords(), unresolved_paths=set(), unresolved_hashes=set(),
        )


def test_existing_held_out_test_hash_is_forced_to_test(tmp_path):
    dataset = tmp_path / "images"
    rows = make_images(dataset)
    protected = tmp_path / "protected.csv"
    write_csv(protected, [{
        "source_relative_path": rows[0]["source_relative_path"],
        "sha256": rows[0]["sha256"],
        "split": "test",
    }], ("source_relative_path", "sha256", "split"))
    other_protected = []
    for index in range(3):
        path = tmp_path / f"other_{index}.csv"
        write_csv(path, [], ("source_relative_path", "sha256", "split"))
        other_protected.append(path)
    output = tmp_path / "out.csv"
    mobile.prepare_manifest(
        write_csv(tmp_path / "intake.csv", rows), dataset, output,
        protected_manifests=(protected, *other_protected),
        unresolved_review_path=write_unresolved(tmp_path),
    )
    manifest = mobile.read_csv(output)
    protected_row = next(row for row in manifest if row["sha256"] == rows[0]["sha256"])
    assert protected_row["split"] == "test"


def test_rejects_unresolved_disagreement_reuse(tmp_path):
    dataset = tmp_path / "images"
    rows = make_images(dataset)
    unresolved = tmp_path / "unresolved.csv"
    write_csv(unresolved, [{
        "source_relative_path": rows[0]["source_relative_path"],
        "sha256": rows[0]["sha256"],
        "review_status": "pending",
    }], ("source_relative_path", "sha256", "review_status"))
    with pytest.raises(mobile.MobileManifestError, match="unresolved disagreement"):
        mobile.validate_intake_rows(
            rows, dataset,
            existing_splits=mobile.ExistingSplitRecords(),
            unresolved_paths={rows[0]["source_relative_path"]},
            unresolved_hashes={rows[0]["sha256"]},
        )


@pytest.mark.parametrize("leak_field", ["physical_device_group_id", "near_duplicate_group_id"])
def test_rejects_group_leakage_across_splits(tmp_path, leak_field):
    dataset = tmp_path / "images"
    rows = make_images(dataset, 60)
    manifest_rows = [dict(row) for row in rows]
    mobile.assign_fixed_splits(manifest_rows)
    left, right = manifest_rows[0], manifest_rows[1]
    if leak_field == "physical_device_group_id":
        right[leak_field] = left[leak_field]
    else:
        left[leak_field] = right[leak_field] = "near-duplicate-cluster"
        left["split"], right["split"] = "train", "test"
    with pytest.raises(mobile.MobileManifestError, match="cross splits"):
        mobile.validate_prepared_manifest(manifest_rows)


def test_output_path_is_never_overwritten(tmp_path):
    output = tmp_path / "existing.csv"
    output.write_text("keep this content", encoding="utf-8")
    with pytest.raises(FileExistsError, match="[Cc]hoose a new versioned output path"):
        mobile.write_manifest_new(output, [])
    assert output.read_text(encoding="utf-8") == "keep this content"


def test_rejects_image_bytes_changed_after_hash_was_recorded(tmp_path):
    dataset = tmp_path / "images"
    row = make_images(dataset, 1)[0]
    image_path = dataset / row["source_relative_path"]
    Image.new("RGB", (12, 12), color=(200, 150, 100)).save(image_path, format="PNG")
    with pytest.raises(mobile.MobileManifestError, match="SHA-256 does not match"):
        mobile.validate_intake_rows(
            [row], dataset,
            existing_splits=mobile.ExistingSplitRecords(),
            unresolved_paths=set(), unresolved_hashes=set(),
        )


def test_split_assignment_is_deterministic_for_identical_input_and_seed(tmp_path):
    rows = make_images(tmp_path / "images", 65)
    first, second = copy.deepcopy(rows), copy.deepcopy(rows)
    mobile.assign_fixed_splits(first, seed=42)
    mobile.assign_fixed_splits(second, seed=42)
    first_splits = {row["physical_device_group_id"]: row["split"] for row in first}
    second_splits = {row["physical_device_group_id"]: row["split"] for row in second}
    assert first_splits == second_splits
    assert sum(split == "train" for split in first_splits.values()) == 30
    assert sum(split == "validation" for split in first_splits.values()) == 10
    assert sum(split == "test" for split in first_splits.values()) == 20


def test_mobile_validator_rejects_symlink_escape(tmp_path):
    dataset = tmp_path / "images"
    outside = tmp_path / "outside.png"
    outside.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(10, 20, 30)).save(outside, format="PNG")
    linked = dataset / "phones" / "linked.png"
    linked.parent.mkdir(parents=True)
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not supported in this test environment")
    row = make_review_row(
        "phones/linked.png", hashlib.sha256(outside.read_bytes()).hexdigest(), "device-1"
    )
    with pytest.raises(mobile.MobileManifestError, match="symlink"):
        mobile.validate_intake_rows(
            [row], dataset,
            existing_splits=mobile.ExistingSplitRecords(),
            unresolved_paths=set(), unresolved_hashes=set(),
        )


@pytest.mark.parametrize("existing_split", ["train", "validation", "test"])
@pytest.mark.parametrize("match_kind", ["path", "hash"])
def test_existing_splits_protect_matching_paths_and_hashes(tmp_path, existing_split, match_kind):
    dataset = tmp_path / "images"
    rows = make_images(dataset)
    candidate = rows[0]
    manifest_path = tmp_path / "prior.csv"
    if match_kind == "path":
        # A missing legacy hash must still reserve an exact path.
        protected_row = {
            "source_relative_path": candidate["source_relative_path"],
            "split": existing_split,
        }
        fields = ("source_relative_path", "split")
    else:
        protected_row = {
            "source_relative_path": "other/name.png",
            "sha256": candidate["sha256"],
            "split": existing_split,
        }
        fields = ("source_relative_path", "sha256", "split")
    write_csv(manifest_path, [protected_row], fields)
    other_manifests = []
    for index in range(3):
        path = tmp_path / f"other_{index}.csv"
        write_csv(path, [], ("source_relative_path", "sha256", "split"))
        other_manifests.append(path)
    intake = write_csv(tmp_path / "intake.csv", rows)
    output = tmp_path / f"out_{existing_split}_{match_kind}.csv"

    if existing_split in {"train", "validation"}:
        with pytest.raises(mobile.MobileManifestError, match=f"existing {existing_split} split"):
            mobile.prepare_manifest(
                intake, dataset, output,
                protected_manifests=(manifest_path, *other_manifests),
                unresolved_review_path=write_unresolved(tmp_path),
            )
        assert not output.exists()
    else:
        mobile.prepare_manifest(
            intake, dataset, output,
            protected_manifests=(manifest_path, *other_manifests),
            unresolved_review_path=write_unresolved(tmp_path),
        )
        prepared = mobile.read_csv(output)
        matched = next(row for row in prepared if row["source_relative_path"] == candidate["source_relative_path"])
        assert matched["split"] == "test"


def test_default_protected_manifests_remain_active_and_duplicates_are_deduplicated(tmp_path, monkeypatch):
    dataset = tmp_path / "images"
    rows = make_images(dataset)
    default = tmp_path / "default_test.csv"
    extra_default = tmp_path / "extra_default.csv"
    custom = tmp_path / "custom.csv"
    write_csv(default, [{
        "source_relative_path": rows[0]["source_relative_path"],
        "sha256": rows[0]["sha256"],
        "split": "test",
    }], ("source_relative_path", "sha256", "split"))
    for path in (extra_default, custom):
        write_csv(path, [], ("source_relative_path", "sha256", "split"))
    monkeypatch.setattr(mobile, "DEFAULT_PROTECTED_MANIFESTS", (default, extra_default))
    merged = mobile.merge_protected_manifests((default, custom, custom.resolve()))
    assert merged == (default.resolve(), extra_default.resolve(), custom.resolve())

    output = tmp_path / "default_plus_custom.csv"
    mobile.prepare_manifest(
        write_csv(tmp_path / "intake.csv", rows), dataset, output,
        protected_manifests=(default, custom),
        unresolved_review_path=write_unresolved(tmp_path),
    )
    prepared = mobile.read_csv(output)
    matched = next(row for row in prepared if row["sha256"] == rows[0]["sha256"])
    assert matched["split"] == "test"


@pytest.mark.parametrize("invalid_split", ["", "development", "hold"])
def test_existing_split_records_fail_closed_when_split_is_not_established(tmp_path, invalid_split):
    manifest = tmp_path / "ambiguous.csv"
    write_csv(manifest, [{
        "source_relative_path": "phones/device.png",
        "sha256": "a" * 64,
        "split": invalid_split,
    }], ("source_relative_path", "sha256", "split"))
    if invalid_split == "":
        # Blank explicitly denotes an unassigned record; it does not reserve a split.
        loaded = mobile.load_existing_split_records((manifest,))
        assert not loaded.path_splits
    else:
        with pytest.raises(mobile.MobileManifestError, match="Cannot establish existing split"):
            mobile.load_existing_split_records((manifest,))


def test_included_existing_record_without_split_fails_closed(tmp_path):
    manifest = tmp_path / "included_without_split.csv"
    write_csv(manifest, [{
        "source_relative_path": "phones/device.png",
        "sha256": "a" * 64,
        "split": "",
        "group_status": "provisional_include",
    }], ("source_relative_path", "sha256", "split", "group_status"))
    with pytest.raises(mobile.MobileManifestError, match="cannot establish split safely"):
        mobile.load_existing_split_records((manifest,))


def test_existing_path_with_changed_hash_fails_closed(tmp_path):
    dataset = tmp_path / "images"
    row = make_images(dataset, 1)[0]
    manifest = tmp_path / "changed_path_hash.csv"
    write_csv(manifest, [{
        "source_relative_path": row["source_relative_path"],
        "sha256": "f" * 64,
        "split": "test",
    }], ("source_relative_path", "sha256", "split"))
    existing = mobile.load_existing_split_records((manifest,))
    with pytest.raises(mobile.MobileManifestError, match="different image hash"):
        mobile.validate_intake_rows(
            [row], dataset,
            existing_splits=existing,
            unresolved_paths=set(), unresolved_hashes=set(),
        )


def test_path_and_hash_references_with_conflicting_splits_fail_closed(tmp_path):
    dataset = tmp_path / "images"
    row = make_images(dataset, 1)[0]
    path_manifest = tmp_path / "path_train.csv"
    hash_manifest = tmp_path / "hash_test.csv"
    write_csv(path_manifest, [{
        "source_relative_path": row["source_relative_path"],
        "sha256": row["sha256"],
        "split": "train",
    }], ("source_relative_path", "sha256", "split"))
    write_csv(hash_manifest, [{
        "source_relative_path": "alternate/duplicate.png",
        "sha256": row["sha256"],
        "split": "test",
    }], ("source_relative_path", "sha256", "split"))
    existing = mobile.load_existing_split_records((path_manifest, hash_manifest))
    with pytest.raises(mobile.MobileManifestError, match="conflicting splits"):
        mobile.validate_intake_rows(
            [row], dataset,
            existing_splits=existing,
            unresolved_paths=set(), unresolved_hashes=set(),
        )


def test_prepare_manifest_refuses_existing_output_before_writing(tmp_path):
    dataset = tmp_path / "images"
    dataset.mkdir()
    intake = write_csv(tmp_path / "intake.csv", [], mobile.REQUIRED_FIELDS)
    output = tmp_path / "existing.csv"
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="[Cc]hoose a new versioned output path"):
        mobile.prepare_manifest(
            intake, dataset, output,
            protected_manifests=make_protected_manifests(tmp_path),
            unresolved_review_path=write_unresolved(tmp_path),
        )
    assert output.read_text(encoding="utf-8") == "preserve"
