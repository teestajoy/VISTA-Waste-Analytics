import csv
from contextlib import nullcontext
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def review_module(monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return __import__("dashboard.review", fromlist=["review"])


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sample_queue(review_module, tmp_path):
    sample_path = tmp_path / "sample.csv"
    rows = [
        {
            "source_relative_path": "paper/paper_1.png",
            "source_folder": "paper",
            "proposed_label": "paper",
        },
        {
            "source_relative_path": "cardboard/cardboard_1.png",
            "source_folder": "cardboard",
            "proposed_label": "cardboard",
        },
    ]
    _write_csv(sample_path, ("source_relative_path", "source_folder", "proposed_label"), rows)
    return review_module.load_review_queue(sample_path)


@pytest.mark.parametrize(
    ("decision", "label", "reason"),
    [
        ("include", "", ""),
        ("hold", "", "  "),
        ("exclude", "", ""),
        ("unknown", "paper", "reason"),
    ],
)
def test_decision_validation_requires_explicit_complete_input(
    review_module, decision, label, reason
):
    with pytest.raises(ValueError):
        review_module.validate_decision(decision, label, reason)


@pytest.mark.parametrize(
    ("decision", "label", "reason"),
    [("include", "cardboard", ""), ("hold", "", "unclear material"), ("exclude", "", "not waste")],
)
def test_decision_validation_accepts_valid_input(review_module, decision, label, reason):
    assert review_module.validate_decision(decision, label, reason) is None


def test_save_and_load_preserve_other_decisions_and_write_only_explicit_rows(
    review_module, sample_queue, tmp_path
):
    review_csv = tmp_path / "review.csv"
    _write_csv(
        review_csv,
        review_module.REVIEW_FIELDS,
        [
            {"source_relative_path": item.source_relative_path, "decision": "", "reviewed_label": "", "reason": ""}
            for item in sample_queue
        ],
    )
    assert review_module.load_saved_decisions(review_csv, sample_queue) == {}

    first, second = sample_queue
    review_module.save_decision(
        review_csv, sample_queue, first.source_relative_path, "include", "paper", ""
    )
    review_module.save_decision(
        review_csv, sample_queue, second.source_relative_path, "hold", "", "mixed material"
    )
    saved = review_module.load_saved_decisions(review_csv, sample_queue)
    assert saved[first.source_relative_path] == {
        "decision": "include", "reviewed_label": "paper", "reason": ""
    }
    assert saved[second.source_relative_path]["decision"] == "hold"
    with review_csv.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2


def test_changing_existing_decision_requires_explicit_confirmation(
    review_module, sample_queue, tmp_path
):
    review_csv = tmp_path / "review.csv"
    item = sample_queue[0]
    review_module.save_decision(review_csv, sample_queue, item.source_relative_path, "include", "paper", "")
    with pytest.raises(ValueError, match="Confirm"):
        review_module.save_decision(
            review_csv, sample_queue, item.source_relative_path, "include", "cardboard", ""
        )
    review_module.save_decision(
        review_csv,
        sample_queue,
        item.source_relative_path,
        "include",
        "cardboard",
        "",
        confirm_replace=True,
    )
    assert review_module.load_saved_decisions(review_csv, sample_queue)[item.source_relative_path]["reviewed_label"] == "cardboard"


@pytest.mark.parametrize("bad_path", ["../outside.png", "paper/../../outside.png", "C:\\outside.png", "/outside.png"])
def test_unsafe_paths_are_rejected(review_module, tmp_path, bad_path):
    item = review_module.ReviewQueueItem(bad_path, "paper", "paper")
    with pytest.raises(ValueError, match="Unsafe"):
        review_module.resolve_image_path(tmp_path, item)


def test_symlink_outside_dataset_is_rejected_when_supported(review_module, tmp_path):
    dataset = tmp_path / "dataset"
    outside = tmp_path / "outside.png"
    (dataset / "paper").mkdir(parents=True)
    Image.new("RGB", (2, 2), "red").save(outside)
    link = dataset / "paper" / "linked.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment.")
    item = review_module.ReviewQueueItem("paper/linked.png", "paper", "paper")
    with pytest.raises(ValueError, match="outside"):
        review_module.resolve_image_path(dataset, item)


def test_load_review_image_returns_rgb_image_and_reports_missing_or_corrupt(
    review_module, tmp_path
):
    root = tmp_path / "dataset"
    category = root / "paper"
    category.mkdir(parents=True)
    image_path = category / "sample.png"
    Image.new("RGB", (3, 2), "green").save(image_path)
    item = review_module.ReviewQueueItem("paper/sample.png", "paper", "paper")
    image = review_module.load_review_image(root, item)
    assert image.size == (3, 2)
    assert image.mode == "RGB"

    missing = review_module.ReviewQueueItem("paper/missing.png", "paper", "paper")
    with pytest.raises(FileNotFoundError, match="not found"):
        review_module.load_review_image(root, missing)
    corrupt_path = category / "broken.png"
    corrupt_path.write_text("not an image", encoding="utf-8")
    corrupt = review_module.ReviewQueueItem("paper/broken.png", "paper", "paper")
    with pytest.raises(ValueError, match="Could not read image"):
        review_module.load_review_image(root, corrupt)


def test_progress_counts_only_explicit_decisions(review_module, sample_queue):
    counts = review_module.progress_counts(
        sample_queue,
        {
            sample_queue[0].source_relative_path: {
                "decision": "include", "reviewed_label": "paper", "reason": ""
            },
            sample_queue[1].source_relative_path: {
                "decision": "hold", "reviewed_label": "", "reason": "unclear"
            },
        },
    )
    assert counts == {"reviewed": 2, "included": 1, "held": 1, "excluded": 0}


def test_pilot_review_save_button_persists_include_and_refreshes_progress(
    review_module, monkeypatch, tmp_path
):
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "paper" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), "white").save(image_path)

    queue_csv = tmp_path / "sample.csv"
    _write_csv(
        queue_csv,
        ("source_relative_path", "source_folder", "proposed_label"),
        [{
            "source_relative_path": "paper/sample.png",
            "source_folder": "paper",
            "proposed_label": "paper",
        }],
    )
    review_csv = tmp_path / "review.csv"
    _write_csv(
        review_csv,
        review_module.REVIEW_FIELDS,
        [{"source_relative_path": "paper/sample.png", "decision": "", "reviewed_label": "", "reason": ""}],
    )
    monkeypatch.setattr(review_module, "DATASET_ROOT", dataset_root)
    monkeypatch.setattr(review_module, "SAMPLE_CSV_PATH", queue_csv)
    monkeypatch.setattr(review_module, "REVIEW_CSV_PATH", review_csv)

    class FakeColumn:
        def __init__(self, metrics):
            self.metrics = metrics

        def metric(self, label, value):
            self.metrics.append((label, value))

    class FakeProgressSlot:
        def empty(self):
            pass

        def container(self):
            return nullcontext()

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.metrics = []
            self.messages = []
            self.selected = {
                "Review queue image": 0,
                "Decision": "Include",
                "Confirm the correct label": "paper",
            }

        def __getattr__(self, name):
            if name in {"subheader", "caption", "code", "write", "image"}:
                return lambda *args, **kwargs: None
            if name in {"error", "info", "success"}:
                return lambda message, **kwargs: self.messages.append((name, message))
            raise AttributeError(name)

        def empty(self):
            return FakeProgressSlot()

        def progress(self, value, text=""):
            self.progress_value = value
            self.progress_text = text

        def columns(self, count):
            return [FakeColumn(self.metrics) for _ in range(count)]

        def selectbox(self, label, options, **kwargs):
            return self.selected[label]

        def radio(self, label, options, **kwargs):
            return "Single image"

        def text_area(self, *args, **kwargs):
            return ""

        def checkbox(self, *args, **kwargs):
            return False

        def button(self, *args, **kwargs):
            return True

    fake_st = FakeStreamlit()
    review_module.render_review_page(fake_st)

    saved = review_module.load_saved_decisions(review_csv, review_module.load_review_queue(queue_csv))
    assert saved["paper/sample.png"] == {
        "decision": "include", "reviewed_label": "paper", "reason": ""
    }
    assert fake_st.metrics[-4:] == [
        ("Reviewed", 1),
        ("Included", 1),
        ("Held", 0),
        ("Excluded", 0),
    ]
    assert ("success", "Decision saved.") in fake_st.messages


def test_batch_save_writes_only_explicit_decisions_and_updates_progress(
    review_module, sample_queue, tmp_path
):
    review_csv = tmp_path / "review.csv"
    first, second = sample_queue
    review_module.save_decisions(
        review_csv,
        sample_queue,
        {
            first.source_relative_path: {
                "decision": "include",
                "reviewed_label": "paper",
                "reason": "",
            }
        },
    )

    review_module.save_decisions(
        review_csv,
        sample_queue,
        {
            second.source_relative_path: {
                "decision": "hold",
                "reviewed_label": "",
                "reason": "Mixed material; needs closer review.",
            }
        },
    )

    saved = review_module.load_saved_decisions(review_csv, sample_queue)
    assert set(saved) == {first.source_relative_path, second.source_relative_path}
    assert saved[first.source_relative_path]["reviewed_label"] == "paper"
    assert saved[second.source_relative_path]["decision"] == "hold"
    assert review_module.progress_counts(sample_queue, saved) == {
        "reviewed": 2,
        "included": 1,
        "held": 1,
        "excluded": 0,
    }
    with review_csv.open("r", encoding="utf-8", newline="") as source:
        assert len(list(csv.DictReader(source))) == 2


@pytest.mark.parametrize(
    "bad_value",
    [
        {"decision": "include", "reviewed_label": "", "reason": ""},
        {"decision": "hold", "reviewed_label": "", "reason": "  "},
        {"decision": "exclude", "reviewed_label": "", "reason": ""},
    ],
)
def test_batch_save_rejects_incomplete_decisions_without_writing(
    review_module, sample_queue, tmp_path, bad_value
):
    review_csv = tmp_path / "review.csv"
    with pytest.raises(ValueError):
        review_module.save_decisions(
            review_csv,
            sample_queue,
            {sample_queue[0].source_relative_path: bad_value},
        )
    assert not review_csv.exists()


def test_batch_save_requires_confirmation_to_replace_existing_decision(
    review_module, sample_queue, tmp_path
):
    review_csv = tmp_path / "review.csv"
    first, second = sample_queue
    original = {
        first.source_relative_path: {
            "decision": "include",
            "reviewed_label": "paper",
            "reason": "",
        }
    }
    review_module.save_decisions(review_csv, sample_queue, original)
    replacement = {
        first.source_relative_path: {
            "decision": "include",
            "reviewed_label": "cardboard",
            "reason": "",
        },
        second.source_relative_path: {
            "decision": "exclude",
            "reviewed_label": "",
            "reason": "Not a waste item.",
        },
    }

    with pytest.raises(ValueError, match="Confirm replacing"):
        review_module.save_decisions(review_csv, sample_queue, replacement)
    assert review_module.load_saved_decisions(review_csv, sample_queue) == original

    review_module.save_decisions(
        review_csv,
        sample_queue,
        replacement,
        confirmed_replacements={first.source_relative_path},
    )
    saved = review_module.load_saved_decisions(review_csv, sample_queue)
    assert saved[first.source_relative_path]["reviewed_label"] == "cardboard"
    assert saved[second.source_relative_path]["decision"] == "exclude"
