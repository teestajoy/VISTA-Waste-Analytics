"""Compare the current category checkpoint with zero-shot CLIP on reviewed test images.

CLIP cosine similarities are ranking scores, not calibrated probabilities. This
experiment makes no confidence/abstention decision for CLIP and never changes
the production classifier or its 0.70 cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

CATEGORY_NAMES = ("paper", "cardboard", "plastic", "glass", "organic", "battery")
REVIEWED_PROVENANCE = "explicit human-reviewed category decision"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "reviewed_test_clip_category_comparison_v3.json"
DEFAULT_REVIEW_REPORT = (
    PROJECT_ROOT / "data" / "processed" / "reviewed_test_model_evaluation_independent_blind_batch.json"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "waste_category_provisional" / "split_manifest.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "waste_category_provisional" / "best_model.pth"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_DATASET_ROOT = Path(r"C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset")
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

# Define prompts before evaluating. Descriptions explicitly contrast the most
# easily confused materials; they are not adapted to evaluation outcomes.
CATEGORY_PROMPTS = {
    "paper": (
        "A photo of loose paper waste, such as a single sheet or notebook page; not thick cardboard packaging.",
        "A close photo of recyclable paper material, a sheet of paper rather than a box.",
    ),
    "cardboard": (
        "A photo of cardboard packaging, corrugated board, or a cardboard box; thicker and stiffer than a sheet of paper.",
        "A close photo of a piece of waste cardboard, not loose paper.",
    ),
    "plastic": (
        "A photo of a plastic waste item, such as a lightweight transparent PET drink bottle with thin flexible walls; not glass.",
        "A close photo of a disposable plastic bottle or plastic packaging, not a rigid glass bottle.",
    ),
    "glass": (
        "A photo of a rigid glass bottle, glass tumbler, or glass container; not a lightweight plastic bottle.",
        "A close photo of a waste item made of glass, with rigid transparent glass walls.",
    ),
    "organic": (
        "A photo of food waste, such as a banana or leftover food; not an insect or non-food object.",
        "A close photo of discarded edible food scraps without packaging.",
    ),
    "battery": (
        "A photo of an individual household battery, such as an AA or cylindrical cell, for electronic-waste sorting.",
        "A close photo of a used household battery, not an ordinary plastic household object.",
    ),
}


class ComparisonError(ValueError):
    """Raised when the review/evaluation inputs are unsafe or incomplete."""


def safe_relative_path(value: str) -> PurePosixPath:
    normalized = (value or "").strip().replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath((value or "").strip())
    if (
        not normalized
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ComparisonError(f"Unsafe source-relative path: {value!r}")
    return posix


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect_reviewed_test_images(
    review_report: Path,
    manifest_path: Path,
    dataset_root: Path,
) -> tuple[list[dict], list[dict]]:
    """Select explicit human-reviewed category cases from canonical test rows only."""
    report_data = json.loads(Path(review_report).read_text(encoding="utf-8"))
    manifest_rows = _read_csv(manifest_path)
    by_path = {row.get("source_relative_path", "").replace("\\", "/"): row for row in manifest_rows}
    train_validation_hashes = {
        (row.get("sha256") or "").strip().lower()
        for row in manifest_rows
        if row.get("group_status") == "provisional_include"
        and row.get("split") in {"train", "validation"}
    }
    excluded: list[dict] = []
    candidates: list[dict] = []
    seen_hashes: set[str] = set()
    root = Path(dataset_root).resolve(strict=True)

    for record in report_data.get("predictions", []):
        if record.get("task") != "category":
            continue
        relative = record.get("source_relative_path") or ""
        try:
            safe = safe_relative_path(relative)
        except ComparisonError as error:
            excluded.append({"source_relative_path": relative, "reason": str(error)})
            continue
        digest = (record.get("sha256") or "").strip().lower()
        expected = record.get("expected_label")
        if record.get("label_provenance") != REVIEWED_PROVENANCE or expected not in CATEGORY_NAMES:
            excluded.append({"source_relative_path": relative, "reason": "not an explicit supported human-reviewed category label"})
            continue
        manifest = by_path.get(safe.as_posix())
        if (
            manifest is None
            or manifest.get("group_status") != "provisional_include"
            or manifest.get("canonical_path", "").replace("\\", "/") != safe.as_posix()
            or manifest.get("split") != "test"
        ):
            excluded.append({"source_relative_path": relative, "reason": "not a canonical held-out test manifest row"})
            continue
        if digest != (manifest.get("sha256") or "").strip().lower():
            excluded.append({"source_relative_path": relative, "reason": "review report hash does not match canonical test manifest"})
            continue
        if digest in train_validation_hashes:
            excluded.append({"source_relative_path": relative, "reason": "hash overlaps training or validation data"})
            continue
        if digest in seen_hashes:
            excluded.append({"source_relative_path": relative, "reason": "duplicate hash already selected"})
            continue
        target = root.joinpath(*safe.parts)
        try:
            resolved = target.resolve(strict=True)
            resolved.relative_to(root)
            current = root
            for part in safe.parts:
                current = current / part
                if current.is_symlink():
                    raise ComparisonError("source path traverses a symlink")
            if not resolved.is_file() or sha256_file(resolved) != digest:
                raise ComparisonError("source file is missing or its SHA-256 changed")
            with Image.open(resolved) as image:
                image = image.convert("RGB")
                image.load()
        except (OSError, RuntimeError, ValueError, UnidentifiedImageError, ComparisonError) as error:
            excluded.append({"source_relative_path": relative, "reason": f"source image validation failed: {error}"})
            continue
        seen_hashes.add(digest)
        candidates.append({
            "source_relative_path": safe.as_posix(),
            "sha256": digest,
            "expected_label": expected,
            "image_path": resolved,
        })
    return candidates, excluded


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def summarize_metrics(expected: list[str], predicted: list[str], *, include_uncertain: bool) -> dict:
    labels = list(CATEGORY_NAMES)
    columns = labels + (["Uncertain"] if include_uncertain else [])
    matrix = {actual: {guess: 0 for guess in columns} for actual in labels}
    for actual, guess in zip(expected, predicted):
        matrix[actual][guess] += 1
    per_class = {}
    f1_values = []
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in labels)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
    total = len(expected)
    return {
        "examples": total,
        "accuracy": sum(a == p for a, p in zip(expected, predicted)) / total if total else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def select_largest_non_person_detection(detections):
    """Use a generic detection only as an ROI proposal, never as a waste label."""
    eligible = [detection for detection in detections if detection.class_label.strip().casefold() != "person"]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda detection: (
            max(0.0, detection.box.x_max - detection.box.x_min)
            * max(0.0, detection.box.y_max - detection.box.y_min),
            detection.confidence,
        ),
    )


def evaluate_yolo_crop_diagnostics(cases: list[dict], detector, category_classifier) -> dict:
    """Evaluate one largest non-person YOLO-proposed crop per reviewed image."""
    from vista.detection.object_cropper import crop_detection

    expected = []
    predicted = []
    records = []
    no_detection = 0
    crop_errors = 0
    for case in cases:
        expected.append(case["expected_label"])
        status = "uncertain"
        selected = None
        try:
            detections = detector.detect(case["image_path"])
            selected = select_largest_non_person_detection(detections)
            if selected is None:
                no_detection += 1
                status = "no non-person proposal"
            else:
                with Image.open(case["image_path"]) as image:
                    source = image.convert("RGB")
                    source.load()
                    crop = crop_detection(source, selected)
                prediction = category_classifier.predict(crop)
                status = "uncertain" if prediction.is_uncertain else "predicted"
                label = "Uncertain" if prediction.is_uncertain else prediction.category.value
                predicted.append(label)
                records.append({
                    "source_relative_path": case["source_relative_path"],
                    "expected_category": case["expected_label"],
                    "proposal_label_generic_only": selected.class_label,
                    "proposal_confidence_generic_only": selected.confidence,
                    "crop_box_xyxy": [selected.box.x_min, selected.box.y_min, selected.box.x_max, selected.box.y_max],
                    "category_prediction": label,
                    "top_candidate": prediction.candidate_category.value,
                    "raw_softmax_score": prediction.score,
                    "uncertain_at_existing_0_70_cutoff": bool(prediction.is_uncertain),
                    "status": status,
                })
                continue
        except Exception as error:
            crop_errors += 1
            status = f"crop/inference error: {error}"
        predicted.append("Uncertain")
        records.append({
            "source_relative_path": case["source_relative_path"],
            "expected_category": case["expected_label"],
            "proposal_label_generic_only": selected.class_label if selected else None,
            "proposal_confidence_generic_only": selected.confidence if selected else None,
            "category_prediction": "Uncertain",
            "top_candidate": None,
            "raw_softmax_score": None,
            "uncertain_at_existing_0_70_cutoff": True,
            "status": status,
        })
    return {
        "method": "largest non-person generic YOLO box, then the existing provisional category classifier",
        "generic_label_use": "ROI proposal metadata only; never interpreted as a waste category",
        "examples": len(cases),
        "images_without_non_person_proposal": no_detection,
        "crop_or_inference_errors": crop_errors,
        "crop_available_coverage": (len(cases) - no_detection - crop_errors) / len(cases) if cases else 0.0,
        "uncertain_count_including_missing_crops": predicted.count("Uncertain"),
        "metrics_with_uncertain_abstentions": summarize_metrics(expected, predicted, include_uncertain=True),
        "predictions": records,
    }


class ClipScorer:
    """Zero-shot category ranking using class-level prompt ensembles."""

    def __init__(self, model, processor, device):
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = device
        self.model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
        self.transformers_version = None
        prompts = [prompt for category in CATEGORY_NAMES for prompt in CATEGORY_PROMPTS[category]]
        inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
        grouped = []
        offset = 0
        for category in CATEGORY_NAMES:
            count = len(CATEGORY_PROMPTS[category])
            mean = text_features[offset:offset + count].mean(dim=0, keepdim=True)
            grouped.append(torch.nn.functional.normalize(mean, dim=-1)[0])
            offset += count
        self.text_features = torch.stack(grouped)

    def predict(self, image: Image.Image) -> dict:
        import torch

        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            image_features = self.model.get_image_features(pixel_values=pixels)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
            scores = (image_features @ self.text_features.T)[0]
        values, indices = scores.topk(2)
        return {
            "ranked_scores": {CATEGORY_NAMES[int(i)]: float(v) for i, v in zip(indices, values)},
            "top_category": CATEGORY_NAMES[int(indices[0])],
            "top_similarity": float(values[0]),
            "top2_margin": float(values[0] - values[1]),
        }


def load_clip_scorer(model_id: str, device: str, *, local_files_only: bool = False) -> ClipScorer:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise ComparisonError(
            "Transformers is not installed. Install it in the project venv to run the optional CLIP experiment."
        ) from error
    try:
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        model = CLIPModel.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception as error:
        raise ComparisonError(f"Could not load CLIP model {model_id!r}: {error}") from error
    import torch

    selected_device = torch.device(device)
    scorer = ClipScorer(model, processor, selected_device)
    import transformers

    scorer.transformers_version = transformers.__version__
    return scorer


def evaluate_comparison(cases: list[dict], category_classifier, clip_scorer, *, model_id: str = DEFAULT_CLIP_MODEL) -> dict:
    expected: list[str] = []
    baseline_top1_labels: list[str] = []
    baseline_labels: list[str] = []
    clip_labels: list[str] = []
    uncertain_count = 0
    predictions = []
    for case in cases:
        with Image.open(case["image_path"]) as image:
            image = image.convert("RGB")
            image.load()
        baseline = category_classifier.predict(image)
        clip = clip_scorer.predict(image)
        expected_label = case["expected_label"]
        uncertain = bool(baseline.is_uncertain)
        baseline_label = "Uncertain" if uncertain else baseline.category.value
        baseline_top1 = baseline.candidate_category.value
        uncertain_count += int(uncertain)
        expected.append(expected_label)
        baseline_top1_labels.append(baseline_top1)
        baseline_labels.append(baseline_label)
        clip_labels.append(clip["top_category"])
        predictions.append({
            "source_relative_path": case["source_relative_path"],
            "sha256": case["sha256"],
            "expected_category": expected_label,
            "current_model": {
                "prediction": baseline_label,
                "top_candidate": baseline_top1,
                "raw_softmax_score": baseline.score,
                "uncertain_at_existing_0_70_cutoff": uncertain,
            },
            "clip": clip,
        })
    return {
        "evaluation_status": "Independent human-reviewed held-out category comparison; both models evaluated on identical images.",
        "expected_label_provenance": REVIEWED_PROVENANCE,
        "clip_model": model_id,
        "clip_model_revision": getattr(clip_scorer, "model_revision", None),
        "transformers_version": getattr(clip_scorer, "transformers_version", None),
        "torch_version": torch.__version__,
        "device": str(getattr(clip_scorer, "device", "unknown")),
        "prompts_by_category": CATEGORY_PROMPTS,
        "clip_score_note": "Cosine similarities are ranking scores, not probabilities; CLIP has no calibrated abstention threshold in this experiment.",
        "current_model_uncertain_count": uncertain_count,
        "current_model_top1": summarize_metrics(expected, baseline_top1_labels, include_uncertain=False),
        "current_model_operating_point": summarize_metrics(expected, baseline_labels, include_uncertain=True),
        "current_model_uncertainty": {
            "uncertain_count": uncertain_count,
            "coverage": (len(cases) - uncertain_count) / len(cases) if cases else 0.0,
            "threshold": 0.70,
            "threshold_note": "Existing uncalibrated cutoff; unchanged for this comparison.",
        },
        "clip_top1": summarize_metrics(expected, clip_labels, include_uncertain=False),
        "clip_uncertainty": "Not calibrated; inspect top-two similarity margin as a diagnostic only.",
        "predictions": predictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--review-report", type=Path, default=DEFAULT_REVIEW_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--category-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--compare-yolo-crops", action="store_true", help="Also evaluate the largest non-person YOLO-proposed crop per image.")
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing report: {args.output}")
    try:
        cases, excluded = collect_reviewed_test_images(args.review_report, args.manifest, args.dataset_root)
        if not cases:
            raise ComparisonError("No independently reviewed canonical held-out category images passed validation.")
        from vista.classification.provisional_waste_category_classifier import ProvisionalWasteCategoryClassifier
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        category_classifier = ProvisionalWasteCategoryClassifier(args.category_checkpoint, device=device)
        clip_scorer = load_clip_scorer(args.clip_model, device, local_files_only=args.local_files_only)
        report = evaluate_comparison(cases, category_classifier, clip_scorer, model_id=args.clip_model)
        if args.compare_yolo_crops:
            if not args.yolo_model.is_file():
                raise ComparisonError(f"YOLO checkpoint is not cached at the configured path: {args.yolo_model}")
            from vista.detection.object_detector import YoloObjectDetector

            detector = YoloObjectDetector(model_path=args.yolo_model)
            report["yolo_crop_diagnostics"] = evaluate_yolo_crop_diagnostics(
                cases, detector, category_classifier
            )
            report["yolo_model_path"] = str(args.yolo_model.resolve())
    except (OSError, ValueError, ComparisonError, RuntimeError) as error:
        parser.exit(2, f"Comparison stopped safely: {error}\n")
    report["excluded_count"] = len(excluded)
    report["excluded_records"] = excluded
    report["category_checkpoint_path"] = str(args.category_checkpoint.resolve())
    report["category_checkpoint_sha256"] = sha256_file(args.category_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved comparison report: {args.output}")
    summary = {
        "examples": len(cases),
        "current_model_top1": report["current_model_top1"],
        "current_model_operating_point": report["current_model_operating_point"],
        "current_model_uncertainty": report["current_model_uncertainty"],
        "current_model_uncertain_count": report["current_model_uncertain_count"],
        "clip_top1": report["clip_top1"],
        "excluded": len(excluded),
        "clip_score_note": report["clip_score_note"],
    }
    if "yolo_crop_diagnostics" in report:
        crop = report["yolo_crop_diagnostics"]
        summary["yolo_crop_diagnostics"] = {
            key: crop[key] for key in (
                "examples", "crop_available_coverage", "images_without_non_person_proposal",
                "crop_or_inference_errors", "uncertain_count_including_missing_crops",
            )
        }
        summary["yolo_crop_diagnostics"]["metrics"] = crop["metrics_with_uncertain_abstentions"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
