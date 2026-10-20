import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pretrained YOLO detector on one local image."
    )
    parser.add_argument("image", type=Path, help="Path to a local image file")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image path does not exist: {args.image}")
    if not args.image.is_file():
        raise SystemExit(f"Image path is not a file: {args.image}")

    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "PyTorch is not installed. Install a compatible CPU or CUDA build first."
        ) from error

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    project_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(project_src))

    from vista.detection.object_detector import YoloObjectDetector

    detector = YoloObjectDetector(model_path="yolo26n.pt")
    detections = detector.detect(args.image)

    if not detections:
        print("No objects detected.")
        return

    for detection in detections:
        box = detection.box
        print(
            f"{detection.class_label}: "
            f"confidence={detection.confidence:.3f}, "
            f"box=({box.x_min:.1f}, {box.y_min:.1f}, "
            f"{box.x_max:.1f}, {box.y_max:.1f})"
        )


if __name__ == "__main__":
    main()
