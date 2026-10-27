"""Show generic YOLO detections from the default webcam."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from vista.detection.object_detector import Detection


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
MODEL_PATH = "yolo26n.pt"
CAMERA_INDEX = 0
PANEL_WIDTH = 520
PANEL_PADDING = 18
FONT_SIZE = 20
TITLE_FONT_SIZE = 24
LINE_SPACING = 7
SECTION_SPACING = 10
GENERAL_DETECTION_NOTICE = (
    "General-purpose YOLO detections — not VISTA waste classifications."
)
EXPLAINABILITY_NOTICE = (
    "VISTA explainability unavailable — no verified waste classifier or "
    "Grad-CAM implementation."
)


def panel_text(detections: list[Detection]) -> list[str]:
    """Return honest status and detection text for the side panel."""
    lines = [GENERAL_DETECTION_NOTICE, EXPLAINABILITY_NOTICE, "Detected objects:"]
    if not detections:
        lines.append("No objects detected.")
    else:
        lines.extend(
            f"{detection.class_label} — confidence {detection.confidence:.2f}"
            for detection in detections
        )
    return lines


def _load_font(size: int) -> ImageFont.ImageFont:
    windows_fonts = Path("C:/Windows/Fonts")
    candidates = [
        windows_fonts / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path(ImageFont.__file__).parent / "fonts" / "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Older Pillow versions do not accept a size argument.
        return ImageFont.load_default()


def _wrap_text(
    text: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw, max_width: int
) -> list[str]:
    """Wrap panel text to fit its available pixel width."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def render_demo_view(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return the webcam frame with boxes and a readable information panel."""
    annotated = frame.copy()
    frame_height, frame_width = annotated.shape[:2]
    for detection in detections:
        box = detection.box
        x_min = max(0, min(frame_width - 1, int(round(box.x_min))))
        y_min = max(0, min(frame_height - 1, int(round(box.y_min))))
        x_max = max(0, min(frame_width - 1, int(round(box.x_max))))
        y_max = max(0, min(frame_height - 1, int(round(box.y_max))))
        if x_max <= x_min or y_max <= y_min:
            continue
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{detection.class_label} {detection.confidence:.2f}"
        label_y = max(18, y_min - 8)
        cv2.putText(
            annotated,
            label,
            (x_min, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    panel = Image.new("RGB", (PANEL_WIDTH, frame_height), (30, 34, 42))
    draw = ImageDraw.Draw(panel)
    title_font = _load_font(TITLE_FONT_SIZE)
    body_font = _load_font(FONT_SIZE)
    available_width = PANEL_WIDTH - 2 * PANEL_PADDING
    panel_lines: list[tuple[str, ImageFont.ImageFont, bool]] = []
    messages = panel_text(detections)
    for index, message in enumerate(messages):
        font = title_font if index == 2 else body_font
        panel_lines.extend(
            (line, font, index == 2)
            for line in _wrap_text(message, font, draw, available_width)
        )

    line_heights = [
        max(1, draw.textbbox((0, 0), line, font=font)[3])
        for line, font, _ in panel_lines
    ]
    heading_count = sum(is_heading for _, _, is_heading in panel_lines)
    content_height = (
        sum(line_heights)
        + LINE_SPACING * max(0, len(panel_lines) - 1)
        + SECTION_SPACING * heading_count
    )
    canvas_height = max(frame_height, 2 * PANEL_PADDING + content_height)
    if canvas_height != frame_height:
        taller_panel = Image.new("RGB", (PANEL_WIDTH, canvas_height), (30, 34, 42))
        taller_panel.paste(panel, (0, 0))
        panel = taller_panel
        draw = ImageDraw.Draw(panel)

    y = PANEL_PADDING
    for (line, font, is_heading), line_height in zip(panel_lines, line_heights):
        color = (255, 255, 255) if is_heading else (225, 230, 238)
        draw.text((PANEL_PADDING, y), line, fill=color, font=font)
        y += line_height + LINE_SPACING
        if is_heading:
            y += SECTION_SPACING

    canvas = np.zeros((canvas_height, frame_width + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:frame_height, :frame_width] = annotated
    panel_bgr = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    canvas[:, frame_width:] = panel_bgr
    return canvas


def main() -> int:
    camera = None
    try:
        if str(PROJECT_SRC) not in sys.path:
            sys.path.insert(0, str(PROJECT_SRC))
        from vista.detection.object_detector import YoloObjectDetector

        print(f"Loading general-purpose YOLO weights: {MODEL_PATH}")
        detector = YoloObjectDetector(model_path=MODEL_PATH)

        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            print(
                f"Could not open camera {CAMERA_INDEX}. Check that a webcam is "
                "connected and available to Windows."
            )
            return 1

        print("Webcam is running. Press q or Escape to quit.")
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Could not read a frame from the webcam.")
                return 1

            try:
                detections = detector.detect_frame(frame)
            except Exception as error:
                print(f"YOLO inference failed for this webcam frame: {error}")
                return 1

            cv2.imshow("VISTA Webcam Detection Demo", render_demo_view(frame, detections))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
    except Exception as error:
        print(f"Could not start the VISTA webcam demo: {error}")
        return 1
    finally:
        try:
            if camera is not None:
                camera.release()
        finally:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # Headless OpenCV builds may not provide a GUI window backend.
                pass


if __name__ == "__main__":
    raise SystemExit(main())
