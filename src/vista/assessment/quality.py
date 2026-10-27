"""Training-free quality scoring: contamination and physical condition.

Both scores are image heuristics, not trained models. They are meant to be
tuned on labelled examples later, so thresholds are named constants.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

CONTAMINATION_LEVELS = ((0.33, "clean"), (0.66, "lightly contaminated"), (1.01, "contaminated"))
CONDITION_LEVELS = ((0.15, "intact"), (0.40, "mild"), (0.70, "moderate"), (1.01, "severe"))


@dataclass(frozen=True)
class QualityScore:
    score: float  # 0 (best) .. 1 (worst)
    verdict: str


def _verdict(score: float, levels) -> str:
    for limit, name in levels:
        if score < limit:
            return name
    return levels[-1][1]


def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def contamination_score(image: Image.Image) -> QualityScore:
    """HSV saturation variance + Laplacian texture energy, mapped to 0..1."""
    bgr = _to_bgr(image)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    sat_term = min(float(sat.std()) / 0.30, 1.0)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_term = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 1000.0, 1.0)
    score = round(0.5 * sat_term + 0.5 * lap_term, 4)
    return QualityScore(score, _verdict(score, CONTAMINATION_LEVELS))


def contour_solidity(image: Image.Image) -> float:
    """Contour area / convex-hull area of the largest foreground blob (1.0 = convex)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(_to_bgr(image), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:  # background is white -> invert so the object is foreground
        mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    largest = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    return 1.0 if hull_area <= 0 else float(min(cv2.contourArea(largest) / hull_area, 1.0))


def condition_score(image: Image.Image) -> QualityScore:
    """Deformation from contour solidity: cracked or crushed items score higher."""
    solidity = contour_solidity(image)
    score = round(min(max((1.0 - solidity) / 0.35, 0.0), 1.0), 4)
    return QualityScore(score, _verdict(score, CONDITION_LEVELS))
