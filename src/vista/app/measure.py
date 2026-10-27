"""Contamination proxy, condition (solidity) heuristic and carbon-economic estimate, all with their raw measurements."""
import cv2
import numpy as np
from PIL import Image

from vista.assessment.quality import CONDITION_LEVELS, CONTAMINATION_LEVELS, _verdict
from vista.impact.carbon import REFERENCE_TABLE

CONT_PARAMS = {"sat_std_norm": 0.30, "laplacian_var_norm": 1000.0, "weights": [0.5, 0.5]}
COND_PARAMS = {"solidity_floor": 0.65, "segmentation": "Gaussian blur 5x5 + Otsu, border-based polarity, largest external contour"}


def _bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def contamination(crop: Image.Image) -> dict:
    bgr = _bgr(crop)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    sat_var, sat_std = float(sat.var()), float(sat.std())
    lap = float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    a, b = min(sat_std / CONT_PARAMS["sat_std_norm"], 1.0), min(lap / CONT_PARAMS["laplacian_var_norm"], 1.0)
    score = round(0.5 * a + 0.5 * b, 4)
    return {"score": score, "verdict": _verdict(score, CONTAMINATION_LEVELS), "saturation_variance": round(sat_var, 6),
            "saturation_std": round(sat_std, 6), "laplacian_energy": round(lap, 3),
            "formula": "score = 0.5*min(std(S)/0.30,1) + 0.5*min(Var(Laplacian(gray))/1000,1), S in [0,1] from HSV", "parameters": CONT_PARAMS,
            "label": "image-based contamination PROXY (not validated against human contamination labels)",
            "limitations": "Affected by material colour, printing, reflections, lighting and background pixels."}


def condition(crop: Image.Image) -> tuple[dict, Image.Image | None]:
    bgr = _bgr(crop)
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    base = {"formula": "solidity = contour_area / convex_hull_area; score = clip((1-solidity)/0.35, 0, 1)", "parameters": COND_PARAMS,
            "label": "visible shape-irregularity heuristic (not a verified damage or functionality assessment)",
            "limitations": "Segmentation errors, occlusion, crop edges and natural shapes also lower solidity."}
    if min(gray.shape) < 8:
        return {**base, "score": None, "verdict": "unavailable", "status": "crop too small"}, None
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return {**base, "score": None, "verdict": "unavailable", "status": "no foreground contour found"}, None
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    area, harea = float(cv2.contourArea(c)), float(cv2.contourArea(hull))
    if harea < 1e-6 or area < 16:
        return {**base, "score": None, "verdict": "unavailable", "status": "degenerate contour"}, None
    sol = min(area / harea, 1.0)
    score = round(min(max((1 - sol) / 0.35, 0.0), 1.0), 4)
    ov = bgr.copy()
    cv2.drawContours(ov, [hull], -1, (255, 160, 0), 2)
    cv2.drawContours(ov, [c], -1, (0, 220, 0), 2)
    return {**base, "score": score, "verdict": _verdict(score, CONDITION_LEVELS), "solidity": round(sol, 5),
            "contour_area": round(area, 1), "convex_hull_area": round(harea, 1), "status": "ok"}, \
        Image.fromarray(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))


def impact(category: str, mass_kg: float | None) -> dict:
    row = REFERENCE_TABLE.get(category)
    if row is None:
        return {"status": "Not available", "reason": f"no reference factor for '{category}'"}
    f, price, default_mass = row
    method = "user-entered mass" if mass_kg else "category default mass (assumption)"
    m = mass_kg if mass_kg else default_mass
    return {"status": "ok", "category": category, "mass_kg": m, "mass_method": method,
            "co2e_avoided_kg": round(f * m, 5), "co2e_factor_kg_per_kg": f, "recovery_value_usd": round(price * m, 5),
            "price_usd_per_kg": price, "formula": "CO2e = factor x mass ; value = price x mass",
            "source": "ILLUSTRATIVE PLACEHOLDER factors (src/vista/impact/carbon.py) - replace with cited LCA/market data before quoting",
            "factors_are_placeholders": True, "geographic_scope": "unspecified", "reference_year": None,
            "notice": "Mass cannot be measured from a single image. Market values vary by location, purity, quantity and time; "
                      "this is an indicative gross value, not resale proceeds. Not related to classification confidence."}
