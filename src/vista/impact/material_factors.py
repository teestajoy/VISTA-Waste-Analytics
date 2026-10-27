"""Configurable, provenance-aware factors for optional material estimates.

The formulas are:

* potential CO2e avoidance (kg) = measured mass (kg) * factor (kg CO2e/kg)
* estimated material value = measured mass (kg) * factor (currency/kg)

Every shipped factor is intentionally ``None``: this repository has no approved,
locally applicable source for category-level emissions or material prices. Add a
factor only after documenting its source, geography, date, and assumptions below.
These outputs are potential estimates, never measured savings or guaranteed prices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


SUPPORTED_CATEGORIES = ("paper", "cardboard", "plastic", "glass", "organic", "battery")
CARBON_UNIT = "kg CO2e potential avoided"
WEIGHT_UNIT = "g"
PLACEHOLDER_STATUS = (
    "Placeholder — no approved, applicable factor source is recorded in this project."
)


@dataclass(frozen=True)
class MaterialFactor:
    """Per-kilogram factors and provenance for one material category."""

    potential_co2e_kg_per_kg: float | None = None
    carbon_source: str = PLACEHOLDER_STATUS
    material_value_per_kg: float | None = None
    currency: str | None = None
    value_source: str = PLACEHOLDER_STATUS


MATERIAL_FACTORS: dict[str, MaterialFactor] = {
    category: MaterialFactor() for category in SUPPORTED_CATEGORIES
}


def _unavailable(reason: str, *, unit: str | None = None, source: str | None = None) -> dict:
    result = {"value": None, "status": f"Estimate unavailable — {reason}"}
    if unit:
        result["unit"] = unit
    if source:
        result["source"] = source
    return result


def _validated_factor(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number or None.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number or None.")
    return float(value)


def _documented_source(source: str | None) -> bool:
    if not isinstance(source, str):
        return False
    return (
        bool(source.strip())
        and source != PLACEHOLDER_STATUS
        and not source.strip().casefold().startswith("placeholder")
    )


def estimate_material_impacts(
    category: str | None,
    weight_grams: float | None,
    *,
    category_is_certain: bool,
    factors: Mapping[str, MaterialFactor] = MATERIAL_FACTORS,
) -> dict:
    """Calculate sourced estimates only when category, mass, and factors exist.

    ``weight_grams`` must come from a measured input; this function never infers
    mass from an image. Passing a custom ``factors`` mapping is useful for tests
    and for a future reviewed configuration, but each value must include source
    information before it is used in the UI.
    """
    if not category_is_certain or category not in SUPPORTED_CATEGORIES:
        reason = "the category is uncertain or unsupported."
        return {
            "category": category,
            "weight_grams": weight_grams,
            "provisional": True,
            "carbon": _unavailable(reason, unit=CARBON_UNIT),
            "material_value": _unavailable(reason),
        }
    if weight_grams is None:
        reason = "enter a measured item weight in grams; the camera does not measure weight."
        return {
            "category": category,
            "weight_grams": None,
            "provisional": True,
            "carbon": _unavailable(reason, unit=CARBON_UNIT),
            "material_value": _unavailable(reason),
        }
    if isinstance(weight_grams, bool) or not isinstance(weight_grams, (int, float)):
        raise ValueError("Measured weight must be a finite number greater than zero grams.")
    if not math.isfinite(weight_grams) or weight_grams <= 0:
        raise ValueError("Measured weight must be a finite number greater than zero grams.")

    factor = factors.get(category)
    if factor is None:
        return {
            "category": category,
            "weight_grams": float(weight_grams),
            "provisional": True,
            "carbon": _unavailable("no material factor is configured.", unit=CARBON_UNIT),
            "material_value": _unavailable("no material factor is configured."),
        }

    weight_kg = float(weight_grams) / 1000.0
    carbon_factor = _validated_factor(
        factor.potential_co2e_kg_per_kg, "potential_co2e_kg_per_kg"
    )
    if carbon_factor is None or not _documented_source(factor.carbon_source):
        carbon = _unavailable(
            factor.carbon_source or "a documented source is missing.",
            unit=CARBON_UNIT,
            source=factor.carbon_source,
        )
    else:
        carbon = {
            "value": weight_kg * carbon_factor,
            "unit": CARBON_UNIT,
            "status": "Potential estimate — not measured savings.",
            "source": factor.carbon_source,
            "formula": f"{weight_kg:g} kg × {carbon_factor:g} kg CO2e/kg",
        }

    value_factor = _validated_factor(factor.material_value_per_kg, "material_value_per_kg")
    if value_factor is None or not factor.currency or not _documented_source(factor.value_source):
        material_value = _unavailable(
            factor.value_source,
            unit=factor.currency or "currency not configured",
            source=factor.value_source,
        )
    else:
        material_value = {
            "value": weight_kg * value_factor,
            "unit": factor.currency,
            "status": "Estimated material value — not a guaranteed market price.",
            "source": factor.value_source,
            "formula": f"{weight_kg:g} kg × {value_factor:g} {factor.currency}/kg",
        }

    return {
        "category": category,
        "weight_grams": float(weight_grams),
        "provisional": True,
        "carbon": carbon,
        "material_value": material_value,
    }
