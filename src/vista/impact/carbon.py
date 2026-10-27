"""Carbon-economic lookup: CO2e avoided = f x m, value = p x m.

IMPORTANT: every factor, price and mass below is an illustrative PLACEHOLDER
that must be replaced with cited life-cycle and market data. Masses are
class-level assumptions, not measurements, so outputs are estimates.
"""

from dataclasses import dataclass

# category -> (CO2e avoided kg per kg, value USD per kg, assumed item mass kg)
REFERENCE_TABLE: dict[str, tuple[float, float, float]] = {
    "plastic": (1.8, 0.15, 0.03), "paper": (0.9, 0.08, 0.02),
    "cardboard": (0.8, 0.06, 0.15), "glass": (0.3, 0.02, 0.35),
    "organic": (0.15, 0.0, 0.10), "non-recyclable": (0.0, 0.0, 0.05),
    "metal": (4.0, 0.9, 0.10), "battery": (2.5, 1.2, 0.05),
    "PCB": (6.5, 8.0, 0.06), "mobile": (6.5, 4.5, 0.18),
    "laptop": (5.0, 2.5, 2.1), "cable": (2.0, 1.1, 0.08),
    "component": (3.0, 1.8, 0.02),
}


@dataclass(frozen=True)
class ImpactEstimate:
    category: str
    co2e_kg: float
    value_usd: float
    assumed_mass_kg: float
    placeholder_factors: bool = True


def estimate_impact(category: str) -> ImpactEstimate | None:
    row = REFERENCE_TABLE.get(category)
    if row is None:
        return None
    factor, price, mass = row
    return ImpactEstimate(category, round(factor * mass, 4), round(price * mass, 4), mass)
