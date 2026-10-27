from pathlib import Path

import pytest


@pytest.fixture
def material_factors(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    from vista.impact import material_factors

    return material_factors


def test_weight_formulas_are_transparent_and_use_grams_to_kilograms(material_factors):
    factors = {
        "paper": material_factors.MaterialFactor(
            potential_co2e_kg_per_kg=1.2,
            carbon_source="Test source, scenario A, 2026-01.",
            material_value_per_kg=0.8,
            currency="USD",
            value_source="Test local price schedule, 2026-01.",
        )
    }

    estimates = material_factors.estimate_material_impacts(
        "paper", 250, category_is_certain=True, factors=factors
    )

    assert estimates["carbon"]["value"] == pytest.approx(0.3)
    assert estimates["carbon"]["unit"] == "kg CO2e potential avoided"
    assert estimates["carbon"]["formula"] == "0.25 kg × 1.2 kg CO2e/kg"
    assert "not measured savings" in estimates["carbon"]["status"]
    assert estimates["material_value"]["value"] == pytest.approx(0.2)
    assert estimates["material_value"]["unit"] == "USD"
    assert "not a guaranteed market price" in estimates["material_value"]["status"]


def test_shipped_supported_categories_have_explicit_factor_placeholders(material_factors):
    assert set(material_factors.MATERIAL_FACTORS) == {
        "paper", "cardboard", "plastic", "glass", "organic", "battery"
    }
    assert all(factor.potential_co2e_kg_per_kg is None for factor in material_factors.MATERIAL_FACTORS.values())
    assert all(factor.material_value_per_kg is None for factor in material_factors.MATERIAL_FACTORS.values())
    assert all(factor.carbon_source == material_factors.PLACEHOLDER_STATUS for factor in material_factors.MATERIAL_FACTORS.values())
    assert all(factor.value_source == material_factors.PLACEHOLDER_STATUS for factor in material_factors.MATERIAL_FACTORS.values())

    estimates = material_factors.estimate_material_impacts("plastic", 500, category_is_certain=True)
    assert estimates["carbon"]["value"] is None
    assert estimates["material_value"]["value"] is None
    assert "Estimate unavailable" in estimates["carbon"]["status"]
    assert "Estimate unavailable" in estimates["material_value"]["status"]
    assert "Placeholder" in estimates["carbon"]["source"]


def test_numeric_factors_without_documented_sources_are_not_used(material_factors):
    factors = {
        "paper": material_factors.MaterialFactor(
            potential_co2e_kg_per_kg=1.2,
            material_value_per_kg=0.8,
            currency="USD",
        )
    }
    estimates = material_factors.estimate_material_impacts(
        "paper", 250, category_is_certain=True, factors=factors
    )
    assert estimates["carbon"]["value"] is None
    assert estimates["material_value"]["value"] is None


@pytest.mark.parametrize(
    ("category", "weight", "certain", "expected_reason"),
    [
        ("paper", None, True, "measured item weight"),
        ("paper", 100, False, "category is uncertain"),
        ("textile", 100, True, "category is uncertain or unsupported"),
    ],
)
def test_missing_inputs_uncertainty_and_unsupported_categories_are_unavailable(
    material_factors, category, weight, certain, expected_reason
):
    estimates = material_factors.estimate_material_impacts(
        category, weight, category_is_certain=certain
    )
    for metric in ("carbon", "material_value"):
        assert estimates[metric]["value"] is None
        assert "Estimate unavailable" in estimates[metric]["status"]
        assert expected_reason in estimates[metric]["status"]


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True])
def test_invalid_measured_weights_are_rejected(material_factors, weight):
    with pytest.raises(ValueError, match="greater than zero grams"):
        material_factors.estimate_material_impacts("paper", weight, category_is_certain=True)


@pytest.mark.parametrize("factor", [-0.1, float("nan"), float("inf"), True])
def test_invalid_configured_factors_are_rejected(material_factors, factor):
    factors = {"paper": material_factors.MaterialFactor(potential_co2e_kg_per_kg=factor)}
    with pytest.raises(ValueError, match="finite non-negative number"):
        material_factors.estimate_material_impacts("paper", 100, category_is_certain=True, factors=factors)
