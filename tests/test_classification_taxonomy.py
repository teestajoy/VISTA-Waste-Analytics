from pathlib import Path


def test_taxonomy_has_exactly_13_unique_category_labels(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from vista.classification.taxonomy import WasteCategory

    labels = [category.value for category in WasteCategory]

    assert len(labels) == 13
    assert len(set(labels)) == 13


def test_taxonomy_contains_the_planned_msw_and_e_waste_categories(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from vista.classification.taxonomy import E_WASTE_CATEGORIES, MSW_CATEGORIES

    assert {category.value for category in MSW_CATEGORIES} == {
        "plastic",
        "paper",
        "cardboard",
        "glass",
        "organic",
        "non-recyclable",
    }
    assert {category.value for category in E_WASTE_CATEGORIES} == {
        "metal",
        "battery",
        "PCB",
        "mobile",
        "laptop",
        "cable",
        "component",
    }
    assert len(MSW_CATEGORIES) == 6
    assert len(E_WASTE_CATEGORIES) == 7


def test_family_mapping_is_complete_and_uses_only_valid_families(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from vista.classification.taxonomy import (
        CATEGORY_FAMILY,
        WasteCategory,
        WasteFamily,
    )

    assert set(CATEGORY_FAMILY) == set(WasteCategory)
    assert all(isinstance(family, WasteFamily) for family in CATEGORY_FAMILY.values())


def test_no_category_is_assigned_to_both_families(monkeypatch):
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.syspath_prepend(src_path)

    from vista.classification.taxonomy import (
        CATEGORY_FAMILY,
        E_WASTE_CATEGORIES,
        MSW_CATEGORIES,
        WasteFamily,
    )

    assert set(MSW_CATEGORIES).isdisjoint(E_WASTE_CATEGORIES)
    assert all(CATEGORY_FAMILY[item] is WasteFamily.MSW for item in MSW_CATEGORIES)
    assert all(
        CATEGORY_FAMILY[item] is WasteFamily.E_WASTE for item in E_WASTE_CATEGORIES
    )
