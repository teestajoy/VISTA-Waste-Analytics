"""Target waste labels and their MSW/e-waste family mapping.

These categories define the labels VISTA intends to learn; they are not
predictions from a trained model. Dataset-labeling rules must resolve ambiguous
cases, especially metal items and electronic components.
"""

from enum import Enum
from types import MappingProxyType
from typing import Mapping


class WasteFamily(str, Enum):
    """Top-level waste family for a target category."""

    MSW = "MSW"
    E_WASTE = "E_WASTE"


class WasteCategory(str, Enum):
    """The planned fine-grained classification labels."""

    PLASTIC = "plastic"
    PAPER = "paper"
    CARDBOARD = "cardboard"
    GLASS = "glass"
    ORGANIC = "organic"
    NON_RECYCLABLE = "non-recyclable"
    METAL = "metal"
    BATTERY = "battery"
    PCB = "PCB"
    MOBILE = "mobile"
    LAPTOP = "laptop"
    CABLE = "cable"
    COMPONENT = "component"


MSW_CATEGORIES: tuple[WasteCategory, ...] = (
    WasteCategory.PLASTIC,
    WasteCategory.PAPER,
    WasteCategory.CARDBOARD,
    WasteCategory.GLASS,
    WasteCategory.ORGANIC,
    WasteCategory.NON_RECYCLABLE,
)

E_WASTE_CATEGORIES: tuple[WasteCategory, ...] = (
    WasteCategory.METAL,
    WasteCategory.BATTERY,
    WasteCategory.PCB,
    WasteCategory.MOBILE,
    WasteCategory.LAPTOP,
    WasteCategory.CABLE,
    WasteCategory.COMPONENT,
)

CATEGORY_FAMILY: Mapping[WasteCategory, WasteFamily] = MappingProxyType(
    {
        **{category: WasteFamily.MSW for category in MSW_CATEGORIES},
        **{category: WasteFamily.E_WASTE for category in E_WASTE_CATEGORIES},
    }
)
