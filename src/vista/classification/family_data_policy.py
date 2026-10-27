"""Conservative MWCD source-folder policy for future family-label review.

Candidate family values indicate only a possible review route. They never
approve an image: every included example requires an explicit human decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from vista.classification.taxonomy import WasteFamily


class SourceDisposition(str, Enum):
    CANDIDATE_REQUIRES_REVIEW = "candidate_requires_review"
    HOLD_FOR_POLICY = "hold_for_policy"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class FamilySourcePolicy:
    candidate_family: WasteFamily | None
    disposition: SourceDisposition
    rationale: str


MWCD_FAMILY_SOURCE_POLICY: Mapping[str, FamilySourcePolicy] = MappingProxyType(
    {
        "paper": FamilySourcePolicy(
            WasteFamily.MSW,
            SourceDisposition.CANDIDATE_REQUIRES_REVIEW,
            "Taxonomy maps paper to MSW; verify each image before inclusion.",
        ),
        "cardboard": FamilySourcePolicy(
            WasteFamily.MSW,
            SourceDisposition.CANDIDATE_REQUIRES_REVIEW,
            "Taxonomy maps cardboard to MSW; verify each image before inclusion.",
        ),
        "plastic": FamilySourcePolicy(
            WasteFamily.MSW,
            SourceDisposition.CANDIDATE_REQUIRES_REVIEW,
            "Taxonomy maps plastic to MSW; verify labels and duplicate groups.",
        ),
        "glass": FamilySourcePolicy(
            WasteFamily.MSW,
            SourceDisposition.CANDIDATE_REQUIRES_REVIEW,
            "Taxonomy maps glass to MSW; verify labels and duplicate groups.",
        ),
        "organic": FamilySourcePolicy(
            WasteFamily.MSW,
            SourceDisposition.CANDIDATE_REQUIRES_REVIEW,
            "Review under VISTA's food-waste-only organic rule.",
        ),
        "battery": FamilySourcePolicy(
            WasteFamily.E_WASTE,
            SourceDisposition.CANDIDATE_REQUIRES_REVIEW,
            "Same-name taxonomy target; define qualifying battery scope and review each image.",
        ),
        "metal": FamilySourcePolicy(
            None,
            SourceDisposition.HOLD_FOR_POLICY,
            "Generic metal is not evidence of E-waste; review for electronic context.",
        ),
        "trash": FamilySourcePolicy(
            None,
            SourceDisposition.HOLD_FOR_POLICY,
            "No approved rule maps the dataset's trash folder to a VISTA family.",
        ),
        "textile": FamilySourcePolicy(
            None,
            SourceDisposition.UNSUPPORTED,
            "Textile has no supported VISTA target label and is excluded for now.",
        ),
    }
)
