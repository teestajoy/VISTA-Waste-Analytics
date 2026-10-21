# MWCD to VISTA Taxonomy Mapping Assessment

This document records a conservative assessment of the MWCD category folders
against VISTA's planned classification taxonomy. Counts use the image
extensions and recursive counting behavior in `scripts/inspect_dataset.py`.

## MWCD category counts and mappings

“Direct” describes a matching category name; it does not confirm that every
image follows VISTA's intended label definition. “Possible mapping” means a
documented labeling rule is needed before using that mapping.

| MWCD category | Images | VISTA target label | Mapping status and rationale | First-dataset disposition |
|---|---:|---|---|---|
| `battery` | 1,727 | `battery` | Direct name match, but the dataset's label meaning is undocumented and unverified. | Hold for review |
| `cardboard` | 3,166 | `cardboard` | Direct category-name match. Folder names alone do not verify image labels. | Candidate |
| `glass` | 4,678 | `glass` | Direct category-name match, but two exact-duplicate groups are shared with `plastic`; the material in those groups is not conclusively verified. | Hold for review |
| `metal` | 2,911 | `metal` | Possible mapping only if a documented rule establishes that this folder contains e-waste metal. Generic metal may include non-electronic items. | Hold for review |
| `organic` | 2,547 | `organic` | Direct name match, but the dataset does not document what qualifies as organic. Unusual samples alone do not establish mislabeling. | Hold for review |
| `paper` | 3,496 | `paper` | Direct category-name match. Folder names alone do not verify image labels. | Candidate |
| `plastic` | 3,897 | `plastic` | Direct category-name match, but two exact-duplicate groups are shared with `glass`; the material in those groups is not conclusively verified. | Hold for review |
| `textile` | 10,964 | No supported target | VISTA has no textile label. Do not silently place textiles in another category. | Unsupported |
| `trash` | 1,782 | `non-recyclable` | Possible mapping only if a documented rule defines `trash` as non-recyclable under VISTA's intended meaning. | Hold for review |
| **Total** | **35,168** | | | |

`paper` and `cardboard` are candidate categories only. Each is subject to
label-quality checks and duplicate-aware splitting before use. Exact duplicates
should be handled before splitting training and evaluation data to avoid
leakage.

## VISTA targets with missing or unverified MWCD support

- `battery` — MWCD has a same-name folder, but its label meaning is unverified.
- `metal` — MWCD `metal` is only a possible source if a documented rule limits
  it to e-waste metal.
- `non-recyclable` — MWCD `trash` is only a possible source if a documented
  rule establishes that equivalence.
- `PCB` — no MWCD source folder.
- `mobile` — no MWCD source folder.
- `laptop` — no MWCD source folder.
- `cable` — no MWCD source folder.
- `component` — no MWCD source folder.

The remaining VISTA targets have direct name matches in MWCD: `cardboard`,
`glass`, `organic`, `paper`, and `plastic`. Direct name matches do not prove
that the dataset labels are correct.

## Unresolved cross-category duplicates

The exact-duplicate audit found two groups shared between MWCD `plastic` and
`glass`. The review observed that the images appear glass-like, but material
was not conclusively verified. Both groups remain unresolved:

- Group 1545 — SHA-256
  `c41b99aec8a3257c8668a2144ebbaf8042018f0a85d88e22bf4e413b43693dd8`
- Group 1828 — SHA-256
  `e971f4f8f50e960e454ea724cec922fc2988bbf4027aa48b553e139bb5890968`

See [the cross-category duplicate review manifest](mwcd_cross_category_duplicate_review.md)
for the file paths and review notes.

## Scope

This document is an assessment only. No dataset cleaning or model training has
been performed. The original dataset remains unchanged.
