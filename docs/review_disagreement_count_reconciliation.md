# Review disagreement count reconciliation

## Finding

The repository does **not** contain a source report that identifies five
distinct category-level disagreements between human review and source-folder
labels. The prior audit repeated the requested count of five as a premise; it
did not cite five image identifiers. Its three-row result was based on the
combined evaluation output, but its source attribution to the
independent-blind decisions CSV was incomplete.

The most plausible reproducible source of a count of five is a **family-review
subset**, not five category disagreements: the mixed-type file
`data/processed/reviewed_test_category_review_decisions.csv` contains five
family review rows for battery-folder images. Four are `include / E_WASTE` and
one is `hold` with no label. Comparing the string `battery` directly with
`E_WASTE` mistakes the category-to-family mapping for a label conflict; the
hold is unresolved, not a contrary label. Two of those five image IDs also
have category decisions that disagree with their folder labels, but the five
family rows are not a set of five category disagreements.

## Report lineage and review-type mix

The category evaluation files show a growing, mixed cohort rather than
changing disagreement counts:

| Report | Category examples | Folder/category disagreements | What it contains |
| --- | ---: | ---: | --- |
| `data/processed/reviewed_test_model_evaluation_category_batch.json` | 2 | 0 | Earlier small report. |
| `data/processed/reviewed_test_model_evaluation_category_batch_v2.json` | 32 | 3 | The prior 30-row category review batch plus two reviewed paper/cardboard pilot cases. All three disagreements listed below are already present here. |
| `data/processed/reviewed_test_model_evaluation_gradcam_audit.json` | 52 | 3 | Later 52-example cohort. |
| `data/processed/reviewed_test_model_evaluation_independent_blind_batch.json` | 52 | 3 | Same 52 category records and same three disagreements as the Grad-CAM audit report; this is not a blind-only cohort. |

The later 52-record report contains the 32 paths from the category-batch-v2
report plus the 20 included category decisions in
`data/processed/reviewed_test_category_review_decisions_independent_blind.csv`.
The blind decision export contains 24 rows total: 20 labeled includes and
four held rows. Its 20 included labels match their source-folder categories;
none of the three disagreements below comes from that blind-only set. The
three reviewed labels instead appear in the regular mixed-type
`reviewed_test_category_review_decisions.csv` export. The blind-named report
therefore combines earlier regular-batch decisions with blind-batch
decisions. Its filename and `label_provenance` field do not identify the
review batch for each prediction.

The four CLIP comparison reports—
`reviewed_test_clip_category_comparison.json`, `_v2.json`, `_v3.json`, and
`_v4.json`—all compare 52 examples and do not reveal a five-image category
disagreement set. The comparison code takes its expected labels from the
model-evaluation report (`scripts/compare_clip_category.py`,
`DEFAULT_REVIEW_REPORT`). Thus those versions inherit the same mixed cohort;
the version changes do not account for a five-versus-three difference.

## The three documented category disagreements

All are canonical held-out `test` examples. Paths and SHA-256 values provide
stable identifiers. The model output is from
`reviewed_test_model_evaluation_independent_blind_batch.json`; raw top class
scores are uncalibrated, and the existing 0.70 cutoff remains in force.

| Image identifier | SHA-256 | Folder label → reviewed category | Recorded category output | Other review type on same image | Recommendation |
| --- | --- | --- | --- | --- | --- |
| `battery/gd_10_battery_battery_636.jpg` | `f69b4323b75bce94f94c00ac1b2b9f39a68a17d868d6e005a72ac5ac100baf73` | battery → plastic | battery, 0.9389 | Family decision is `hold`; no reviewed family label. | Keep the family decision held. Recheck the category as plastic vs. battery; do not infer a family label from this held record. |
| `battery/gd_10_battery_battery_738.jpg` | `91366f62949fda145db46272bf040834579b4575f2cbe92c3b4be22f8829e23a` | battery → plastic | battery, 0.9992 | Family decision is `include / E_WASTE`, which agrees with the folder's provisional family mapping, while the category decision plastic implies MSW. | Re-review category and family together. Resolve the plastic vs. battery category conflict before using either label for training. |
| `plastic/realwaste_Plastic_Plastic_220.jpg` | `c6d69c0546d8fccba10becd89e7183d257e6ca668b7fac56b1533f4adf04b3eb` | plastic → organic | plastic, 0.8296 | Family decision is `include / MSW`; both categories map to MSW. | Recheck plastic vs. organic at category level; family may remain MSW if confirmed. |

Evidence is in `reviewed_test_category_review_decisions.csv` (category and
family decisions), `reviewed_test_category_review_batch.csv` (test split and
hash), `models/waste_category_provisional/split_manifest.csv` (category split),
and the evaluation JSON above (reviewed expected label and model output).

## The five family-review rows that can be mistaken for the count

These five distinct battery images are all in the held-out `test` split in
`data/processed/family_provisional_split_manifest.csv` and are present in
`data/processed/reviewed_test_category_review_batch.csv`. The decisions and
hashes below are from `data/processed/reviewed_test_category_review_decisions.csv`.

| Image identifier | SHA-256 | Family decision | Reconciliation |
| --- | --- | --- | --- |
| `battery/gd_10_battery_battery_636.jpg` | `f69b4323b75bce94f94c00ac1b2b9f39a68a17d868d6e005a72ac5ac100baf73` | hold; unable to identify confidently | Unresolved family record, not a label disagreement. Also has the category disagreement above. Keep held and re-review if needed. |
| `battery/gd_10_battery_battery_547.jpg` | `1a61300b002772852e848a4d03e976c27699afd4c65eefeed6b7822528322a0e` | include / E_WASTE | Consistent with the provisional battery → E_WASTE mapping. No category disagreement is recorded for this image. |
| `battery/gd_10_battery_battery_738.jpg` | `91366f62949fda145db46272bf040834579b4575f2cbe92c3b4be22f8829e23a` | include / E_WASTE | Family decision matches the provisional mapping; it conflicts with the separate category decision plastic. Re-review both decisions together. |
| `battery/garbage_12_battery_battery2.jpg` | `7512687fa02f315f1057d58e95414f333f11d603ce4d32dd371f0564b2b92888` | include / E_WASTE | Consistent with the provisional mapping. No category disagreement is recorded for this image. |
| `battery/gd_10_battery_battery_685.jpg` | `c1df5b3733e5f446dbb3d06ec6878c5253fe0730695ef6d6c9e5325e40aaa0df` | include / E_WASTE | Consistent with the provisional mapping. No category disagreement is recorded for this image. |

The five-row family subset overlaps the three category-disagreement IDs at
`battery/gd_10_battery_battery_636.jpg` and
`battery/gd_10_battery_battery_738.jpg`. It contributes no two additional
category disagreements: the other three family decisions are consistent with
the folder-to-family mapping, and one of the five is held without a label.

## Disposition of the count difference

- **Different report versions:** The reports progress from 2 to 32 to 52
  category examples. The three category mismatches are stable from the
  32-example report onward; no inspected report establishes five.
- **Duplicates:** The 52-example evaluation contains unique source paths. The
  five family rows are also five distinct paths; duplicate counting does not
  explain five category mismatches.
- **Different review types:** Yes. The mixed CSV contains `family` and
  `category` rows for the same image. Family `E_WASTE` must be compared with
  the folder's mapped family, not the literal folder name `battery`. One
  family row is held and has no label.
- **Excluded, held, unresolved:** The held image
  `battery/gd_10_battery_battery_636.jpg` is not a reviewed family label. The
  independent-blind decisions export has four other held rows, excluded from
  the labeled 20-case category subset. None supplies an extra folder/category
  mismatch.
- **Missing or outdated exports:** The 52-record report is not blind-only and
  does not carry per-row batch provenance. The blind decision CSV alone does
  not contain the three disagreements. The earlier audit's statement that
  this CSV substantiated the three was inaccurate. Preserve the exports and
  add explicit review-batch provenance to any future reconciliation output;
  do not rewrite historical reports.
- **Remaining two:** No two additional, stable category-mismatch IDs can be
  supported. If “five” refers to the five family rows, the extra count comes
  from mixing family and category review types and treating a hold or a raw
  label-string mismatch as a category disagreement. As a claim of five
  category disagreements, it is unsupported by the saved reports.

## Recommended resolution

Keep all saved decisions unchanged. Re-review the three category conflicts
individually: the two battery-folder images require category decisions, and
the `636` family record must remain held until that task is independently
resolved; the `738` family/category conflict needs a coordinated adjudication;
the plastic-folder item needs a category-only plastic-versus-organic check.
For the three other battery family rows, retain the existing E_WASTE decisions
as family-level decisions unless new evidence warrants review. Any new review
should be saved as a new, batch-identified export with task type, reviewer
blinding status, and stable image hash so future reports can distinguish the
cohorts.
