# Independent review vs. source-folder label audit

## Scope and source files

This audit compares explicit independently reviewed category labels with each
image's original source-folder label. It uses the category rows in
`data/processed/reviewed_test_model_evaluation_independent_blind_batch.json`,
the category split manifest at
`models/waste_category_provisional/split_manifest.csv`, and the recorded
independent-blind review decisions in
`data/processed/reviewed_test_category_review_decisions_independent_blind.csv`.
No labels, review decisions, manifests, model outputs, or checkpoints were
changed.

The request described five disagreements. The current independent-blind
evaluation and decision files substantiate **three distinct images** whose
reviewed category differs from the source-folder category. They contain 52
reviewed category examples. I did not add two unsubstantiated rows to reach
five. This difference should be reconciled against any newer review export
before acting on a five-image count.

## Disagreements found

All three cases are in the held-out `test` split. The recorded category
prediction below is the provisional classifier's decision at the existing
0.70 cutoff; the raw top candidate and score are shown where available.

| Image identifier (source-relative path) | Split | Original folder label | Human-reviewed category | Recorded category prediction | Family-level disagreement |
| --- | --- | --- | --- | --- | --- |
| `battery/gd_10_battery_battery_636.jpg` | test | battery | plastic | battery (candidate, 0.9389) | Yes: folder implies E_WASTE; reviewed plastic maps to MSW. No matching per-image family prediction is recorded in this category evaluation. |
| `battery/gd_10_battery_battery_738.jpg` | test | battery | plastic | battery (candidate, 0.9992) | Yes: folder implies E_WASTE; reviewed plastic maps to MSW. No matching per-image family prediction is recorded in this category evaluation. |
| `plastic/realwaste_Plastic_Plastic_220.jpg` | test | plastic | organic | plastic (candidate, 0.8296) | No: both plastic and organic map to MSW under the provisional family mapping. |

The model's raw top candidate is reported even when the operational prediction
is `Uncertain`; it must not be read as a replacement for the cutoff behavior.
The reviewed category is an audit reference only and has not been promoted to
a training label.

## Recommended next action

Reconcile the asserted five-image list with the independent-blind review
export, then have a reviewer recheck the two battery-folder images marked
plastic, since those decisions would change family as well as category. Keep
the current decisions intact pending that check. Once the disagreement set is
confirmed, use these examples to guide a targeted, separately recorded review
batch and evaluate it without changing the 0.70 cutoff. Do not retrain from
these three rows alone: the reviewed sample is small, and these disagreements
do not explain all reported field errors such as notebooks or batteries routed
to MSW.
