# VISTA provisional MSW versus E-waste baseline

## Status

This is an **experimental, provisional baseline**, not a verified waste-family
classifier. Its labels were derived from source-folder names solely to enable a
time-limited experiment. They are not human-verified ground truth, and the
reported test scores measure agreement with those folder-derived labels only.
Do not use the model to make operational disposal decisions or describe the
scores as validated MSW/E-waste accuracy.

The existing human-review decisions, review queue, paper/cardboard pilot
manifest, and paper/cardboard checkpoint remain separate and unchanged.

## Provisional label policy

The explicit mapping used for this run was:

| Source folder | Provisional family | Limitation |
| --- | --- | --- |
| `paper`, `cardboard`, `plastic`, `glass`, `organic` | MSW | Folder-derived only; `organic` is not verified to contain only food waste. |
| `battery` | E_WASTE | Folder-derived only; only five battery examples had prior individual human review. The remaining images were not individually verified. |
| `metal`, `trash`, `textile` | Excluded | Their folder names do not establish the required family under VISTA policy. |

Human review decisions were not used as model labels. Existing hold, exclude,
and duplicate-conflict safeguards were respected. Exact SHA-256 groups were
kept within one split; cross-folder hash conflicts were excluded. The
preparation manifest records label provenance, source mapping, canonical path,
hash, split, and exclusion reason for auditing.

## Reproduction commands

Run from the project root in PowerShell. The preparation script refuses to
overwrite an existing manifest, so choose a new output path for a fresh run.
The command below records the settings used for this run:

```powershell
.\venv\Scripts\python.exe scripts\prepare_provisional_family.py `
  --inventory data\processed\family_dataset_inventory.csv `
  --dataset-root "C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset" `
  --review-queue data\processed\family_review_queue.csv `
  --review-decisions data\processed\family_review_decisions.csv `
  --batch-decisions data\processed\family_msw_review_batch_decisions.csv `
  --msw-folders paper cardboard plastic glass organic `
  --e-waste-folders battery `
  --max-per-folder 1000 `
  --seed 42 `
  --output data\processed\family_provisional_split_manifest.csv
```

The training command used cached ImageNet-pretrained torchvision ResNet18
weights and the existing Python environment. It writes only to the separate
provisional output directory and refuses to overwrite it:

```powershell
.\venv\Scripts\python.exe scripts\train_waste_family_provisional.py `
  --manifest data\processed\family_provisional_split_manifest.csv `
  --dataset-root "C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset" `
  --output-dir models\waste_family_provisional `
  --seed 42 `
  --epochs 8 `
  --batch-size 16
```

The current run used a frozen ResNet18 backbone, trained its final layer for
eight epochs, selected the checkpoint using validation macro-F1 and loss, and
opened the test split only for final evaluation. Inputs were RGB 224×224 images
normalized with ImageNet mean and standard deviation. Artifacts and exact
settings are in `models/waste_family_provisional/`.

## Dataset split and observed results

There are 6,000 canonical, readable hash groups in the prepared split:

| Split | MSW | E_WASTE | Total |
| --- | ---: | ---: | ---: |
| Train | 3,500 | 700 | 4,200 |
| Validation | 750 | 150 | 900 |
| Test | 750 | 150 | 900 |

The per-source-folder split counts are recorded in
`models/waste_family_provisional/training_metadata.json`. Preparation excluded
unmapped source folders, same-folder duplicate copies, cross-folder
same-hash conflicts, and samples beyond the configured per-folder cap. It
re-hashed and checked image readability without copying or changing source
images.

The held-out folder-label results in
`models/waste_family_provisional/final_test_metrics.json` are:

- Accuracy: **0.990**
- Macro F1: **0.9823**
- MSW: precision 0.9987, recall 0.9893, F1 0.9940 (750 test samples)
- E_WASTE: precision 0.9490, recall 0.9933, F1 0.9707 (150 test samples)
- Confusion matrix, rows/columns `MSW`, `E_WASTE`: `[[742, 8], [1, 149]]`

These high numbers do **not** establish real-world classification performance.
All evaluation targets came from folders. The model may learn dataset-specific
visual or collection-source cues, and the battery folder was not individually
reviewed in full. A representative human-labeled dataset and an independently
collected evaluation set are needed before this model can be called verified.

## Dashboard behavior and limits

The separate six-way folder-derived category experiment is documented in
[provisional_waste_category.md](provisional_waste_category.md). It does not
change the provisional status or evaluation of this binary family baseline.

Launch the existing Streamlit dashboard from the project root:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The detection page displays the verified family result separately from the
experimental result. The verified result remains **Uncertain** while no
verified family model is available. The provisional panel is visibly marked
**PROVISIONAL**, shows its raw score, and returns Uncertain below the
uncalibrated 0.70 demonstration threshold. A score above that threshold is
still a provisional folder-derived result.

This binary model does not identify waste subcategories. The separate
paper/cardboard pilot remains limited to those two classes. The dashboard does
not produce recycling decisions, contamination measurements, carbon savings,
or economic-recovery estimates from this model. Grad-CAM is also unavailable.
