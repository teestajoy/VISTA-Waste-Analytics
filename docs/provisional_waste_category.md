# Provisional six-category image classifier

## Status and label provenance

This is an **experimental, provisional whole-image classifier** trained from
MWCD source-folder labels. The folder names are not verified ground truth. The
test metrics therefore describe agreement with those folder-derived labels,
not verified category accuracy. Do not use this model as an operational
sorting or disposal authority.

The model supports only `paper`, `cardboard`, `plastic`, `glass`, `organic`,
and `battery`. The first five were mapped provisionally to MSW; `battery` was
mapped provisionally to E_WASTE. The `organic` folder was not fully confirmed
as food waste and the battery folder was not individually reviewed in full.
`metal`, `trash`, and `textile` were excluded. Human review CSV decisions were
not used as training labels.

Preparation reused the existing deterministic train/validation/test
assignments from `data/processed/family_provisional_split_manifest.csv`, which
had already grouped images by SHA-256. Exact-hash conflicts across source
folders and non-canonical duplicates were excluded. Candidate train and
validation source files were re-hashed and checked for image readability;
the test files were checked only after validation selected the checkpoint.
No source images or previous manifests/checkpoints were changed.

## Saved artifacts

`models/waste_category_provisional/` contains:

- `best_model.pth` — six-output ResNet18 checkpoint.
- `class_mapping.json` — class order and provisional category-to-family map.
- `split_manifest.csv` — category-specific included and excluded rows and split assignments.
- `training_history.csv` — per-epoch training and validation metrics.
- `training_metadata.json` — provenance, settings, versions, and split counts.
- `final_test_metrics.json` — final held-out results.

Class index order is `paper`, `cardboard`, `plastic`, `glass`, `organic`,
`battery`. The model used cached ImageNet-pretrained torchvision ResNet18,
froze its backbone, trained its six-class head for eight epochs, used seed 42
and batch size 16, and selected a checkpoint by validation macro-F1 followed
by validation loss. Image preprocessing was RGB, resized to 224×224, converted
to a tensor, and normalized with ImageNet mean and standard deviation.

## Split counts

| Category | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| paper | 715 | 139 | 146 | 1,000 |
| cardboard | 705 | 150 | 145 | 1,000 |
| plastic | 693 | 152 | 155 | 1,000 |
| glass | 689 | 152 | 159 | 1,000 |
| organic | 698 | 157 | 145 | 1,000 |
| battery | 700 | 150 | 150 | 1,000 |
| **Total** | **4,200** | **900** | **900** | **6,000** |

There are 6,000 unique included SHA-256 groups; no hash crosses splits. The
input family manifest had capped each category at 1,000 canonical samples.
The category-specific manifest retains upstream exclusion reasons for
non-canonical and conflicting rows from the six mapped source folders.

## Held-out folder-label results

Checkpoint selection ended at **epoch 6**. On the untouched 900-image test
split, accuracy was **0.8767** and macro-F1 was **0.8796**.

| Category | Precision | Recall | F1 | Test support |
| --- | ---: | ---: | ---: | ---: |
| paper | 0.9141 | 0.8014 | 0.8540 | 146 |
| cardboard | 0.8873 | 0.8690 | 0.8780 | 145 |
| plastic | 0.7016 | 0.8645 | 0.7746 | 155 |
| glass | 0.8690 | 0.7925 | 0.8289 | 159 |
| organic | 0.9720 | 0.9586 | 0.9653 | 145 |
| battery | 0.9735 | 0.9800 | 0.9767 | 150 |

These are **provisional folder-label metrics**. They do not establish
real-world category performance. Plastic and glass are particularly confused
in this test. Collection-source cues, label noise, and unreviewed examples can
inflate or distort the measured scores. Human-reviewed labels and an
independently collected test set are needed for a verified model.

## Dashboard behavior

Run from the project root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The dashboard keeps the verified family result fail-closed as **Uncertain**
while no verified family model exists. The folder-derived binary family model
and the six-category image model appear in separate panels as **PROVISIONAL**.
Both show raw scores and use an uncalibrated 0.70 demonstration threshold; a
below-threshold result is withheld as Uncertain with a manual-review message.
When both models produce confident outputs that imply different families, the
dashboard reports a conflict and asks for manual review. A matching family
mapping is not independent verification.

The category model predicts one label for the whole image. It does not locate
items or count category instances. YOLO box counts, when there are multiple
detections, are explicitly labeled as generic YOLO detections and are not
presented as waste-category counts. Practical handling text is general
guidance, not a recycling acceptance decision. No carbon, economic,
contamination, or recyclability estimates are produced.
