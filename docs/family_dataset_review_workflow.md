# MSW versus E-waste family review workflow

This workflow creates a new family dataset independently of the paper/cardboard
pilot. It does not change the source dataset, existing pilot review CSVs,
manifests, split assignments, or checkpoints. Candidate suggestions are kept
separate from human decisions. A guarded family-training command is available,
but stops until a prepared manifest has enough approved examples from both
families.

## Review routes are not labels

The inventory assigns only a route to help organize human review:

| Source folder | Review route | Human-review rule |
|---|---|---|
| `paper`, `cardboard`, `plastic`, `glass` | `candidate_msw` | Inspect the image and explicitly choose MSW or E-waste. The folder name is not approval. |
| `organic` | `candidate_msw_food_waste_only` | Include as MSW only after confirming the visible item is food waste; record that evidence in the reason. |
| `battery` | `candidate_e_waste_verify_content` | Include as E-waste only after confirming the visible content; record the evidence in the reason. |
| `metal`, `trash` | `hold_manual_review` | Inspect the image. Do not infer family from the folder name. |
| `textile` | `unsupported` | Cannot be included until VISTA defines its intended treatment. |

The family labels accepted for explicit decisions are `MSW` and `E_WASTE`.
Review statuses are `pending`, `include`, `hold`, and `exclude`. An `include`
requires a human-selected family. `hold` and `exclude` require a reason. Pending
images remain out of all splits. Exact duplicates are grouped by file-byte
SHA-256; cross-folder groups enter a split only after every path is explicitly
included with the same family. Conflicting labels are held. For an approved
same-family group, only the lexically first source-relative path is the
canonical training row; its exact duplicate copies receive no split.

## Create a separate metadata-only candidate audit

After the existing inventory, queue, and decisions CSVs exist, run:

```powershell
.\venv\Scripts\python.exe scripts\generate_family_candidates.py --inventory data\processed\family_dataset_inventory.csv --queue data\processed\family_review_queue.csv --decisions data\processed\family_review_decisions.csv --output data\processed\family_candidate_audit.csv
```

This reads existing CSV metadata only; it does not rescan or open source
images. It writes a new report and refuses to overwrite an existing file. It
shows folder-policy suggestions separately from saved human decisions, lists
exact-SHA duplicate groups, marks cross-folder duplicates for review, and
summarizes counts by folder, suggestion, review status, and reason. Filenames
and YOLO labels are not used to infer image content. Suggestions never approve
rows or assign splits. Choose a new output name for a later report.

The existing inventory records 35,168 files. The saved decision CSV records
five `E_WASTE` includes and no `MSW` includes; preparation must still revalidate
their paths, hashes, and readability. The review queue has 332 entries.

## Check approved family data before preparation

Run the read-only readiness report:

```powershell
.\venv\Scripts\python.exe scripts\family_dataset_workflow.py readiness
```

It counts eligible, human-approved SHA-256 groups once each using the same
canonical-path, duplicate-conflict, hold, exclude, and label-consistency rules
as manifest preparation. It reads the saved inventory hash values; the later
`prepare` command rechecks current bytes and image readability before writing.
At least three groups in each family are required merely to populate train,
validation, and test. That is a mechanical minimum, not a defensible estimate
of model performance; more reviewed examples and visual diversity are needed.

## 1. Create a reproducible inventory

Run from the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe scripts\family_dataset_workflow.py inventory --dataset-root "C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset" --output data\processed\family_dataset_inventory.csv
```

The script recursively reads supported image files in all nine known folders,
records SHA-256 and image-readability errors, and marks rows `pending`. It writes
only the new inventory CSV and refuses to overwrite an existing output. This
step may take a while because it reads every supported file. It does not modify
or copy source images.

## 2. Create a visual review queue

```powershell
.\venv\Scripts\python.exe scripts\family_dataset_workflow.py queue --dataset-root "C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset" --inventory data\processed\family_dataset_inventory.csv --per-folder 40 --seed 42 --queue data\processed\family_review_queue.csv --decisions data\processed\family_review_decisions.csv --sheets-dir test-data\family_review
```

This selects up to 40 deterministic, readable, canonical images per supported
source folder. Exact duplicates within one folder are sampled once. All
cross-folder exact-hash groups are added to the queue even if that makes a
folder exceed the sample limit, so the conflict can be reviewed. The script
creates a multi-page PDF per queued folder in `test-data\family_review\`, plus
a queue CSV and a separate decisions CSV. Each PDF labels the image with its
source folder, proposed review route, and full source-relative path. The
decisions CSV is initialized with `pending` and blank family/reason fields; no
image is approved automatically. Existing outputs are never overwritten, so
choose new output names if you intentionally make another sample.

## Create the compact paper/cardboard/plastic review batch

To reduce manual navigation, select up to 20 pending candidates per source
folder from the existing queue:

```powershell
.\venv\Scripts\python.exe scripts\create_msw_review_batch.py --per-folder 20 --seed 42
```

This creates separate queue and decision CSVs plus one contact-sheet PDF per
selected folder. It accepts only canonical, readable, non-conflicting images
already in the original queue, validates the candidate audit against the
inventory, and rechecks the selected source hashes and image readability.
Every new decision starts `pending`; these source folders remain candidate
routes, not labels. Existing queue and decision files are left untouched.

Alternatively, to edit the CSV manually, open the PDFs and inspect each image. In
`data\processed\family_review_decisions.csv`, keep `pending` where you have not
made a decision. For a reviewed image, set `review_status` to `include`, `hold`,
or `exclude`. For `include`, enter `MSW` or `E_WASTE` in `reviewed_family` based
on the image content, not the folder. For `hold` or `exclude`, provide a clear
reason. For an organic image, include only confirmed food waste and note the
visible evidence. For a battery image, confirm its content and note the
evidence. Leave mixed or unclear material held.

## 3. Review images in the local interface

From the repository root in PowerShell, run:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\family_review.py
```

To review only the compact batch, set temporary PowerShell environment
variables before launching:

```powershell
$env:VISTA_FAMILY_REVIEW_QUEUE = "data\processed\family_msw_review_batch_queue.csv"
$env:VISTA_FAMILY_REVIEW_DECISIONS = "data\processed\family_msw_review_batch_decisions.csv"
.\venv\Scripts\python.exe -m streamlit run dashboard\family_review.py
```

The page displays the active queue and decisions paths. After review, clear the
temporary settings with `Remove-Item Env:VISTA_FAMILY_REVIEW_QUEUE` and
`Remove-Item Env:VISTA_FAMILY_REVIEW_DECISIONS` to return to the full queue.

The interface reads only the queue selected by `VISTA_FAMILY_REVIEW_QUEUE`.
The current 60-image MSW batch uses:

- Queue: `data\processed\family_msw_review_batch_queue.csv`
- Decisions: `data\processed\family_msw_review_batch_decisions.csv`
- Contact sheets: `test-data\family_msw_review_batch\paper.pdf`,
  `test-data\family_msw_review_batch\cardboard.pdf`, and
  `test-data\family_msw_review_batch\plastic.pdf`

Use **Single image** for one-at-a-time review or **Batch review** to display 20,
30, or 40 images per page. Choose **All 60 (compact review)** to show the whole
queue together in a six-column grid. Each tile shows the image, full source-relative
path, source folder, image identifier, current decision, and a separate
suggested-family note. Source-folder suggestions are display-only: they are
never written as reviewed labels. Organic is only an MSW candidate after
confirming visible food waste; battery is only an E_WASTE candidate after
confirming its content. Metal and trash have no family suggestion and are
marked for inspection; textile is unsupported. Cross-folder or conflicting
duplicate groups are marked for manual review and blocked from batch family
approval.

In either mode choose an explicit `MSW`, `E_WASTE`, `HOLD`, `EXCLUDE`, or
`PENDING` decision. `HOLD` and `EXCLUDE` require a reason. Organic and battery
approvals require explicit content confirmation and an evidence reason. Batch
mode saves only selected images and first presents the number and source
folders affected. Its MSW confirmation preview also reports how many distinct,
currently readable, hash-verified, non-conflicting MSW groups would be approved
across the full inventory. Inspect the summary and click **Confirm and save batch**
once to write. If an
existing non-pending decision would change, confirm its replacement. After
choosing a family, **Select matching pending ... candidates on this page** can
select readable, hashed, non-conflicting canonical rows whose display-only
candidate matches that choice. This only selects rows; it does not approve or
save them. Inspect every image in the grid and leave ambiguous examples
unselected or mark them HOLD. The normal explicit batch confirmation is still
required. The app preserves all other decisions and writes only
configured family-review decisions CSV using the exact schema
`source_relative_path,review_status,reviewed_family,reason`. The decision file
is not reset when the interface opens. Changed decisions are appended to the
separate `data\processed\family_review_audit.csv` log. Missing, unreadable, changed, unsafe, or
invalid paths are reported; unreadable images cannot be included. Filters are
available for source folder, pending/reviewed status, and suggested family.
**Export review summary CSV** downloads counts by folder and decision.

`PENDING` remains outside all future splits. Leave ambiguous or mixed-material
images pending or mark them HOLD with a reason. The app only records human
decisions: it does not promote suggestions, prepare a manifest, or train a model.

## 4. Validate decisions and prepare a separate family manifest

After explicit review, merge compact-batch decisions into a new, separate CSV.
This requires `--confirm`, preserves the original decisions, and validates the
selected paths and included source hashes:

```powershell
.\venv\Scripts\python.exe scripts\family_dataset_workflow.py merge-review-batch --batch-queue data\processed\family_msw_review_batch_queue.csv --batch-decisions data\processed\family_msw_review_batch_decisions.csv --output data\processed\family_review_decisions_msw_batch.csv --confirm
```

Check the merged review file with:

```powershell
.\venv\Scripts\python.exe scripts\family_dataset_workflow.py readiness --review-csv data\processed\family_review_decisions_msw_batch.csv
```

At least three unique approved hash groups per family are required only to
populate each split; that minimum does not establish reliable evaluation.

```powershell
.\venv\Scripts\python.exe scripts\family_dataset_workflow.py prepare --dataset-root "C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset" --inventory data\processed\family_dataset_inventory.csv --queue data\processed\family_review_queue.csv --review-csv data\processed\family_review_decisions.csv --output data\processed\family_split_manifest.csv --seed 42
```

Before writing, this step validates decision values, queue membership, safe
relative paths (including symlinks), current file hashes, image readability,
duplicate-group consistency, and split leakage. The manifest records proposed
route separately from the reviewed family, per-image review status, duplicate
group, inventory errors, group disposition, split, the original human
`review_reason`, and a separate group-level inclusion/exclusion reason. If either family has fewer than
three unique approved hash groups, it leaves every split blank and reports the
counts. Three groups per family is only the minimum for a mechanical
train/validation/test partition; it is not enough evidence for a useful or
reliable classifier. With enough groups, it uses a seeded, per-family 70/15/15
split. Only explicitly included canonical rows can receive splits.

Generated files include `data\processed\family_dataset_inventory.csv`,
`data\processed\family_review_queue.csv`,
`data\processed\family_review_decisions.csv`,
`data\processed\family_review_audit.csv`,
`data\processed\family_candidate_audit.csv`,
`data\processed\family_split_manifest.csv`, and contact sheets under
`test-data\family_review\`. `data/` and `test-data/` are Git-ignored here.

## Train after both families have enough approved examples

The trainer uses existing duplicate-aware split assignments; it never
re-splits. It requires canonical human-included rows, both families in train
and validation, both families in the held-out test set, current source hashes,
readable images, and no duplicate hash across splits. Training uses inverse-frequency
class weights so the larger reviewed family does not dominate the loss.
ImageNet ResNet18 weights are used only if already cached locally. Validation selects the checkpoint.
Test labels and files are inspected only after model selection. Reports include
per-class precision, recall, F1 and support, accuracy, macro-F1, and a confusion
matrix.

When the separate family manifest is ready, run:

```powershell
.\venv\Scripts\python.exe scripts\train_waste_family.py --manifest data\processed\family_split_manifest.csv --dataset-root "C:\Users\teest\Downloads\Waste_classification_dataset\final_merged_dataset" --seed 42 --epochs 20 --batch-size 8
```

Outputs go only to the new `models\waste_family_resnet18_pilot\` directory;
existing output directories are refused. Training uses 224×224 RGB resizing,
`ToTensor`, and ImageNet mean/std normalization. Inference verifies the same
preprocessing metadata, class order, and both-class final test report. The
runtime 0.70 score threshold is provisional and not calibrated; lower scores
remain Uncertain. A small pilot cannot establish production performance.

The dashboard loads a family model only when complete local artifacts pass
those checks. Until then it remains Uncertain. Generic YOLO detections and the
paper/cardboard pilot remain separate outputs.
