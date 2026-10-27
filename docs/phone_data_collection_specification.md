# Mobile phone data collection and split specification

## Current repository evidence

The category split manifest contains no image paths with `phone`, `mobile`,
`smartphone`, or `iphone`. The local source image directory
`data/raw/processed/detection/classification` currently contains no image
files, so the referenced review images cannot be independently inspected from
this checkout. The saved YOLO crop diagnostics contain two generic `cell phone`
proposal hits: `battery/gd_10_battery_battery_547.jpg` and
`battery/gd_10_battery_battery_669.jpg`. Both records remain battery-folder,
battery-category examples; a generic detector proposal is not a human label or
evidence that either image depicts a mobile phone. Therefore there are **zero
verified, usable mobile-phone examples** in the current indexed dataset.

## Collection instructions

Collect new, real images of discarded mobile phones. Do not treat the battery
folder or a detector's `cell phone` prediction as a phone label. A qualified
reviewer must confirm both `family=E_WASTE` and `category=mobile` for every
included image. Keep batteries photographed alone labeled as battery; phones
with visible batteries still require the category policy to specify the
primary item before review begins.

For a useful first dataset, target at least 60 distinct physical phones, with
2–3 photographs per phone where practical. Include varied front/back views,
screen states, cases, damaged and intact devices, sizes, backgrounds, lighting,
camera distances, and ordinary clutter. Avoid legible personal information on
screens and identifiers such as IMEI/serial numbers. Retain original image
files and record the capture source, date/session, physical-device group ID,
reviewer decision, and review status. Hold ambiguous images rather than
forcing a label.

## Proposed group-aware allocation

Assign each physical phone and capture session to exactly one split before
training. Keep all views of the same device together. Also group exact SHA-256
duplicates and near-duplicates (perceptual hash or manual burst/session
review); if a group spans splits, move it to one split or exclude it. A
reasonable initial allocation of 60 distinct devices is 30 train, 10
validation, and 20 locked test devices. This supports a pilot, not a strong
precision estimate; collect more independent test devices for a stable
performance claim.

Existing test examples, including all three rows in
`docs/category_disagreement_re_review_v1.csv`, remain evaluation-only
and must not enter training or validation even if a new review confirms a
label. Do not use the test set to tune thresholds or choose checkpoints.

## Before training

1. Complete a versioned human review and preserve pending/held decisions.
2. Add mobile as an explicit category and family mapping in a new dataset
   preparation version; do not relabel battery examples as mobile.
3. Build a new manifest with source paths, label provenance, reviewer status,
   physical-device group IDs, exact hashes, near-duplicate groups, and split.
4. Validate that groups do not cross splits and existing test paths are absent
   from train/validation.
5. Train only after the reviewed training/validation data are ready; evaluate
   once on the locked test data and report sample counts and uncertainty.
