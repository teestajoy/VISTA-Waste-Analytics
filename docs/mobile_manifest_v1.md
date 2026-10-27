# Mobile phone review and split manifest v1

## Current status

`data/processed/mobile_phone_review_batch_v1.csv` is a header-only intake
template. No phone rows or labels have been invented, and no verified mobile
examples are currently available. The script does not infer a phone label from
a source folder, model output, or detector proposal.

## Intake columns

| Column | Meaning |
| --- | --- |
| `source_relative_path` | Image path relative to the supplied dataset root. Must be safe and point to a readable image. |
| `sha256` | Exact SHA-256 of the source image bytes; the script recomputes it. |
| `proposed_category`, `proposed_family` | Intake target, required to be `mobile` and `E_WASTE`; these are not accepted labels by themselves. |
| `reviewed_category`, `reviewed_family` | Human-confirmed labels. Included records require `mobile` and `E_WASTE`. |
| `category_decision`, `family_decision` | Separate human decisions: `include`, `hold`, or `exclude`. Included records require both to be `include`. |
| `review_status` | `pending`, `complete`, `held`, or `excluded`. Only `complete` rows with both include decisions can receive a split. |
| `reviewer_id`, `reviewed_utc` | Reviewer identity and review timestamp for any non-pending decision. |
| `physical_device_group_id` | Stable ID for the physical phone. Required for included records; all photos of one device stay together. |
| `capture_session_id` | Capture-session identifier where available. |
| `image_provenance` | How/where the image was obtained. Required for included records. |
| `duplicate_group_id` | Optional exact-duplicate group. If supplied, it must be `sha256:<hash>`. Duplicate image hashes are rejected. |
| `near_duplicate_group_id` | Optional group for burst, resized, or otherwise near-duplicate images; groups cannot cross splits. |
| `split` | Leave blank in intake. The script assigns it only after review and sufficient eligible data. |
| `exclusion_reason` | Leave blank in intake. The output records why a pending, held, excluded, or unselected row has no split. |

Pending rows may be recorded for review tracking, but must not contain partial
review decisions and never receive a split. Held and excluded rows also remain
unsplit. A row marked complete without both explicit include decisions and
both reviewed labels is rejected.

## Validation and protected data

Run `scripts/prepare_mobile_manifest.py` with the phone image dataset root.
The script checks safe relative paths, source readability, exact hashes,
unique paths/hashes, allowed labels, complete review state, required device
group IDs, and split leakage by physical-device and near-duplicate groups.
It reads the existing family, category, MSW, and paper/cardboard split
manifests and reserves any matching existing test path/hash for test only.
The three unresolved rows in
`docs/category_disagreement_re_review_v1.csv` are explicitly blocked from
being reused as mobile examples. Existing manifests and review decisions are
read-only inputs.

The proposed pilot is exactly 30 distinct physical devices for training, 10
for validation, and 20 for a locked test set. This is a collection target, not
a model-quality guarantee. Multiple photographs of one phone count as one
device and remain in the same split. Exact duplicates are rejected; if a
near-duplicate group would cross splits, validation stops instead of silently
changing the target allocation. Additional complete device groups beyond the
60-device pilot remain unassigned in the generated manifest, with a reason.

If fewer than 60 distinct, eligible devices are available, the script reports
the shortage and writes no split manifest. It never fills gaps with pending
records, inferred labels, or fabricated examples. Once a manifest has been
written, it is never overwritten: choose a new versioned output path such as
`mobile_phone_split_manifest_v2.csv` to prepare a later revision.

The existing disagreement examples remain test-only and unresolved. They are
not eligible for phone training or validation. This workflow prepares data
only; it does not train a model or change inference behavior.
