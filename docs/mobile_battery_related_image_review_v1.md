# Mobile and Battery_Waste candidate review worksheet (v1)

This worksheet records the candidate relationships in [`mobile_battery_related_image_audit_v1.md`](mobile_battery_related_image_audit_v1.md). It is a human-review aid only. The audit report and source ZIP remain unchanged.

## Contents

- Worksheet: [`data/processed/mobile_battery_related_image_review_v1.csv`](../data/processed/mobile_battery_related_image_review_v1.csv)
- Audit source: [`mobile_battery_related_image_audit_v1.md`](mobile_battery_related_image_audit_v1.md)
- Candidate image rows: **166 unique image paths**. The worksheet includes both the 35 visual-similarity components and the 20 timestamp groups; where an image appears in both evidence views, its `candidate_group_id` and `similarity_evidence` fields list both.
- Timestamp member records parsed from the report: **64**.
- All decisions start as `pending`; reviewer, review date, physical-device group ID, capture-session ID, and notes are blank.
- Original class labels and split assignments are copied from the audit’s archived image identifiers and source annotation paths. No label or split is changed.

| Original class | train | valid | test | Total rows |
|---|---:|---:|---:|---:|
| Mobile | 28 | 21 | 21 | 70 |
| Battery_Waste | 51 | 24 | 21 | 96 |

## Field meanings

- `candidate_group_id`: one or more audit-local `ND-*` visual candidate group IDs and/or `TS-*` timestamp group IDs. A semicolon indicates that the same image was included by multiple candidate methods.
- `image_path`: exact archive-relative path (`train/images/...`, `valid/images/...`, or `test/images/...`).
- `original_class`, `original_split`: the existing annotation class and source split; these are informational and are not reviewer decisions.
- `exact_hash_relationship`: the v1 audit found no exact SHA-256 duplicates among these two classes. Similarity evidence is recorded separately.
- `similarity_evidence`: group-level pHash/ORB/RANSAC ranges where available and/or the shared filename timestamp. Component ranges are not per-image-pair metrics. Neither method establishes physical-device identity.
- `human_decision`: allowed values are `pending`, `same_device`, `different_device`, or `uncertain`. `pending` is the initial state. The decision is about whether this image belongs to the candidate physical-device group under review, relative to the reviewer’s documented reference image/group; it does not validate the class label.
- `reviewer`, `review_date`: person completing the visual review and date in ISO `YYYY-MM-DD` format.
- `physical_device_group_id`: populate only after a reviewer confirms a physical device relationship. Leave blank for pending, uncertain, or different-device decisions.
- `capture_session_id`: populate only when the capture session is supported by evidence or direct knowledge. Do not infer it solely from timestamps.
- `notes`: explain visual evidence, rejected similarity links, group splitting/merging, or why a decision remains uncertain.

## Review procedure

1. Use the exact `image_path` to inspect each image without changing the archive. Review all images in a candidate component together; do not rely on timestamp, pHash, ORB/RANSAC, filename suffix, or shared background alone.
2. Choose `same_device` only when the physical item can be matched visually or from reliable capture provenance. Choose `different_device` when the reviewed image is not the same physical item as the documented candidate reference. Use `uncertain` when the evidence cannot settle the relationship. Keep unreviewed rows `pending`.
3. Record the reviewer and review date for every completed decision. Document the reference image and reasoning in `notes`. Do not fill physical-device or session IDs with guessed values.
4. For connected components that combine multiple devices, review/split their memberships explicitly. Connected similarity edges are transitive candidates and do not guarantee that every member matches every other member.
5. Keep label verification separate. This worksheet does not change `Mobile` or `Battery_Waste` labels, and no visual similarity result is permission to relabel an image.

## Later split-leakage prevention

After human review is complete, create a separate, versioned grouping/split artifact. Assign confirmed images of the same physical device—and all confirmed augmentation/source-image relatives—to one group, then assign each group wholly to exactly one split. Use verified capture-session IDs to group related captures when appropriate. Do not split a group across train, validation, and test. Leave unresolved candidates out of training and validation until resolved. Preserve existing locked evaluation examples as test-only; if a candidate family links to a locked test example, keep its related members out of training and validation. Do not reuse the original split membership as proof that candidate families are independent.

No reviewer decisions, physical-device IDs, capture-session IDs, new labels, or new split assignments have been supplied or invented in this version.
