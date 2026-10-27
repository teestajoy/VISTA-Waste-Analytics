# CAND-0002 weak-cut review proposal (v1)

## Scope and graph construction

This is a read-only analysis of the saved v2 grouping graph. It uses:

- `scripts/group_mobile_battery_dataset_v2.py`
- `data/processed/mobile_battery_auto_group_edges_v2.csv`
- `data/processed/mobile_battery_auto_group_manifest_v2.csv`
- `docs/mobile_battery_automated_grouping_v2.md`
- `data/processed/mobile_battery_strong_bridge_review_v2.csv`

The script forms `CAND-*` components by connecting exact-hash, strong-candidate, and weak-match edges. In `CAND-0002`, the v2 manifest and edge table show **214 images**, **121 strong groups**, **221 strong-candidate edges**, and **299 weak edges** (520 edges total). There are no exact duplicate edges. The component spans `train`, `valid`, and `test`, and contains both `Mobile` and `Battery_Waste` labels. These graph links are candidate relationships, not device identities.

The bridge-review CSV contains two explicit `same_device` decisions: **#3 and #4**. Both are strong-candidate edges inside `AUTO-G0005`; together they link three images (one test, one train, one valid). The reviewed relationships were already connected in the v2 visual graph, so these decisions do not add new connectivity to `CAND-0002`. They are human-confirmed grouping constraints and must remain distinct from automatically inferred evidence. The worksheet decisions stored in the v2 manifest are a separate review workflow and do not resolve bridge #21.

## Minimum weak-cut candidates

The saved graph has **four weak edges whose individual removal disconnects `CAND-0002`**. Thus the minimum edge count for any split of this component is one; each row below is an alternative one-edge cut. All four connect different `strong_group_id` components. The table ranks them for review using evidence weakness, class conflict, and the size of the cut side. No edge is removed by this report.

If all four were later confirmed unrelated and omitted from split constraints, the current graph would divide into components of **188, 9, 7, 5, and 5 images**. This would still leave an oversized 188-image component. It is therefore not a complete resolution or a proposed automatic cut set.

| Review priority | Candidate edge | Strong groups | Image A (class, split) | Image B (class, split) | SHA-256 equal | pHash distance | ORB/RANSAC inliers / good matches (fraction) | Timestamp match | Single-edge graph cut | Why review |
|---:|---|---|---|---|---|---:|---|---|---:|---|
| 1 | `W-CUT-01` | `AUTO-G0042` ↔ `AUTO-G0101` | `train/images/IMG_20250803_215704_jpg.rf.0d941aee39572d80f929eb96b4640aaa.jpg` (Battery_Waste, train) | `train/images/IMG_20250805_203846_jpg.rf.9e943104abcc04412073581826afec4d.jpg` (Mobile, train) | No | 6 | 0 / 1 (0.00) | No | 209 + 5 | Very weak keypoint support and a cross-class conflict; the edge is the only connection between its strong groups in this graph. |
| 2 | `W-CUT-04` | `AUTO-G0076` ↔ `AUTO-G0164` | `train/images/IMG_20250803_220312_jpg.rf.8fbb3d0ef325d176bd1eb49c8056621d.jpg` (Battery_Waste, train) | `train/images/IMG_20250822_231255_1_jpg.rf.e32021e200cdfed4065135f35ee4bca8.jpg` (Battery_Waste, train) | No | 10 | 0 / 0 (0.00) | No | 209 + 5 | No ORB/RANSAC support; pHash is at the maximum weak-candidate distance. It is the only graph connection between distinct strong groups. |
| 3 | `W-CUT-02` | `AUTO-G0048` ↔ `AUTO-G0138` | `train/images/IMG_20250803_215808_jpg.rf.c547051ab6eefb114e85665da4474239.jpg` (Battery_Waste, train) | `train/images/IMG_20250822_231153_1_jpg.rf.f19abd752876b09f640fce636287ab42.jpg` (Battery_Waste, train) | No | 10 | 0 / 1 (0.00) | No | 205 + 9 | Minimal keypoint support and maximum-radius pHash match; sole link between distinct strong groups. |
| 4 | `W-CUT-03` | `AUTO-G0055` ↔ `AUTO-G0153` | `train/images/IMG_20250803_220046_jpg.rf.e75fd5549c943e12ae1d5c9f747e2c80.jpg` (Mobile, train) | `train/images/IMG_20250822_231220_jpg.rf.9c02b05d7b7b49ba3c228d2dc9bf9b7e.jpg` (Battery_Waste, train) | No | 10 | 4 / 4 (1.00) | No | 207 + 7 | Cross-class conflict and fewer than the v2 weak-ORB minimum of 8 inliers; it is the sole link between distinct strong groups. The 1.00 fraction comes from only four matches. |

All four are recorded as `weak_match`, are train/train, and have `same_timestamp=False`. A cut edge means only that the saved graph has no alternate route between the two resulting sides. It does not show that the images are unrelated or that removing the edge is safe. Review the image pair and its label context before changing any future grouping constraint.

## Strong bridge #21

Bridge **#21 is unresolved**: there is no #21 decision in `mobile_battery_strong_bridge_review_v2.csv`. It is a **strong-candidate**, not a weak edge, and is not part of the four weak cuts above.

| Bridge | Image A (class, split) | Image B (class, split) | SHA-256 equal | pHash distance | ORB/RANSAC inliers / good matches (fraction) | Timestamp match | Group and graph effect |
|---|---|---|---|---:|---|---|---|
| `#21` | `train/images/IMG_20250805_203922_jpg.rf.fbc486f138d29a49b1f3de783c184c12.jpg` (Mobile, train) | `train/images/IMG_20250805_203923_jpg.rf.5c6fc3c85b160812cf0a3d80250b24d1.jpg` (Mobile, train) | No | 6 | 228 / 238 (0.958) | No | `AUTO-G0121` has 2 images; removing this strong bridge separates its 1 + 1 members. In the full all-evidence component, the prior report gives a 211 + 3 cut. Both endpoints are in `CAND-0002`. |

The high visual score does not establish a shared physical device. Keep #21 in the manual review queue because it is a graph boundary, not because the visual evidence is weak.

## Recommended next manual review

1. Review bridge #21 as a separate strong-visual boundary.
2. Inspect `W-CUT-01` and `W-CUT-03` carefully because each joins different source classes and is flagged as a class conflict.
3. Review `W-CUT-04` and `W-CUT-02` for whether their very limited ORB evidence reflects a real relationship or a weak visual collision.
4. Record a human decision for every pair. Retain uncertain links as unresolved split blockers; only remove a link from future split constraints after review supports that decision.
5. Even after all four weak cuts are reviewed, do not assign splits until the remaining 188-image connected component and its other weak links are resolved or explicitly held together.

No source images, ZIP members, labels, review decisions, manifests, reports, or splits were changed. No final split assignments are proposed here.
