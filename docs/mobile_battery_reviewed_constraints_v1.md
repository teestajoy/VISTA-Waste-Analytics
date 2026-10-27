# Human-reviewed Mobile/Battery split constraints (v1)

## Confirmed pairwise reviews

Bridge decisions were read from these separate files and matched by bridge ID **and exact unordered image-path pair** against `data/processed/mobile_battery_auto_group_edges_v2.csv`:

- `data/processed/mobile_battery_strong_bridge_review_v2.csv`
- `data/processed/mobile_battery_strong_bridge_21_review_v1.csv`

All three recorded decisions are `same_device`. The decision means the reviewer assessed the two images in that row as the same physical device; it does not mean the image bytes are identical. The SHA-256 hashes differ for all three pairs.

| Bridge | Image A | Image B | Original class/split | Human decision and note |
|---|---|---|---|---|
| #3 | `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg` | `train/images/IMG_20250803_220022_jpg.rf.dc0fdee22f504920709e6321bf117c22.jpg` | Mobile; test ↔ train | `same_device` — same device but different image |
| #4 | `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg` | `valid/images/IMG_20250803_220024_jpg.rf.2fc166ace133b8664434156a034ba786.jpg` | Mobile; test ↔ valid | `same_device` — same device but different image |
| #21 | `train/images/IMG_20250805_203922_jpg.rf.fbc486f138d29a49b1f3de783c184c12.jpg` | `train/images/IMG_20250805_203923_jpg.rf.5c6fc3c85b160812cf0a3d80250b24d1.jpg` | Mobile; train ↔ train | `same_device` — same physical mobile phone, but different images |

For future leakage-safe split preparation, each reviewed pair is a grouping constraint. Bridges #3 and #4 share one endpoint, so their three explicitly reviewed image paths form one connected split constraint. Bridge #21 forms a separate pairwise constraint. These reviews do not establish relationships with any images outside those endpoints. Existing strong candidate groups may independently impose additional provisional constraints; weak visual links are not automatically merged.

The grouping script now accepts both review files by default or multiple `--bridge-review-csv` options. It checks each bridge ID against the v2 edge-table row and rejects a path mismatch or conflicting decisions for the same bridge. `same_device` decisions participate in `split_constraint_group_id`; `pending`, `different_device`, and `uncertain` do not. No final split assignment is created.

## Remaining unresolved grouping conflicts

`CAND-0002` remains an oversized candidate component of **214 images**, comprising **121 strong groups** and connected by **299 weak edges**. The three reviewed pairs do not resolve this weak-link chain.

The four weak cut edges listed below remain proposals only. Each independently disconnects the saved `CAND-0002` graph if removed, but none has a human decision. Removing all four would still leave a 188-image component. Their evidence does not establish that they are unrelated; do not remove them without reviewing the actual images.

| Proposed review ID | Strong groups | Image A ↔ Image B | Evidence summary | Review concern |
|---|---|---|---|---|
| W-CUT-01 | AUTO-G0042 ↔ AUTO-G0101 | `train/images/IMG_20250803_215704_jpg.rf.0d941aee39572d80f929eb96b4640aaa.jpg` ↔ `train/images/IMG_20250805_203846_jpg.rf.9e943104abcc04412073581826afec4d.jpg` | SHA differs; pHash 6; ORB/RANSAC 0/1 (0.00); timestamps differ | Battery_Waste ↔ Mobile class conflict; single-edge cut yields 209 + 5 |
| W-CUT-04 | AUTO-G0076 ↔ AUTO-G0164 | `train/images/IMG_20250803_220312_jpg.rf.8fbb3d0ef325d176bd1eb49c8056621d.jpg` ↔ `train/images/IMG_20250822_231255_1_jpg.rf.e32021e200cdfed4065135f35ee4bca8.jpg` | SHA differs; pHash 10; ORB/RANSAC 0/0 (0.00); timestamps differ | Very weak visual support; single-edge cut yields 209 + 5 |
| W-CUT-02 | AUTO-G0048 ↔ AUTO-G0138 | `train/images/IMG_20250803_215808_jpg.rf.c547051ab6eefb114e85665da4474239.jpg` ↔ `train/images/IMG_20250822_231153_1_jpg.rf.f19abd752876b09f640fce636287ab42.jpg` | SHA differs; pHash 10; ORB/RANSAC 0/1 (0.00); timestamps differ | Minimal keypoint support; single-edge cut yields 205 + 9 |
| W-CUT-03 | AUTO-G0055 ↔ AUTO-G0153 | `train/images/IMG_20250803_220046_jpg.rf.e75fd5549c943e12ae1d5c9f747e2c80.jpg` ↔ `train/images/IMG_20250822_231220_jpg.rf.9c02b05d7b7b49ba3c228d2dc9bf9b7e.jpg` | SHA differs; pHash 10; ORB/RANSAC 4/4 (1.00); timestamps differ | Mobile ↔ Battery_Waste class conflict; 4 matches are below the v2 weak threshold; single-edge cut yields 207 + 7 |

Bridge #21 was the highest-priority unresolved strong boundary in the earlier audit. It is now marked `same_device` in the dedicated review CSV, so it is no longer unresolved as a pairwise decision. Its strong visual edge remains in the evidence graph. The remaining **24 of the original 27 strong graph bridges** have no decision in these two review files.

## Recommended next review

Review **W-CUT-01** and **W-CUT-03** first because they cross source classes, then **W-CUT-04** and **W-CUT-02** because of their weak feature support. Keep each undecided link as an unresolved split constraint. Even after these four are reviewed, hold or further inspect the remaining 188-image component before proposing any split.

This report records the existing human decisions and saved graph evidence only. It does not modify source images, labels, review CSVs, manifests, edges, splits, or model artifacts.
