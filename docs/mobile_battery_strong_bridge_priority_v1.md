# Mobile/Battery strong-bridge priority review (v1)

## Scope and interpretation

This report ranks the **27** edges marked `strong_graph_bridge` in the v2 edge CSV. Sources: `data/processed/mobile_battery_auto_group_manifest_v2.csv`, `data/processed/mobile_battery_auto_group_edges_v2.csv`, and grouping rules in `docs/mobile_battery_automated_grouping_v2.md`.

A strong bridge is an edge whose removal disconnects its provisional strong-link graph component. It is not proof of a shared physical device. All 27 have different SHA-256 hashes and satisfy the saved v2 strong-visual rule (pHash distance ≤10 plus ORB/RANSAC ≥20 inliers and ≥0.50 inlier fraction). All are same-original-class edges; none is cross-class. There are no timestamp-only edges in this strong-bridge set. One edge (#23) also has a matching filename timestamp; the other 26 do not.

The all-evidence `CAND-0002` component has 214 images across 121 strong groups. Removing only three of these bridges disconnects it: #3 (212|2), #4 (213|1), and #21 (211|3). These are graph cut sizes, not identity claims. Other strong bridges split their local strong group but the all-evidence component remains connected through weaker links.

## Prioritization

1. **Immediate review and split blocking: #3 and #4.** Both cross test to train/validation and each is an articulation edge in the 214-image all-evidence component.
2. **High graph-boundary review: #21.** It is train/train and does not itself leak across original splits, but it disconnects the giant candidate graph into 211 and 3.
3. **Next: the remaining 11 cross-split bridges** (#1, #2, #5–#7, #9–#12, #16–#17). For future split construction, conservatively keep their strong groups together or block assignment pending review.
4. **Lower split-leakage urgency: the remaining 13 same-split bridges.** Preserve their strong groups for future grouping; they affect boundaries but do not directly cross original splits.

Conservatively co-group the endpoints of every strong bridge (and retain each full strong_group_id component together) for any future split proposal until reviewed. This is a split-safety rule, not a claim of shared device identity. Do not merge the 214-image component as one identity group: it is connected through weak evidence and remains an unresolved conflict.

## Bridge evidence, ranked

`Core cut` is the local strong-group side sizes if that bridge alone is removed. `214-graph cut` shows the result for the all-evidence component only when this edge disconnects it. SHA values are compared using v2's exact archive-byte hashes. ORB is RANSAC inliers / good matches (inlier fraction). Timestamp is parsed from filename tokens.

| Priority | Bridge / strong group | Image A (class, split) | Image B (class, split) | SHA-256 equal | pHash distance | ORB/RANSAC inliers / good matches (fraction) | Same timestamp | Core cut | 214-graph cut |
|---:|---|---|---|---|---:|---:|---|---:|---|
| 1 (#3) | `AUTO-G0005` | `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg`<br>Mobile, test | `train/images/IMG_20250803_220022_jpg.rf.dc0fdee22f504920709e6321bf117c22.jpg`<br>Mobile, train | No | 10 | 69/87 (0.79) | No | 2 + 1 | 212 + 2 |
| 2 (#4) | `AUTO-G0005` | `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg`<br>Mobile, test | `valid/images/IMG_20250803_220024_jpg.rf.2fc166ace133b8664434156a034ba786.jpg`<br>Mobile, valid | No | 6 | 96/119 (0.81) | No | 2 + 1 | 213 + 1 |
| 3 (#21) | `AUTO-G0121` | `train/images/IMG_20250805_203922_jpg.rf.fbc486f138d29a49b1f3de783c184c12.jpg`<br>Mobile, train | `train/images/IMG_20250805_203923_jpg.rf.5c6fc3c85b160812cf0a3d80250b24d1.jpg`<br>Mobile, train | No | 6 | 228/238 (0.96) | No | 1 + 1 | 211 + 3 |
| 4 (#1) | `AUTO-G0002` | `test/images/IMG_20250803_215727_jpg.rf.ed0c24ebb5a6e14c10a1c4d99dbf6530.jpg`<br>Battery_Waste, test | `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg`<br>Battery_Waste, valid | No | 6 | 81/84 (0.96) | No | 1 + 1 | connected |
| 5 (#2) | `AUTO-G0003` | `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg`<br>Battery_Waste, test | `valid/images/IMG_20250803_215740_jpg.rf.758acdd6ad672a13af608eafffc805dd.jpg`<br>Battery_Waste, valid | No | 4 | 94/94 (1.00) | No | 1 + 1 | connected |
| 6 (#5) | `AUTO-G0006` | `test/images/IMG_20250803_220034_jpg.rf.ff0c9e529ddd3dfcdb4934dc21b66ca1.jpg`<br>Mobile, test | `train/images/IMG_20250803_220030_jpg.rf.1f79ea661a57dc1bd9c74db29adb4ec2.jpg`<br>Mobile, train | No | 10 | 108/126 (0.86) | No | 3 + 2 | connected |
| 7 (#6) | `AUTO-G0006` | `test/images/IMG_20250803_220034_jpg.rf.ff0c9e529ddd3dfcdb4934dc21b66ca1.jpg`<br>Mobile, test | `valid/images/IMG_20250803_220037_jpg.rf.a1e1878bef855197a7260cca88694e51.jpg`<br>Mobile, valid | No | 8 | 116/132 (0.88) | No | 3 + 2 | connected |
| 8 (#7) | `AUTO-G0006` | `test/images/IMG_20250803_220038_jpg.rf.7649ac00470edad7f06ef522f2ffa0ae.jpg`<br>Mobile, test | `valid/images/IMG_20250803_220037_jpg.rf.a1e1878bef855197a7260cca88694e51.jpg`<br>Mobile, valid | No | 10 | 197/210 (0.94) | No | 4 + 1 | connected |
| 9 (#9) | `AUTO-G0008` | `test/images/IMG_20250803_220051_jpg.rf.39ebee3d80d991a2fd3bd1e036ae9b05.jpg`<br>Mobile, test | `train/images/IMG_20250803_220053_jpg.rf.143f84f401009a4e0c09c5b7c33c1ea3.jpg`<br>Mobile, train | No | 10 | 209/221 (0.95) | No | 2 + 1 | connected |
| 10 (#10) | `AUTO-G0009` | `test/images/IMG_20250803_220138_jpg.rf.c32f7dc03280433f7176d3be9f2d22a3.jpg`<br>Mobile, test | `train/images/IMG_20250803_220139_jpg.rf.36b05a61cd0dc1dc31f88bba93e1bb4a.jpg`<br>Mobile, train | No | 6 | 176/185 (0.95) | No | 1 + 1 | connected |
| 11 (#11) | `AUTO-G0017` | `test/images/IMG_20250803_220351_jpg.rf.450e818f2d954bda365821a90b9c2a70.jpg`<br>Mobile, test | `train/images/IMG_20250803_220353_jpg.rf.1a768a733a3016bb320de04418c64568.jpg`<br>Mobile, train | No | 10 | 232/237 (0.98) | No | 7 + 1 | connected |
| 12 (#12) | `AUTO-G0026` | `test/images/IMG_20250805_203918_jpg.rf.0320c79bd3505204e3c7b6d35a1d716a.jpg`<br>Mobile, test | `train/images/IMG_20250805_203917_jpg.rf.41ede18eba5e4814a68b0143d4b637b7.jpg`<br>Mobile, train | No | 6 | 274/286 (0.96) | No | 1 + 1 | connected |
| 13 (#16) | `AUTO-G0006` | `train/images/IMG_20250803_220030_jpg.rf.1f79ea661a57dc1bd9c74db29adb4ec2.jpg`<br>Mobile, train | `valid/images/IMG_20250803_220031_jpg.rf.980993866f095141d958d78983b61f11.jpg`<br>Mobile, valid | No | 4 | 131/140 (0.94) | No | 4 + 1 | connected |
| 14 (#17) | `AUTO-G0065` | `train/images/IMG_20250803_220207_jpg.rf.0f17df4cec5b7117887884be5ea29430.jpg`<br>Mobile, train | `valid/images/IMG_20250803_220206_jpg.rf.de998e9b904dcfe77468124a9e729352.jpg`<br>Mobile, valid | No | 6 | 309/314 (0.98) | No | 1 + 1 | connected |
| 15 (#8) | `AUTO-G0008` | `test/images/IMG_20250803_220050_jpg.rf.440e59bfd126dbb87848a17d6ef10ad8.jpg`<br>Mobile, test | `test/images/IMG_20250803_220051_jpg.rf.39ebee3d80d991a2fd3bd1e036ae9b05.jpg`<br>Mobile, test | No | 10 | 246/255 (0.96) | No | 2 + 1 | connected |
| 16 (#13) | `AUTO-G0039` | `train/images/IMG_20250803_215701_jpg.rf.352a524931e1295dac040a7f6381d4f1.jpg`<br>Battery_Waste, train | `train/images/IMG_20250803_215704_jpg.rf.9e7e0d24e11f27af943cdf2c6ea3daa4.jpg`<br>Battery_Waste, train | No | 10 | 66/72 (0.92) | No | 1 + 1 | connected |
| 17 (#14) | `AUTO-G0045` | `train/images/IMG_20250803_215753_jpg.rf.6b4e8f027d529c142a005c8042a782a1.jpg`<br>Battery_Waste, train | `train/images/IMG_20250803_215755_jpg.rf.86629ff209a27b91f91ada50042844d8.jpg`<br>Battery_Waste, train | No | 2 | 255/264 (0.97) | No | 1 + 1 | connected |
| 18 (#15) | `AUTO-G0046` | `train/images/IMG_20250803_215753_jpg.rf.fcef9d6acea49b62724194527689a086.jpg`<br>Battery_Waste, train | `train/images/IMG_20250803_215755_jpg.rf.02fea5708529fa5762186f65ab447c9e.jpg`<br>Battery_Waste, train | No | 10 | 233/236 (0.99) | No | 1 + 1 | connected |
| 19 (#18) | `AUTO-G0091` | `train/images/IMG_20250805_203804_jpg.rf.23acd615013748d7bdd3d547bb50935a.jpg`<br>Mobile, train | `train/images/IMG_20250805_203806_jpg.rf.a88c5e5ff1cddfe7235a919241c54687.jpg`<br>Mobile, train | No | 10 | 241/244 (0.99) | No | 1 + 1 | connected |
| 20 (#19) | `AUTO-G0108` | `train/images/IMG_20250805_203902_jpg.rf.bb8f8213eb4744ef892e1fd9c46edc8d.jpg`<br>Mobile, train | `train/images/IMG_20250805_203903_1_jpg.rf.e6eac1f51cfb25d40b34be0a5ec6f1d5.jpg`<br>Mobile, train | No | 10 | 220/231 (0.95) | No | 1 + 1 | connected |
| 21 (#20) | `AUTO-G0112` | `train/images/IMG_20250805_203907_jpg.rf.842e4fc0cf7a9464305766b439d364c5.jpg`<br>Mobile, train | `train/images/IMG_20250805_203908_jpg.rf.f30f1e3780a6cd716c1f2a3b2f984f8b.jpg`<br>Mobile, train | No | 2 | 198/212 (0.93) | No | 1 + 1 | connected |
| 22 (#22) | `AUTO-G0128` | `train/images/IMG_20250822_231134_1_jpg.rf.82a4dc567acd78e78794d5bba1b3300a.jpg`<br>Battery_Waste, train | `train/images/IMG_20250822_231136_jpg.rf.892f7aaf1ac6a2fe9c249f6744bb872e.jpg`<br>Battery_Waste, train | No | 10 | 116/126 (0.92) | No | 1 + 1 | connected |
| 23 (#23) | `AUTO-G0135` | `train/images/IMG_20250822_231138_1_jpg.rf.793734fc9cc4adbb935973beb2fd4034.jpg`<br>Battery_Waste, train | `train/images/IMG_20250822_231138_jpg.rf.c2066d8066e40dd14cf4c6af5b886414.jpg`<br>Battery_Waste, train | No | 10 | 241/247 (0.98) | Yes | 1 + 1 | connected |
| 24 (#24) | `AUTO-G0156` | `train/images/IMG_20250822_231226_jpg.rf.af949f552ee40abb9ba0d2ea22afdb16.jpg`<br>Battery_Waste, train | `train/images/IMG_20250822_231229_jpg.rf.74aa09b68e143f0d05c0449cf7200946.jpg`<br>Battery_Waste, train | No | 6 | 101/110 (0.92) | No | 1 + 1 | connected |
| 25 (#25) | `AUTO-G0169` | `train/images/IMG_20250822_231256_jpg.rf.b5c48e1da6f07d1a4053ac81da1a3b2c.jpg`<br>Battery_Waste, train | `train/images/IMG_20250822_231257_jpg.rf.faf456eebb5f026fde69eb667b9ae98e.jpg`<br>Battery_Waste, train | No | 10 | 144/148 (0.97) | No | 1 + 1 | connected |
| 26 (#26) | `AUTO-G0172` | `valid/images/IMG_20250803_220126_jpg.rf.16a1c934ac06cbc5d2ee0f8b4c02e96b.jpg`<br>Mobile, valid | `valid/images/IMG_20250803_220127_jpg.rf.29454ccaf0bd8058e12a5925d3619982.jpg`<br>Mobile, valid | No | 4 | 89/94 (0.95) | No | 1 + 1 | connected |
| 27 (#27) | `AUTO-G0175` | `valid/images/IMG_20250805_203851_jpg.rf.30549cb0368d0ebf082a10e12534ea69.jpg`<br>Mobile, valid | `valid/images/IMG_20250805_203852_jpg.rf.14242e22d41ed3ca487f99b193502f87.jpg`<br>Mobile, valid | No | 6 | 275/285 (0.96) | No | 1 + 1 | connected |

## Timestamp-only and conflict distinction

The 27 rows above are strong visual matches; timestamp does not establish or drive their strong classification. Same timestamp is true only for #23 and is corroborating filename metadata. Timestamp-only edges, weaker visual edges, and timestamp/visual conflicts remain separate in the v2 edge file and are not elevated by this report. The v2 report records 351 weak edges and 117 evidence conflicts overall; those should not be chained into device identities. The 214-image component is a conservative unresolved split-blocking boundary, not one asserted group identity.

## Recommended handling

- Keep strong-group members together for a later split proposal until the relevant bridge is reviewed; do not move or relabel source examples based on this report.
- Review #3 and #4 first, then #21, then the other cross-split bridges. Record visual relationship separately from physical-device identity.
- Keep weak links as explicit constraints or unresolved conflicts. If respecting all candidate links produces the 214-image component, do not split inside it; resolve weak bridges or hold that cohort out from split assignment.
- No source labels, review decisions, original splits, model artifacts, or manifests were changed. No new splits are assigned.