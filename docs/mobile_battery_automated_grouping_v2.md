# Mobile and Battery_Waste automated grouping report (v2)

- Source archive read in place: `~/Downloads/VISTA_mobile_dataset_raw.zip` (local user Downloads folder)
- No archive members were extracted, moved, renamed, or modified.
- Target image records: **299**; all output split assignments are intentionally blank.
- Pairwise combinations: 44,551; candidate evidence edges saved: 618; no-evidence pairs omitted from the edge table: 43,933.
- Starting review worksheet rows carried forward: 166.

## Original split counts

| Class | train | valid | test | Total |
|---|---:|---:|---:|---:|
| Mobile | 99 | 26 | 26 | 151 |
| Battery_Waste | 99 | 26 | 23 | 148 |

## Evidence rules

- **Exact duplicate:** original ZIP-member SHA-256 values are identical. This is a file-level match, not physical-device identity.
- **Strong candidate:** pHash Hamming distance ≤10 AND ORB/RANSAC has at least 20 inliers with inlier fraction ≥0.50. These thresholds reuse the prior audit. Strong links form provisional candidate groups.
- **Weak match:** same parsed filename timestamp, or perceptual/keypoint evidence that does not meet the strong threshold. Weak matches do not merge provisional strong groups; they are recorded as split-blocking bridges pending review.
- **Oversized transitive candidate graph:** if chaining all exact, strong, and weak edges creates a component larger than 12 images, it is flagged as a grouping conflict rather than accepted as one automatic group. Its unresolved weak bridges must be reviewed before split assignment.
- **Unrelated:** no exact hash, same-timestamp, pHash-shortlist, or qualifying ORB/RANSAC evidence. These pairs are counted but omitted from the edge CSV to keep the artifact compact.
- Exact timestamps, pHash, ORB/RANSAC matches, and image similarity do not prove images depict the same physical device. A timestamp/visual disagreement and cross-class label conflict are explicitly flagged.

## Group and match counts

- Exact duplicate groups: **0**; duplicate member images: **0**.
- Strong candidate groups (including singleton groups): **175**; non-singleton strong groups: **49**; members in non-singleton strong groups: **173**.
- Strong candidate pair edges: **267**; weak match/bridge edges: **351**; exact-match edges: **0**.
- Images with no candidate edges (unrelated singletons): **8**.
- Strong graph bridge edges requiring boundary review: **27**.
- Evidence conflict edges: **117**; cross-class conflict edges: **26**.
- Strong components larger than 12 images (review flagged): **0**.

### Strong-group size distribution

| Images per group | Number of groups |
|---:|---:|
| 1 | 126 |
| 2 | 18 |
| 3 | 13 |
| 4 | 7 |
| 5 | 3 |
| 6 | 5 |
| 8 | 2 |
| 9 | 1 |

## Transitive chain and weak-bridge inspection

- Connected components using all exact, strong, and weak candidate edges: **24**; largest component: **214 images**.
- Connected components using weak edges only: **87**; largest weak-only component: **168 images**.
- All-evidence components larger than 12 images: **1**. These are not merged into `strong_group_id`; they are flagged as unresolved split conflicts.
- Automatically coalescing every weak edge would create an oversized transitive component. This is a documented conflict: weak links remain explicit constraints for review, not one identity group. No split should be assigned inside an unresolved oversized component.

### All-evidence component size distribution

| Images per component | Number of components |
|---:|---:|
| 1 | 8 |
| 2 | 1 |
| 3 | 3 |
| 4 | 4 |
| 5 | 2 |
| 6 | 1 |
| 7 | 1 |
| 8 | 1 |
| 9 | 1 |
| 10 | 1 |
| 214 | 1 |

### Oversized components requiring bridge review

- **CAND-0002: 214 images**, 121 strong groups, classes=Battery_Waste,Mobile, original splits=test,train,valid, 299 weak links (100 flagged conflicts). This graph component is not an identity group; resolve its weak bridges or keep it out of split assignment.
  - Strong core `AUTO-G0002` (2 images): `test/images/IMG_20250803_215727_jpg.rf.ed0c24ebb5a6e14c10a1c4d99dbf6530.jpg`; `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg`
  - Strong core `AUTO-G0003` (2 images): `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg`; `valid/images/IMG_20250803_215740_jpg.rf.758acdd6ad672a13af608eafffc805dd.jpg`
  - Strong core `AUTO-G0004` (3 images): `test/images/IMG_20250803_215807_jpg.rf.949f4e1a94325c40ad658940123c68d8.jpg`; `test/images/IMG_20250803_215810_jpg.rf.bf03fc4486612281e874dcbde3267f8b.jpg`; `train/images/IMG_20250803_215808_jpg.rf.8a06ce738b95b31fc02705ce61b6d0ad.jpg`
  - Strong core `AUTO-G0005` (3 images): `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg`; `train/images/IMG_20250803_220022_jpg.rf.dc0fdee22f504920709e6321bf117c22.jpg`; `valid/images/IMG_20250803_220024_jpg.rf.2fc166ace133b8664434156a034ba786.jpg`
  - Strong core `AUTO-G0007` (3 images): `test/images/IMG_20250803_220045_jpg.rf.337135edaed79d430b95b610ec32c0db.jpg`; `train/images/IMG_20250803_220046_jpg.rf.593a44abf20cee25f06845f455285742.jpg`; `train/images/IMG_20250803_220047_jpg.rf.f69036399cb844d4fa35545cbe8bf18c.jpg`
  - Strong core `AUTO-G0012` (3 images): `test/images/IMG_20250803_220228_jpg.rf.8e94e7ce7153325b8e181e6a1d331ef0.jpg`; `train/images/IMG_20250803_220226_jpg.rf.ff911118579e3a71da019d386eb1d7eb.jpg`; `valid/images/IMG_20250803_220227_jpg.rf.c42373ef7c64d08b36804cc7fcad4d15.jpg`
  - Strong core `AUTO-G0013` (3 images): `test/images/IMG_20250803_220244_jpg.rf.25e694fad0069d6b3720bcfeff342f09.jpg`; `train/images/IMG_20250803_220242_jpg.rf.c47cf376f90ec8109c0115c6ef00398e.jpg`; `train/images/IMG_20250803_220243_jpg.rf.c17d27aa58b11aa0475d5b41c8ddbd97.jpg`
  - Strong core `AUTO-G0015` (3 images): `test/images/IMG_20250803_220311_jpg.rf.72c9015fb578049a797e0d93bd65aee3.jpg`; `train/images/IMG_20250803_220312_jpg.rf.07000a53f58fd8b7ca0a0d97613d25f3.jpg`; `valid/images/IMG_20250803_220313_jpg.rf.64f586e62091d23ee0746e2a666392b5.jpg`
  - Strong core `AUTO-G0016` (6 images): `test/images/IMG_20250803_220320_jpg.rf.10f96e152fbca7bb72a58524897b976a.jpg`; `test/images/IMG_20250803_220323_1_jpg.rf.573457ae657057432e59564c621a62b2.jpg`; `train/images/IMG_20250803_220321_jpg.rf.7d603d397b0d570f006ad3ed948b1fd8.jpg`; …
  - Strong core `AUTO-G0017` (8 images): `test/images/IMG_20250803_220351_jpg.rf.450e818f2d954bda365821a90b9c2a70.jpg`; `train/images/IMG_20250803_220350_jpg.rf.51437498308c9763e0717636cb26c374.jpg`; `train/images/IMG_20250803_220352_jpg.rf.af73197979f1b112320f3f06a0058c12.jpg`; …
  - Strong core `AUTO-G0023` (4 images): `test/images/IMG_20250805_203805_jpg.rf.ecf4905797f5b1dbc5887402d335664a.jpg`; `train/images/IMG_20250805_203804_jpg.rf.d0d25dd17b163cc60b1a269ea27346dd.jpg`; `train/images/IMG_20250805_203806_jpg.rf.7fd019ac3a6efc21e4b37227c5bb73e1.jpg`; …
  - Strong core `AUTO-G0024` (3 images): `test/images/IMG_20250805_203845_jpg.rf.8c15db31b863733a22036fd715b6cc3a.jpg`; `train/images/IMG_20250805_203846_jpg.rf.66d7b40dd70798c0e6a723d0bdc26269.jpg`; `valid/images/IMG_20250805_203847_jpg.rf.0b7b578193561a5e7b7297568d1862c0.jpg`
  - Strong core `AUTO-G0029` (5 images): `test/images/IMG_20250822_231124_jpg.rf.454eb5a36ebdb2ad770c60899ed1ff5d.jpg`; `test/images/IMG_20250822_231126_jpg.rf.7712b4701c0e3c8ad2f513a37cac8107.jpg`; `train/images/IMG_20250822_231125_1_jpg.rf.055929b00c16ff17c6d4a49b7cef58a9.jpg`; …
  - Strong core `AUTO-G0030` (5 images): `test/images/IMG_20250822_231151_jpg.rf.2607ab2dc8acaf81e39a0adf8e0950f7.jpg`; `test/images/IMG_20250822_231152_jpg.rf.e62543112ad7bd0d4593d533a2e3f231.jpg`; `train/images/IMG_20250822_231153_1_jpg.rf.055d65f1f0714e9fda6d890f79bc15ab.jpg`; …
  - Strong core `AUTO-G0031` (6 images): `test/images/IMG_20250822_231209_1_jpg.rf.fcf651145e2d95b81efe4e9904f3cfed.jpg`; `test/images/IMG_20250822_231210_jpg.rf.3e928d15c853583abf8b89eff296af81.jpg`; `test/images/IMG_20250822_231211_jpg.rf.29439c714ba00a43492ccf5b2673e53f.jpg`; …
  - Strong core `AUTO-G0032` (6 images): `test/images/IMG_20250822_231221_jpg.rf.f1b09c45232ce7f53d9929c1a6024cc0.jpg`; `train/images/IMG_20250822_231219_jpg.rf.03d08967efba3804f23ba15aac3d4cca.jpg`; `train/images/IMG_20250822_231220_jpg.rf.847cf565415d2346e0fabc16a3fb1f28.jpg`; …
  - Strong core `AUTO-G0033` (6 images): `test/images/IMG_20250822_231228_1_jpg.rf.8374d1d76854c1115c33fd7b7bedb107.jpg`; `test/images/IMG_20250822_231228_jpg.rf.5878b0c9eb72ca9d1cdcfc596fc61234.jpg`; `train/images/IMG_20250822_231226_jpg.rf.a384c4024b55ba7bf2eda542344eef69.jpg`; …
  - Strong core `AUTO-G0034` (8 images): `test/images/IMG_20250822_231238_jpg.rf.3c4cd2453b779bbf5224c35672da6c9d.jpg`; `test/images/IMG_20250822_231239_jpg.rf.a708ada26d463e1fc8125634fc78cbf5.jpg`; `test/images/IMG_20250822_231241_1_jpg.rf.08f4254c1a6be413d72d48bc233c8175.jpg`; …
  - Strong core `AUTO-G0035` (6 images): `test/images/IMG_20250822_231256_1_jpg.rf.aea99bc61d2369c19f56241e3f0876cf.jpg`; `train/images/IMG_20250822_231255_1_jpg.rf.5f530548951cb35377a1987db865ecc0.jpg`; `train/images/IMG_20250822_231255_jpg.rf.36ef9257a4993f8d263c5dd4cb7bf7f5.jpg`; …
  - Strong core `AUTO-G0036` (1 images): `train/images/IMG_20250803_215648_jpg.rf.76ccb3808e396afdab350d5bc97f6366.jpg`
  - Strong core `AUTO-G0037` (4 images): `train/images/IMG_20250803_215648_jpg.rf.c339d9561861e32aec3649527f4423f7.jpg`; `valid/images/IMG_20250803_215638_jpg.rf.ba4f3127b341131ea269136e3d2b124e.jpg`; `valid/images/IMG_20250803_215639_jpg.rf.fbe999007ecf4fd037195142c1702267.jpg`; …
  - Strong core `AUTO-G0038` (1 images): `train/images/IMG_20250803_215648_jpg.rf.e06da5e80cf6d6815419245cec540057.jpg`
  - Strong core `AUTO-G0039` (2 images): `train/images/IMG_20250803_215701_jpg.rf.352a524931e1295dac040a7f6381d4f1.jpg`; `train/images/IMG_20250803_215704_jpg.rf.9e7e0d24e11f27af943cdf2c6ea3daa4.jpg`
  - Strong core `AUTO-G0040` (1 images): `train/images/IMG_20250803_215701_jpg.rf.953215ef1f4ce660a9ce216ed10cd5fe.jpg`
  - Strong core `AUTO-G0041` (1 images): `train/images/IMG_20250803_215701_jpg.rf.e582db8c0d1f0560b6d54cc15f102097.jpg`
  - Strong core `AUTO-G0042` (1 images): `train/images/IMG_20250803_215704_jpg.rf.0d941aee39572d80f929eb96b4640aaa.jpg`
  - Strong core `AUTO-G0043` (1 images): `train/images/IMG_20250803_215704_jpg.rf.f00cf5b73c3af9c585d0df618a428c39.jpg`
  - Strong core `AUTO-G0048` (1 images): `train/images/IMG_20250803_215808_jpg.rf.c547051ab6eefb114e85665da4474239.jpg`
  - Strong core `AUTO-G0049` (1 images): `train/images/IMG_20250803_215808_jpg.rf.e0e84a12bb13158f05811974c97c5c67.jpg`
  - Strong core `AUTO-G0050` (1 images): `train/images/IMG_20250803_220022_jpg.rf.1b9bf10def62ebe2424913192f6033b6.jpg`
  - Strong core `AUTO-G0051` (1 images): `train/images/IMG_20250803_220022_jpg.rf.2cd6a4959481bac2347cd2b2f4dc6d61.jpg`
  - Strong core `AUTO-G0054` (1 images): `train/images/IMG_20250803_220046_jpg.rf.3a0decfaa0e347e8daf2186f2dca2af1.jpg`
  - Strong core `AUTO-G0055` (1 images): `train/images/IMG_20250803_220046_jpg.rf.e75fd5549c943e12ae1d5c9f747e2c80.jpg`
  - Strong core `AUTO-G0056` (1 images): `train/images/IMG_20250803_220047_jpg.rf.13ad331e1f469809c4b1ba6f0748d0cf.jpg`
  - Strong core `AUTO-G0057` (1 images): `train/images/IMG_20250803_220047_jpg.rf.dcce9a993294f06fa6c2078e37f74309.jpg`
  - Strong core `AUTO-G0060` (1 images): `train/images/IMG_20250803_220114_jpg.rf.402c2f2a849eda8b99f516509e711de9.jpg`
  - Strong core `AUTO-G0061` (3 images): `train/images/IMG_20250803_220114_jpg.rf.68d25d49b60c21e990f42a77571309d7.jpg`; `valid/images/IMG_20250803_220117_jpg.rf.0594f79bcf0bff0635472395fec31752.jpg`; `valid/images/IMG_20250803_220119_jpg.rf.64e2198d7a2d24e7786b2bbc40de885d.jpg`
  - Strong core `AUTO-G0062` (1 images): `train/images/IMG_20250803_220114_jpg.rf.b245c7433620d8fc6007ccff153e2fb3.jpg`
  - Strong core `AUTO-G0068` (1 images): `train/images/IMG_20250803_220226_jpg.rf.b4392a8eca529f4ed6fed45e99eeeb5e.jpg`
  - Strong core `AUTO-G0069` (1 images): `train/images/IMG_20250803_220226_jpg.rf.eb9c0141249018f67e7bdc8bf241a8a2.jpg`
  - Strong core `AUTO-G0070` (1 images): `train/images/IMG_20250803_220242_jpg.rf.7522feb05c688af1d5d44ab984f05a1c.jpg`
  - Strong core `AUTO-G0071` (1 images): `train/images/IMG_20250803_220242_jpg.rf.8d0d22aab4dcc12eed7a126bce99cd55.jpg`
  - Strong core `AUTO-G0072` (1 images): `train/images/IMG_20250803_220243_jpg.rf.0b0201bb99cd8531e9b7ca91ed720816.jpg`
  - Strong core `AUTO-G0073` (1 images): `train/images/IMG_20250803_220243_jpg.rf.2b74ac43ae204204eedb898b9845ea08.jpg`
  - Strong core `AUTO-G0076` (1 images): `train/images/IMG_20250803_220312_jpg.rf.8fbb3d0ef325d176bd1eb49c8056621d.jpg`
  - Strong core `AUTO-G0077` (1 images): `train/images/IMG_20250803_220312_jpg.rf.c5b01155df012f0543c0ea06f87af29e.jpg`
  - Strong core `AUTO-G0078` (1 images): `train/images/IMG_20250803_220321_jpg.rf.18be45e7ac54d6c3c9f1bac80e64585e.jpg`
  - Strong core `AUTO-G0079` (1 images): `train/images/IMG_20250803_220321_jpg.rf.25adc2d462f062f4163541ffbcb1da33.jpg`
  - Strong core `AUTO-G0080` (1 images): `train/images/IMG_20250803_220350_jpg.rf.1f5aa0f107ac49e368b6c16523ac1c14.jpg`
  - Strong core `AUTO-G0081` (1 images): `train/images/IMG_20250803_220350_jpg.rf.cc9680936d403e013dd1e8a4863afe8d.jpg`
  - Strong core `AUTO-G0082` (1 images): `train/images/IMG_20250803_220352_jpg.rf.8d5cd74618268102d6ee8e431e2dc659.jpg`
  - Strong core `AUTO-G0083` (1 images): `train/images/IMG_20250803_220352_jpg.rf.9962226a88efa23a6ca92f4e4337f01f.jpg`
  - Strong core `AUTO-G0084` (1 images): `train/images/IMG_20250803_220353_jpg.rf.ce1bfbf5249837f379ded71315f8e5a0.jpg`
  - Strong core `AUTO-G0091` (2 images): `train/images/IMG_20250805_203804_jpg.rf.23acd615013748d7bdd3d547bb50935a.jpg`; `train/images/IMG_20250805_203806_jpg.rf.a88c5e5ff1cddfe7235a919241c54687.jpg`
  - Strong core `AUTO-G0092` (1 images): `train/images/IMG_20250805_203804_jpg.rf.2aa2f4ad307106ebef06541aae137975.jpg`
  - Strong core `AUTO-G0093` (1 images): `train/images/IMG_20250805_203806_jpg.rf.34c678b8361cf794b5af0e3b63a58a07.jpg`
  - Strong core `AUTO-G0101` (1 images): `train/images/IMG_20250805_203846_jpg.rf.9e943104abcc04412073581826afec4d.jpg`
  - Strong core `AUTO-G0102` (1 images): `train/images/IMG_20250805_203846_jpg.rf.cfa5e7649e94c6cde734514ca4acd259.jpg`
  - Strong core `AUTO-G0106` (1 images): `train/images/IMG_20250805_203902_jpg.rf.327b0dd828626fc4dc5867660d12285a.jpg`
  - Strong core `AUTO-G0107` (3 images): `train/images/IMG_20250805_203902_jpg.rf.a23fe3b1c30257e8947ef9b1f48a0746.jpg`; `train/images/IMG_20250805_203903_1_jpg.rf.a847a14922cb7973b83d4e2f3c655fb5.jpg`; `train/images/IMG_20250805_203903_jpg.rf.e34bfc57526f72ced179b5397bc93f15.jpg`
  - Strong core `AUTO-G0108` (2 images): `train/images/IMG_20250805_203902_jpg.rf.bb8f8213eb4744ef892e1fd9c46edc8d.jpg`; `train/images/IMG_20250805_203903_1_jpg.rf.e6eac1f51cfb25d40b34be0a5ec6f1d5.jpg`
  - Strong core `AUTO-G0109` (1 images): `train/images/IMG_20250805_203903_1_jpg.rf.f5cee8b017104a9e8573eb4524f0ff98.jpg`
  - Strong core `AUTO-G0110` (1 images): `train/images/IMG_20250805_203903_jpg.rf.8f402e3b8d9da9885fbbcabec1f42576.jpg`
  - Strong core `AUTO-G0111` (1 images): `train/images/IMG_20250805_203903_jpg.rf.f23b1a96ab603bb4dbb0d5dc1b359a97.jpg`
  - Strong core `AUTO-G0112` (2 images): `train/images/IMG_20250805_203907_jpg.rf.842e4fc0cf7a9464305766b439d364c5.jpg`; `train/images/IMG_20250805_203908_jpg.rf.f30f1e3780a6cd716c1f2a3b2f984f8b.jpg`
  - Strong core `AUTO-G0113` (1 images): `train/images/IMG_20250805_203907_jpg.rf.bfe8f9064ad57b8bfa2534f603f5300f.jpg`
  - Strong core `AUTO-G0114` (1 images): `train/images/IMG_20250805_203907_jpg.rf.dd69ee28911b2c95d949c538ac5799d4.jpg`
  - Strong core `AUTO-G0115` (1 images): `train/images/IMG_20250805_203908_jpg.rf.4c4645aca4c09485234d7bb650187ba9.jpg`
  - Strong core `AUTO-G0116` (1 images): `train/images/IMG_20250805_203908_jpg.rf.eda7b9d24b1a975e131ef6c9b7f43fe2.jpg`
  - Strong core `AUTO-G0119` (1 images): `train/images/IMG_20250805_203922_jpg.rf.9772c2edbd0d2281767d124674a04042.jpg`
  - Strong core `AUTO-G0120` (1 images): `train/images/IMG_20250805_203922_jpg.rf.b747b03d920d3d0c33908474edce093c.jpg`
  - Strong core `AUTO-G0121` (2 images): `train/images/IMG_20250805_203922_jpg.rf.fbc486f138d29a49b1f3de783c184c12.jpg`; `train/images/IMG_20250805_203923_jpg.rf.5c6fc3c85b160812cf0a3d80250b24d1.jpg`
  - Strong core `AUTO-G0122` (1 images): `train/images/IMG_20250805_203923_jpg.rf.c391e7c791e4efc10d2f5cd5e5ffabea.jpg`
  - Strong core `AUTO-G0123` (1 images): `train/images/IMG_20250805_203923_jpg.rf.d38a90effba05a5b6fd5e7ac1cef65ec.jpg`
  - Strong core `AUTO-G0124` (1 images): `train/images/IMG_20250822_231125_1_jpg.rf.24e5f0335754fa334350f28e5fc26691.jpg`
  - Strong core `AUTO-G0125` (1 images): `train/images/IMG_20250822_231125_1_jpg.rf.51f8307cc11b5cddecd0ca3e02555e70.jpg`
  - Strong core `AUTO-G0126` (9 images): `train/images/IMG_20250822_231134_1_jpg.rf.19857d21bf15db9b7b5a3e2669c53f03.jpg`; `train/images/IMG_20250822_231135_1_jpg.rf.9c3779c2a1dd1c261e91e6a7e74a3e6b.jpg`; `train/images/IMG_20250822_231136_jpg.rf.744ffa1266ae25359cf64542d5c6e0ef.jpg`; …
  - Strong core `AUTO-G0127` (1 images): `train/images/IMG_20250822_231134_1_jpg.rf.3f4911838e7d399e2c29009938d92851.jpg`
  - Strong core `AUTO-G0128` (2 images): `train/images/IMG_20250822_231134_1_jpg.rf.82a4dc567acd78e78794d5bba1b3300a.jpg`; `train/images/IMG_20250822_231136_jpg.rf.892f7aaf1ac6a2fe9c249f6744bb872e.jpg`
  - Strong core `AUTO-G0129` (1 images): `train/images/IMG_20250822_231135_1_jpg.rf.40d27b8048589bd7f6a91ff8c07f1d84.jpg`
  - Strong core `AUTO-G0130` (1 images): `train/images/IMG_20250822_231135_1_jpg.rf.d42078c33df83bcadfbe932693eb51fb.jpg`
  - Strong core `AUTO-G0131` (1 images): `train/images/IMG_20250822_231136_jpg.rf.facafbb903d0e49f4895b43e2448f178.jpg`
  - Strong core `AUTO-G0132` (1 images): `train/images/IMG_20250822_231137_jpg.rf.082d8b78b35dc56a1b88168f6e387d7d.jpg`
  - Strong core `AUTO-G0133` (1 images): `train/images/IMG_20250822_231137_jpg.rf.96ed63464751f15129b76e9093788b0a.jpg`
  - Strong core `AUTO-G0134` (1 images): `train/images/IMG_20250822_231138_1_jpg.rf.0ac4f5a7cb3daffeb355a4abc877a794.jpg`
  - Strong core `AUTO-G0135` (2 images): `train/images/IMG_20250822_231138_1_jpg.rf.793734fc9cc4adbb935973beb2fd4034.jpg`; `train/images/IMG_20250822_231138_jpg.rf.c2066d8066e40dd14cf4c6af5b886414.jpg`
  - Strong core `AUTO-G0136` (1 images): `train/images/IMG_20250822_231138_jpg.rf.327c80badfebf025a88bbe352e1fd3cc.jpg`
  - Strong core `AUTO-G0137` (1 images): `train/images/IMG_20250822_231153_1_jpg.rf.0ccc4241354b45d12d7828899bc92f05.jpg`
  - Strong core `AUTO-G0138` (1 images): `train/images/IMG_20250822_231153_1_jpg.rf.f19abd752876b09f640fce636287ab42.jpg`
  - Strong core `AUTO-G0139` (1 images): `train/images/IMG_20250822_231155_jpg.rf.504426a470f2b291095627bb367e7dad.jpg`
  - Strong core `AUTO-G0140` (1 images): `train/images/IMG_20250822_231155_jpg.rf.af8d1320d7a24080846ef501f2abf92c.jpg`
  - Strong core `AUTO-G0141` (4 images): `train/images/IMG_20250822_231200_jpg.rf.0f6db813145281c72c353be21cbd72aa.jpg`; `train/images/IMG_20250822_231202_jpg.rf.1763ce102999536a602ed04b81bf0435.jpg`; `valid/images/IMG_20250822_231201_1_jpg.rf.3ee4562ffda16d4de26bfe881d3c0a8b.jpg`; …
  - Strong core `AUTO-G0142` (1 images): `train/images/IMG_20250822_231200_jpg.rf.81019355e8515e8e16bfa79d6ab2b910.jpg`
  - Strong core `AUTO-G0143` (1 images): `train/images/IMG_20250822_231200_jpg.rf.ff53aaba394c0304da0a909829ab90c5.jpg`
  - Strong core `AUTO-G0144` (1 images): `train/images/IMG_20250822_231202_jpg.rf.9985948cecec55daa36ed7a5c51b5073.jpg`
  - Strong core `AUTO-G0145` (1 images): `train/images/IMG_20250822_231202_jpg.rf.f9ca8ef5be33db85e95f12d237090966.jpg`
  - Strong core `AUTO-G0146` (1 images): `train/images/IMG_20250822_231209_jpg.rf.5fbcecd2f828c4d73be632426a997488.jpg`
  - Strong core `AUTO-G0147` (1 images): `train/images/IMG_20250822_231209_jpg.rf.71062767ff841e54501edf3ddc7a9c7a.jpg`
  - Strong core `AUTO-G0148` (1 images): `train/images/IMG_20250822_231210_1_jpg.rf.7e7d56093797517d49417300b1eceb44.jpg`
  - Strong core `AUTO-G0149` (1 images): `train/images/IMG_20250822_231210_1_jpg.rf.b643e8ffe7294d17bcf888a766981b1a.jpg`
  - Strong core `AUTO-G0150` (1 images): `train/images/IMG_20250822_231219_jpg.rf.152f0425291ed55f66ec4bea49524c43.jpg`
  - Strong core `AUTO-G0151` (1 images): `train/images/IMG_20250822_231219_jpg.rf.392d276fdfc40327967ce3017eaf01ac.jpg`
  - Strong core `AUTO-G0152` (1 images): `train/images/IMG_20250822_231220_jpg.rf.2b36d46a7e22d101be3c9448b72ef6f1.jpg`
  - Strong core `AUTO-G0153` (1 images): `train/images/IMG_20250822_231220_jpg.rf.9c02b05d7b7b49ba3c228d2dc9bf9b7e.jpg`
  - Strong core `AUTO-G0154` (1 images): `train/images/IMG_20250822_231222_jpg.rf.6da281988f9a7ad6de6a237c773340b0.jpg`
  - Strong core `AUTO-G0155` (1 images): `train/images/IMG_20250822_231222_jpg.rf.a82c3365421ee2a0e83bc207acc9927d.jpg`
  - Strong core `AUTO-G0156` (2 images): `train/images/IMG_20250822_231226_jpg.rf.af949f552ee40abb9ba0d2ea22afdb16.jpg`; `train/images/IMG_20250822_231229_jpg.rf.74aa09b68e143f0d05c0449cf7200946.jpg`
  - Strong core `AUTO-G0157` (1 images): `train/images/IMG_20250822_231226_jpg.rf.f5c80448ec1e3edadd7fe58129065dd8.jpg`
  - Strong core `AUTO-G0158` (1 images): `train/images/IMG_20250822_231227_jpg.rf.03756de636cf7cc1ebd60892c444e82f.jpg`
  - Strong core `AUTO-G0159` (1 images): `train/images/IMG_20250822_231229_jpg.rf.9286a8bea2098e273163b8a58657f4c0.jpg`
  - Strong core `AUTO-G0160` (1 images): `train/images/IMG_20250822_231239_1_jpg.rf.1b55959316611efa584bdc9a073bc60b.jpg`
  - Strong core `AUTO-G0161` (1 images): `train/images/IMG_20250822_231239_1_jpg.rf.ff7c20ffa1b97e6f85e4860b6c2eefbc.jpg`
  - Strong core `AUTO-G0162` (1 images): `train/images/IMG_20250822_231240_1_jpg.rf.29e3db85d081590d2c545b6a25bf448f.jpg`
  - Strong core `AUTO-G0163` (1 images): `train/images/IMG_20250822_231240_1_jpg.rf.c599e12109d4ea69fd2c676c98f032ff.jpg`
  - Strong core `AUTO-G0164` (1 images): `train/images/IMG_20250822_231255_1_jpg.rf.e32021e200cdfed4065135f35ee4bca8.jpg`
  - Strong core `AUTO-G0165` (1 images): `train/images/IMG_20250822_231255_1_jpg.rf.e3ebfec65816d16aff257a0db04a3372.jpg`
  - Strong core `AUTO-G0166` (1 images): `train/images/IMG_20250822_231255_jpg.rf.7c431d715c6666390961f2fd29ff327e.jpg`
  - Strong core `AUTO-G0167` (1 images): `train/images/IMG_20250822_231255_jpg.rf.874ad060677087d05bc1987392adbfbb.jpg`
  - Strong core `AUTO-G0168` (1 images): `train/images/IMG_20250822_231256_jpg.rf.77db81774004ef79c28d543c812b7e54.jpg`
  - Strong core `AUTO-G0169` (2 images): `train/images/IMG_20250822_231256_jpg.rf.b5c48e1da6f07d1a4053ac81da1a3b2c.jpg`; `train/images/IMG_20250822_231257_jpg.rf.faf456eebb5f026fde69eb667b9ae98e.jpg`
  - Strong core `AUTO-G0170` (1 images): `train/images/IMG_20250822_231257_jpg.rf.2db20ee5a5e4fda240d26419f21c7ee8.jpg`

## Original split leakage indicators

- Strong provisional components spanning more than one original split: **35**.
- Strong/exact candidate edges directly crossing original splits: **174**.
- Weak candidate bridges crossing original splits: **56**.
- Original split assignments are retained only as source metadata. No new train/validation/test membership is assigned.
- A future split creator must keep each strong group together and must also avoid splitting across unresolved weak bridges; otherwise resolve or document the bridge before assigning splits.

## Strong components and bridge boundaries

### AUTO-G0002 — 2 images; classes=Battery_Waste; original splits=test,valid; status=provisional_strong_component
- `test/images/IMG_20250803_215727_jpg.rf.ed0c24ebb5a6e14c10a1c4d99dbf6530.jpg`
- `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg`
- Weak bridge to `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215810_jpg.rf.bf03fc4486612281e874dcbde3267f8b.jpg` (pHash=10, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg` (pHash=4, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215807_jpg.rf.949f4e1a94325c40ad658940123c68d8.jpg` (pHash=6, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_215808_jpg.rf.8a06ce738b95b31fc02705ce61b6d0ad.jpg` (pHash=8, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_215740_jpg.rf.758acdd6ad672a13af608eafffc805dd.jpg` (pHash=6, ORB inliers=0/0, same timestamp=False, conflict=none).
- **Strong graph bridge to review:** `test/images/IMG_20250803_215727_jpg.rf.ed0c24ebb5a6e14c10a1c4d99dbf6530.jpg` ↔ `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg`.

### AUTO-G0003 — 2 images; classes=Battery_Waste; original splits=test,valid; status=provisional_strong_component
- `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg`
- `valid/images/IMG_20250803_215740_jpg.rf.758acdd6ad672a13af608eafffc805dd.jpg`
- Weak bridge to `test/images/IMG_20250803_215727_jpg.rf.ed0c24ebb5a6e14c10a1c4d99dbf6530.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215807_jpg.rf.949f4e1a94325c40ad658940123c68d8.jpg` (pHash=4, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215810_jpg.rf.bf03fc4486612281e874dcbde3267f8b.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_215808_jpg.rf.8a06ce738b95b31fc02705ce61b6d0ad.jpg` (pHash=6, ORB inliers=0/0, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg` (pHash=4, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215807_jpg.rf.949f4e1a94325c40ad658940123c68d8.jpg` (pHash=6, ORB inliers=0/0, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_215808_jpg.rf.8a06ce738b95b31fc02705ce61b6d0ad.jpg` (pHash=8, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg` (pHash=6, ORB inliers=0/0, same timestamp=False, conflict=none).
- **Strong graph bridge to review:** `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg` ↔ `valid/images/IMG_20250803_215740_jpg.rf.758acdd6ad672a13af608eafffc805dd.jpg`.

### AUTO-G0004 — 3 images; classes=Battery_Waste; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250803_215807_jpg.rf.949f4e1a94325c40ad658940123c68d8.jpg`
- `test/images/IMG_20250803_215810_jpg.rf.bf03fc4486612281e874dcbde3267f8b.jpg`
- `train/images/IMG_20250803_215808_jpg.rf.8a06ce738b95b31fc02705ce61b6d0ad.jpg`
- Weak bridge to `test/images/IMG_20250803_215727_jpg.rf.ed0c24ebb5a6e14c10a1c4d99dbf6530.jpg` (pHash=10, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg` (pHash=4, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg` (pHash=6, ORB inliers=0/0, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg` (pHash=6, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_215740_jpg.rf.758acdd6ad672a13af608eafffc805dd.jpg` (pHash=6, ORB inliers=0/0, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_215808_jpg.rf.c547051ab6eefb114e85665da4474239.jpg` (pHash=36, ORB inliers=7/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_215808_jpg.rf.e0e84a12bb13158f05811974c97c5c67.jpg` (pHash=24, ORB inliers=121/123, same timestamp=True, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg` (pHash=8, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_215740_jpg.rf.758acdd6ad672a13af608eafffc805dd.jpg` (pHash=8, ORB inliers=0/3, same timestamp=False, conflict=none).

### AUTO-G0005 — 3 images; classes=Mobile; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg`
- `train/images/IMG_20250803_220022_jpg.rf.dc0fdee22f504920709e6321bf117c22.jpg`
- `valid/images/IMG_20250803_220024_jpg.rf.2fc166ace133b8664434156a034ba786.jpg`
- Weak bridge to `train/images/IMG_20250803_220022_jpg.rf.1b9bf10def62ebe2424913192f6033b6.jpg` (pHash=26, ORB inliers=102/106, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220022_jpg.rf.2cd6a4959481bac2347cd2b2f4dc6d61.jpg` (pHash=30, ORB inliers=142/152, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg` ↔ `train/images/IMG_20250803_220022_jpg.rf.dc0fdee22f504920709e6321bf117c22.jpg`.
- **Strong graph bridge to review:** `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg` ↔ `valid/images/IMG_20250803_220024_jpg.rf.2fc166ace133b8664434156a034ba786.jpg`.

### AUTO-G0006 — 5 images; classes=Mobile; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250803_220034_jpg.rf.ff0c9e529ddd3dfcdb4934dc21b66ca1.jpg`
- `test/images/IMG_20250803_220038_jpg.rf.7649ac00470edad7f06ef522f2ffa0ae.jpg`
- `train/images/IMG_20250803_220030_jpg.rf.1f79ea661a57dc1bd9c74db29adb4ec2.jpg`
- `valid/images/IMG_20250803_220031_jpg.rf.980993866f095141d958d78983b61f11.jpg`
- `valid/images/IMG_20250803_220037_jpg.rf.a1e1878bef855197a7260cca88694e51.jpg`
- Weak bridge to `train/images/IMG_20250803_220030_jpg.rf.760f092cef34c6d36f6e2aed8301671b.jpg` (pHash=34, ORB inliers=254/268, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220030_jpg.rf.bc24b1cc81bf8a1602888ab65ab72e10.jpg` (pHash=22, ORB inliers=276/283, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `test/images/IMG_20250803_220034_jpg.rf.ff0c9e529ddd3dfcdb4934dc21b66ca1.jpg` ↔ `train/images/IMG_20250803_220030_jpg.rf.1f79ea661a57dc1bd9c74db29adb4ec2.jpg`.
- **Strong graph bridge to review:** `test/images/IMG_20250803_220034_jpg.rf.ff0c9e529ddd3dfcdb4934dc21b66ca1.jpg` ↔ `valid/images/IMG_20250803_220037_jpg.rf.a1e1878bef855197a7260cca88694e51.jpg`.
- **Strong graph bridge to review:** `test/images/IMG_20250803_220038_jpg.rf.7649ac00470edad7f06ef522f2ffa0ae.jpg` ↔ `valid/images/IMG_20250803_220037_jpg.rf.a1e1878bef855197a7260cca88694e51.jpg`.
- **Strong graph bridge to review:** `train/images/IMG_20250803_220030_jpg.rf.1f79ea661a57dc1bd9c74db29adb4ec2.jpg` ↔ `valid/images/IMG_20250803_220031_jpg.rf.980993866f095141d958d78983b61f11.jpg`.

### AUTO-G0007 — 3 images; classes=Mobile; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250803_220045_jpg.rf.337135edaed79d430b95b610ec32c0db.jpg`
- `train/images/IMG_20250803_220046_jpg.rf.593a44abf20cee25f06845f455285742.jpg`
- `train/images/IMG_20250803_220047_jpg.rf.f69036399cb844d4fa35545cbe8bf18c.jpg`
- Weak bridge to `train/images/IMG_20250803_220046_jpg.rf.3a0decfaa0e347e8daf2186f2dca2af1.jpg` (pHash=22, ORB inliers=175/180, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220046_jpg.rf.e75fd5549c943e12ae1d5c9f747e2c80.jpg` (pHash=32, ORB inliers=176/194, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220047_jpg.rf.13ad331e1f469809c4b1ba6f0748d0cf.jpg` (pHash=24, ORB inliers=5/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_220047_jpg.rf.dcce9a993294f06fa6c2078e37f74309.jpg` (pHash=26, ORB inliers=4/10, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).

### AUTO-G0008 — 3 images; classes=Mobile; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250803_220050_jpg.rf.440e59bfd126dbb87848a17d6ef10ad8.jpg`
- `test/images/IMG_20250803_220051_jpg.rf.39ebee3d80d991a2fd3bd1e036ae9b05.jpg`
- `train/images/IMG_20250803_220053_jpg.rf.143f84f401009a4e0c09c5b7c33c1ea3.jpg`
- Weak bridge to `train/images/IMG_20250803_220053_jpg.rf.2aaa4021739cfba02e1bb2e1696ac316.jpg` (pHash=24, ORB inliers=163/182, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220053_jpg.rf.56719d3db139b1ca520a8b54447c4abd.jpg` (pHash=18, ORB inliers=269/275, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `test/images/IMG_20250803_220050_jpg.rf.440e59bfd126dbb87848a17d6ef10ad8.jpg` ↔ `test/images/IMG_20250803_220051_jpg.rf.39ebee3d80d991a2fd3bd1e036ae9b05.jpg`.
- **Strong graph bridge to review:** `test/images/IMG_20250803_220051_jpg.rf.39ebee3d80d991a2fd3bd1e036ae9b05.jpg` ↔ `train/images/IMG_20250803_220053_jpg.rf.143f84f401009a4e0c09c5b7c33c1ea3.jpg`.

### AUTO-G0009 — 2 images; classes=Mobile; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250803_220138_jpg.rf.c32f7dc03280433f7176d3be9f2d22a3.jpg`
- `train/images/IMG_20250803_220139_jpg.rf.36b05a61cd0dc1dc31f88bba93e1bb4a.jpg`
- Weak bridge to `train/images/IMG_20250803_220139_jpg.rf.b251b658131e5aa6fd0b7b709752795a.jpg` (pHash=34, ORB inliers=96/99, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220139_jpg.rf.bd9b54c63814502bd5ac48c1d34207ce.jpg` (pHash=32, ORB inliers=11/14, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `test/images/IMG_20250803_220138_jpg.rf.c32f7dc03280433f7176d3be9f2d22a3.jpg` ↔ `train/images/IMG_20250803_220139_jpg.rf.36b05a61cd0dc1dc31f88bba93e1bb4a.jpg`.

### AUTO-G0011 — 3 images; classes=Mobile; original splits=test,valid; status=provisional_strong_component
- `test/images/IMG_20250803_220146_jpg.rf.c669407739cff140d229870182988583.jpg`
- `valid/images/IMG_20250803_220148_jpg.rf.a4e62c59a53971b3d4a9abbcf8b459b1.jpg`
- `valid/images/IMG_20250803_220149_jpg.rf.ea36148ba2419eaf41dc5822d2e5d88a.jpg`

### AUTO-G0012 — 3 images; classes=Mobile; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250803_220228_jpg.rf.8e94e7ce7153325b8e181e6a1d331ef0.jpg`
- `train/images/IMG_20250803_220226_jpg.rf.ff911118579e3a71da019d386eb1d7eb.jpg`
- `valid/images/IMG_20250803_220227_jpg.rf.c42373ef7c64d08b36804cc7fcad4d15.jpg`
- Weak bridge to `train/images/IMG_20250805_203902_jpg.rf.a23fe3b1c30257e8947ef9b1f48a0746.jpg` (pHash=10, ORB inliers=7/8, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_220244_jpg.rf.25e694fad0069d6b3720bcfeff342f09.jpg` (pHash=10, ORB inliers=4/4, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250803_220244_jpg.rf.25e694fad0069d6b3720bcfeff342f09.jpg` (pHash=10, ORB inliers=6/13, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220226_jpg.rf.b4392a8eca529f4ed6fed45e99eeeb5e.jpg` (pHash=34, ORB inliers=6/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_220226_jpg.rf.eb9c0141249018f67e7bdc8bf241a8a2.jpg` (pHash=20, ORB inliers=199/201, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220243_jpg.rf.c17d27aa58b11aa0475d5b41c8ddbd97.jpg` (pHash=10, ORB inliers=4/4, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203902_jpg.rf.a23fe3b1c30257e8947ef9b1f48a0746.jpg` (pHash=10, ORB inliers=0/5, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220243_jpg.rf.c17d27aa58b11aa0475d5b41c8ddbd97.jpg` (pHash=10, ORB inliers=11/20, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203902_jpg.rf.a23fe3b1c30257e8947ef9b1f48a0746.jpg` (pHash=10, ORB inliers=0/0, same timestamp=False, conflict=none).

### AUTO-G0013 — 3 images; classes=Mobile; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250803_220244_jpg.rf.25e694fad0069d6b3720bcfeff342f09.jpg`
- `train/images/IMG_20250803_220242_jpg.rf.c47cf376f90ec8109c0115c6ef00398e.jpg`
- `train/images/IMG_20250803_220243_jpg.rf.c17d27aa58b11aa0475d5b41c8ddbd97.jpg`
- Weak bridge to `train/images/IMG_20250803_220226_jpg.rf.ff911118579e3a71da019d386eb1d7eb.jpg` (pHash=10, ORB inliers=4/4, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_220227_jpg.rf.c42373ef7c64d08b36804cc7fcad4d15.jpg` (pHash=10, ORB inliers=6/13, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220226_jpg.rf.ff911118579e3a71da019d386eb1d7eb.jpg` (pHash=10, ORB inliers=4/4, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220242_jpg.rf.7522feb05c688af1d5d44ab984f05a1c.jpg` (pHash=16, ORB inliers=128/128, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220242_jpg.rf.8d0d22aab4dcc12eed7a126bce99cd55.jpg` (pHash=24, ORB inliers=90/92, same timestamp=True, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_215638_jpg.rf.ba4f3127b341131ea269136e3d2b124e.jpg` (pHash=8, ORB inliers=0/3, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `valid/images/IMG_20250803_215639_jpg.rf.fbe999007ecf4fd037195142c1702267.jpg` (pHash=10, ORB inliers=5/6, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250803_220243_jpg.rf.0b0201bb99cd8531e9b7ca91ed720816.jpg` (pHash=30, ORB inliers=2/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_220243_jpg.rf.2b74ac43ae204204eedb898b9845ea08.jpg` (pHash=28, ORB inliers=7/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `valid/images/IMG_20250803_220227_jpg.rf.c42373ef7c64d08b36804cc7fcad4d15.jpg` (pHash=10, ORB inliers=11/20, same timestamp=False, conflict=none).

### AUTO-G0014 — 4 images; classes=Mobile; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250803_220254_jpg.rf.11410f220a6391f2c4b55aa7acabf966.jpg`
- `test/images/IMG_20250803_220256_jpg.rf.3b9f89aa439c4e4d73ed04ed18e16668.jpg`
- `test/images/IMG_20250803_220257_jpg.rf.8553c912c825e21ed3ec17bc2f25ff9a.jpg`
- `train/images/IMG_20250803_220255_jpg.rf.3511615b828854235bed04fcf894a5f5.jpg`
- Weak bridge to `train/images/IMG_20250803_220255_jpg.rf.0bb096336a37b9aa702cc0e8a429e701.jpg` (pHash=36, ORB inliers=8/17, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220255_jpg.rf.4848154bd5d1a0e715eeb042fbeb929b.jpg` (pHash=34, ORB inliers=232/244, same timestamp=True, conflict=none).

### AUTO-G0015 — 3 images; classes=Battery_Waste; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250803_220311_jpg.rf.72c9015fb578049a797e0d93bd65aee3.jpg`
- `train/images/IMG_20250803_220312_jpg.rf.07000a53f58fd8b7ca0a0d97613d25f3.jpg`
- `valid/images/IMG_20250803_220313_jpg.rf.64f586e62091d23ee0746e2a666392b5.jpg`
- Weak bridge to `train/images/IMG_20250803_220312_jpg.rf.8fbb3d0ef325d176bd1eb49c8056621d.jpg` (pHash=24, ORB inliers=3/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_220312_jpg.rf.c5b01155df012f0543c0ea06f87af29e.jpg` (pHash=26, ORB inliers=0/3, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).

### AUTO-G0016 — 6 images; classes=Battery_Waste; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250803_220320_jpg.rf.10f96e152fbca7bb72a58524897b976a.jpg`
- `test/images/IMG_20250803_220323_1_jpg.rf.573457ae657057432e59564c621a62b2.jpg`
- `train/images/IMG_20250803_220321_jpg.rf.7d603d397b0d570f006ad3ed948b1fd8.jpg`
- `valid/images/IMG_20250803_220320_1_jpg.rf.4004befc73e0f7a956fcd25d6de4037b.jpg`
- `valid/images/IMG_20250803_220322_jpg.rf.4ad5d2a122e2ec7a03c34971a4269fd9.jpg`
- `valid/images/IMG_20250803_220323_jpg.rf.dae4ff08759027feadbee436a4098da3.jpg`
- Weak bridge to `train/images/IMG_20250803_220321_jpg.rf.18be45e7ac54d6c3c9f1bac80e64585e.jpg` (pHash=28, ORB inliers=9/11, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220321_jpg.rf.25adc2d462f062f4163541ffbcb1da33.jpg` (pHash=32, ORB inliers=162/167, same timestamp=True, conflict=none).

### AUTO-G0017 — 8 images; classes=Mobile; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250803_220351_jpg.rf.450e818f2d954bda365821a90b9c2a70.jpg`
- `train/images/IMG_20250803_220350_jpg.rf.51437498308c9763e0717636cb26c374.jpg`
- `train/images/IMG_20250803_220352_jpg.rf.af73197979f1b112320f3f06a0058c12.jpg`
- `train/images/IMG_20250803_220353_jpg.rf.1a768a733a3016bb320de04418c64568.jpg`
- `train/images/IMG_20250803_220353_jpg.rf.38ba2d3b1f3eadc387125573185c1d16.jpg`
- `valid/images/IMG_20250803_220348_jpg.rf.2f88a5bd0de4e00950763150d344a288.jpg`
- `valid/images/IMG_20250803_220349_1_jpg.rf.c2488b27ffdb7b551091ec1aff857cce.jpg`
- `valid/images/IMG_20250803_220349_jpg.rf.9a2cadc69014490f60e59e1826cdc947.jpg`
- Weak bridge to `train/images/IMG_20250803_220350_jpg.rf.1f5aa0f107ac49e368b6c16523ac1c14.jpg` (pHash=24, ORB inliers=152/160, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220350_jpg.rf.cc9680936d403e013dd1e8a4863afe8d.jpg` (pHash=24, ORB inliers=210/216, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220352_jpg.rf.8d5cd74618268102d6ee8e431e2dc659.jpg` (pHash=26, ORB inliers=152/157, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220352_jpg.rf.9962226a88efa23a6ca92f4e4337f01f.jpg` (pHash=24, ORB inliers=180/185, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220353_jpg.rf.38ba2d3b1f3eadc387125573185c1d16.jpg` (pHash=12, ORB inliers=236/244, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220353_jpg.rf.ce1bfbf5249837f379ded71315f8e5a0.jpg` (pHash=24, ORB inliers=6/13, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_220353_jpg.rf.ce1bfbf5249837f379ded71315f8e5a0.jpg` (pHash=28, ORB inliers=5/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- **Strong graph bridge to review:** `test/images/IMG_20250803_220351_jpg.rf.450e818f2d954bda365821a90b9c2a70.jpg` ↔ `train/images/IMG_20250803_220353_jpg.rf.1a768a733a3016bb320de04418c64568.jpg`.

### AUTO-G0018 — 3 images; classes=Mobile; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250805_203323_jpg.rf.a182df57cbdc73a0a473a3e488e1725f.jpg`
- `train/images/IMG_20250805_203321_jpg.rf.fda2603315b3ef543f6b3a2d3c796074.jpg`
- `valid/images/IMG_20250805_203322_jpg.rf.55299a747169161eadaecc211e423a31.jpg`
- Weak bridge to `train/images/IMG_20250805_203321_jpg.rf.5eee9af99b5e4d1d2a4ee31868b16180.jpg` (pHash=26, ORB inliers=0/1, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250805_203321_jpg.rf.d9ba5210eb68c3c23cb593d8a6d9c888.jpg` (pHash=28, ORB inliers=0/2, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).

### AUTO-G0019 — 3 images; classes=Mobile; original splits=test,valid; status=provisional_strong_component
- `test/images/IMG_20250805_203327_1_jpg.rf.fa4eadf62370e6bdea8c64d3f20085bf.jpg`
- `valid/images/IMG_20250805_203327_jpg.rf.32be60b70e6f9a50917a48881f5ab85a.jpg`
- `valid/images/IMG_20250805_203328_jpg.rf.593c94263769d9d875503af03e97d8f7.jpg`

### AUTO-G0022 — 4 images; classes=Mobile; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250805_203336_jpg.rf.ab9901e1027c275718c876cd3fd594e8.jpg`
- `test/images/IMG_20250805_203338_1_jpg.rf.bf2025683e05750526dc29fc829b8ac5.jpg`
- `train/images/IMG_20250805_203337_jpg.rf.619c09a827a0821443b40ad63fcaba9c.jpg`
- `train/images/IMG_20250805_203338_jpg.rf.68cadbff708a1a27b20e57a030073a53.jpg`
- Weak bridge to `train/images/IMG_20250805_203338_jpg.rf.29bc49ddc66d6aeb9a9b51d1b216f1ec.jpg` (pHash=26, ORB inliers=0/3, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250805_203338_jpg.rf.967c7c7b3f77ddf047ed91906ea0686d.jpg` (pHash=12, ORB inliers=255/258, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203337_jpg.rf.6ce280063c439dcff00dc47da971944f.jpg` (pHash=16, ORB inliers=225/225, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203337_jpg.rf.6da83fbcfaa0da8d3a9211eba95e434d.jpg` (pHash=14, ORB inliers=286/289, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203338_jpg.rf.29bc49ddc66d6aeb9a9b51d1b216f1ec.jpg` (pHash=26, ORB inliers=0/2, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250805_203338_jpg.rf.967c7c7b3f77ddf047ed91906ea0686d.jpg` (pHash=12, ORB inliers=277/277, same timestamp=True, conflict=none).

### AUTO-G0023 — 4 images; classes=Mobile; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250805_203805_jpg.rf.ecf4905797f5b1dbc5887402d335664a.jpg`
- `train/images/IMG_20250805_203804_jpg.rf.d0d25dd17b163cc60b1a269ea27346dd.jpg`
- `train/images/IMG_20250805_203806_jpg.rf.7fd019ac3a6efc21e4b37227c5bb73e1.jpg`
- `valid/images/IMG_20250805_203803_jpg.rf.46ed623c40ba0974562c6bbf6953d2cf.jpg`
- Weak bridge to `train/images/IMG_20250805_203804_jpg.rf.23acd615013748d7bdd3d547bb50935a.jpg` (pHash=24, ORB inliers=205/215, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203804_jpg.rf.2aa2f4ad307106ebef06541aae137975.jpg` (pHash=22, ORB inliers=256/267, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203806_jpg.rf.34c678b8361cf794b5af0e3b63a58a07.jpg` (pHash=26, ORB inliers=160/179, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203806_jpg.rf.a88c5e5ff1cddfe7235a919241c54687.jpg` (pHash=32, ORB inliers=197/206, same timestamp=True, conflict=none).

### AUTO-G0024 — 3 images; classes=Mobile; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250805_203845_jpg.rf.8c15db31b863733a22036fd715b6cc3a.jpg`
- `train/images/IMG_20250805_203846_jpg.rf.66d7b40dd70798c0e6a723d0bdc26269.jpg`
- `valid/images/IMG_20250805_203847_jpg.rf.0b7b578193561a5e7b7297568d1862c0.jpg`
- Weak bridge to `train/images/IMG_20250805_203846_jpg.rf.9e943104abcc04412073581826afec4d.jpg` (pHash=28, ORB inliers=30/36, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203846_jpg.rf.cfa5e7649e94c6cde734514ca4acd259.jpg` (pHash=28, ORB inliers=30/35, same timestamp=True, conflict=none).

### AUTO-G0026 — 2 images; classes=Mobile; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250805_203918_jpg.rf.0320c79bd3505204e3c7b6d35a1d716a.jpg`
- `train/images/IMG_20250805_203917_jpg.rf.41ede18eba5e4814a68b0143d4b637b7.jpg`
- Weak bridge to `train/images/IMG_20250805_203917_jpg.rf.3512fa1899ab4b236c8abd34a2e32b74.jpg` (pHash=34, ORB inliers=260/287, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203917_jpg.rf.cfecadfe8ec76d0786a7fb3ac71b33db.jpg` (pHash=38, ORB inliers=278/293, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `test/images/IMG_20250805_203918_jpg.rf.0320c79bd3505204e3c7b6d35a1d716a.jpg` ↔ `train/images/IMG_20250805_203917_jpg.rf.41ede18eba5e4814a68b0143d4b637b7.jpg`.

### AUTO-G0027 — 4 images; classes=Mobile; original splits=test,valid; status=provisional_strong_component
- `test/images/IMG_20250805_203950_jpg.rf.e20c8cf0ae6b59ee38b2a9c2dadbe0eb.jpg`
- `valid/images/IMG_20250805_203947_jpg.rf.d549f89d842b21852845f82bff9abb09.jpg`
- `valid/images/IMG_20250805_203948_jpg.rf.dbcb8f50ff9c3d8ab30e1018b3995fa7.jpg`
- `valid/images/IMG_20250805_203949_jpg.rf.c78fa0f75f1aea8e269f90a2c79abeaa.jpg`

### AUTO-G0029 — 5 images; classes=Battery_Waste; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250822_231124_jpg.rf.454eb5a36ebdb2ad770c60899ed1ff5d.jpg`
- `test/images/IMG_20250822_231126_jpg.rf.7712b4701c0e3c8ad2f513a37cac8107.jpg`
- `train/images/IMG_20250822_231125_1_jpg.rf.055929b00c16ff17c6d4a49b7cef58a9.jpg`
- `valid/images/IMG_20250822_231124_1_jpg.rf.15f6be87893a7f3e91d86d821962f40f.jpg`
- `valid/images/IMG_20250822_231125_jpg.rf.712cf5e8bb04d06029470cfebfd9db8a.jpg`
- Weak bridge to `train/images/IMG_20250822_231125_1_jpg.rf.24e5f0335754fa334350f28e5fc26691.jpg` (pHash=30, ORB inliers=237/247, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231125_1_jpg.rf.51f8307cc11b5cddecd0ca3e02555e70.jpg` (pHash=24, ORB inliers=172/174, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231125_1_jpg.rf.24e5f0335754fa334350f28e5fc26691.jpg` (pHash=30, ORB inliers=225/239, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231125_1_jpg.rf.51f8307cc11b5cddecd0ca3e02555e70.jpg` (pHash=24, ORB inliers=179/182, same timestamp=True, conflict=none).

### AUTO-G0030 — 5 images; classes=Battery_Waste; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250822_231151_jpg.rf.2607ab2dc8acaf81e39a0adf8e0950f7.jpg`
- `test/images/IMG_20250822_231152_jpg.rf.e62543112ad7bd0d4593d533a2e3f231.jpg`
- `train/images/IMG_20250822_231153_1_jpg.rf.055d65f1f0714e9fda6d890f79bc15ab.jpg`
- `train/images/IMG_20250822_231155_jpg.rf.656e96d585807828d9bd58402edf9500.jpg`
- `valid/images/IMG_20250822_231153_jpg.rf.c0581636936ffd3f551a62ae4e3aadb3.jpg`
- Weak bridge to `train/images/IMG_20250822_231153_1_jpg.rf.0ccc4241354b45d12d7828899bc92f05.jpg` (pHash=30, ORB inliers=77/83, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231153_1_jpg.rf.f19abd752876b09f640fce636287ab42.jpg` (pHash=34, ORB inliers=7/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231153_1_jpg.rf.0ccc4241354b45d12d7828899bc92f05.jpg` (pHash=28, ORB inliers=91/99, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231153_1_jpg.rf.f19abd752876b09f640fce636287ab42.jpg` (pHash=34, ORB inliers=8/13, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231155_jpg.rf.504426a470f2b291095627bb367e7dad.jpg` (pHash=32, ORB inliers=58/65, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231155_jpg.rf.af8d1320d7a24080846ef501f2abf92c.jpg` (pHash=30, ORB inliers=116/130, same timestamp=True, conflict=none).

### AUTO-G0031 — 6 images; classes=Battery_Waste; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250822_231209_1_jpg.rf.fcf651145e2d95b81efe4e9904f3cfed.jpg`
- `test/images/IMG_20250822_231210_jpg.rf.3e928d15c853583abf8b89eff296af81.jpg`
- `test/images/IMG_20250822_231211_jpg.rf.29439c714ba00a43492ccf5b2673e53f.jpg`
- `train/images/IMG_20250822_231209_jpg.rf.4c130981a9fbbd3987e2dd770795972d.jpg`
- `train/images/IMG_20250822_231210_1_jpg.rf.bf5f830d9485eb8d3bf5ca84d009a28d.jpg`
- `valid/images/IMG_20250822_231211_1_jpg.rf.81b7eb1b4bb750b87e50a18ac5a90583.jpg`
- Weak bridge to `train/images/IMG_20250822_231209_jpg.rf.5fbcecd2f828c4d73be632426a997488.jpg` (pHash=32, ORB inliers=8/18, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231209_jpg.rf.71062767ff841e54501edf3ddc7a9c7a.jpg` (pHash=26, ORB inliers=6/11, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231210_1_jpg.rf.7e7d56093797517d49417300b1eceb44.jpg` (pHash=34, ORB inliers=123/137, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231210_1_jpg.rf.b643e8ffe7294d17bcf888a766981b1a.jpg` (pHash=32, ORB inliers=6/11, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231209_jpg.rf.5fbcecd2f828c4d73be632426a997488.jpg` (pHash=36, ORB inliers=6/15, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231209_jpg.rf.71062767ff841e54501edf3ddc7a9c7a.jpg` (pHash=24, ORB inliers=7/15, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231210_1_jpg.rf.7e7d56093797517d49417300b1eceb44.jpg` (pHash=34, ORB inliers=116/132, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231210_1_jpg.rf.b643e8ffe7294d17bcf888a766981b1a.jpg` (pHash=32, ORB inliers=8/16, same timestamp=True, conflict=none).

### AUTO-G0032 — 6 images; classes=Battery_Waste; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250822_231221_jpg.rf.f1b09c45232ce7f53d9929c1a6024cc0.jpg`
- `train/images/IMG_20250822_231219_jpg.rf.03d08967efba3804f23ba15aac3d4cca.jpg`
- `train/images/IMG_20250822_231220_jpg.rf.847cf565415d2346e0fabc16a3fb1f28.jpg`
- `train/images/IMG_20250822_231222_jpg.rf.44ed716c90773067e2cf27b0d901024b.jpg`
- `valid/images/IMG_20250822_231220_1_jpg.rf.32d046cd028697cf3c99f250db87866d.jpg`
- `valid/images/IMG_20250822_231221_1_jpg.rf.187ee67700b833c74c2c876f9f8165c7.jpg`
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.e7e6036503debb760b34e87faa28dc66.jpg` (pHash=10, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250822_231228_1_jpg.rf.8374d1d76854c1115c33fd7b7bedb107.jpg` (pHash=10, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231219_jpg.rf.152f0425291ed55f66ec4bea49524c43.jpg` (pHash=24, ORB inliers=0/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231219_jpg.rf.392d276fdfc40327967ce3017eaf01ac.jpg` (pHash=22, ORB inliers=136/141, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231220_jpg.rf.2b36d46a7e22d101be3c9448b72ef6f1.jpg` (pHash=28, ORB inliers=4/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231220_jpg.rf.2b36d46a7e22d101be3c9448b72ef6f1.jpg` (pHash=28, ORB inliers=6/10, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231220_jpg.rf.9c02b05d7b7b49ba3c228d2dc9bf9b7e.jpg` (pHash=26, ORB inliers=5/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.e7e6036503debb760b34e87faa28dc66.jpg` (pHash=8, ORB inliers=0/6, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231220_jpg.rf.9c02b05d7b7b49ba3c228d2dc9bf9b7e.jpg` (pHash=28, ORB inliers=5/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231222_jpg.rf.6da281988f9a7ad6de6a237c773340b0.jpg` (pHash=28, ORB inliers=0/2, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231222_jpg.rf.a82c3365421ee2a0e83bc207acc9927d.jpg` (pHash=28, ORB inliers=4/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.e7e6036503debb760b34e87faa28dc66.jpg` (pHash=10, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.e7e6036503debb760b34e87faa28dc66.jpg` (pHash=8, ORB inliers=6/6, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.e7e6036503debb760b34e87faa28dc66.jpg` (pHash=10, ORB inliers=5/5, same timestamp=False, conflict=none).

### AUTO-G0033 — 6 images; classes=Battery_Waste; original splits=test,train; status=provisional_strong_component
- `test/images/IMG_20250822_231228_1_jpg.rf.8374d1d76854c1115c33fd7b7bedb107.jpg`
- `test/images/IMG_20250822_231228_jpg.rf.5878b0c9eb72ca9d1cdcfc596fc61234.jpg`
- `train/images/IMG_20250822_231226_jpg.rf.a384c4024b55ba7bf2eda542344eef69.jpg`
- `train/images/IMG_20250822_231227_jpg.rf.8ff7f770de4f04e022e8f12c213838c1.jpg`
- `train/images/IMG_20250822_231227_jpg.rf.e13a9b5400f8a9c58d97fea44a360ba9.jpg`
- `train/images/IMG_20250822_231229_jpg.rf.e7e6036503debb760b34e87faa28dc66.jpg`
- Weak bridge to `test/images/IMG_20250822_231221_jpg.rf.f1b09c45232ce7f53d9929c1a6024cc0.jpg` (pHash=10, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231220_jpg.rf.847cf565415d2346e0fabc16a3fb1f28.jpg` (pHash=10, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231220_jpg.rf.847cf565415d2346e0fabc16a3fb1f28.jpg` (pHash=8, ORB inliers=0/6, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231222_jpg.rf.44ed716c90773067e2cf27b0d901024b.jpg` (pHash=10, ORB inliers=0/3, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231226_jpg.rf.af949f552ee40abb9ba0d2ea22afdb16.jpg` (pHash=32, ORB inliers=127/138, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231226_jpg.rf.f5c80448ec1e3edadd7fe58129065dd8.jpg` (pHash=30, ORB inliers=3/11, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231227_jpg.rf.03756de636cf7cc1ebd60892c444e82f.jpg` (pHash=14, ORB inliers=102/110, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231227_jpg.rf.03756de636cf7cc1ebd60892c444e82f.jpg` (pHash=18, ORB inliers=186/197, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.74aa09b68e143f0d05c0449cf7200946.jpg` (pHash=34, ORB inliers=106/132, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.9286a8bea2098e273163b8a58657f4c0.jpg` (pHash=28, ORB inliers=4/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `valid/images/IMG_20250822_231220_1_jpg.rf.32d046cd028697cf3c99f250db87866d.jpg` (pHash=8, ORB inliers=6/6, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231221_1_jpg.rf.187ee67700b833c74c2c876f9f8165c7.jpg` (pHash=10, ORB inliers=5/5, same timestamp=False, conflict=none).

### AUTO-G0034 — 8 images; classes=Battery_Waste; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250822_231238_jpg.rf.3c4cd2453b779bbf5224c35672da6c9d.jpg`
- `test/images/IMG_20250822_231239_jpg.rf.a708ada26d463e1fc8125634fc78cbf5.jpg`
- `test/images/IMG_20250822_231241_1_jpg.rf.08f4254c1a6be413d72d48bc233c8175.jpg`
- `train/images/IMG_20250822_231239_1_jpg.rf.d7c89c449cafe2a0f21c3245793671f9.jpg`
- `train/images/IMG_20250822_231240_1_jpg.rf.4547dfb9464eb55f94020eafbe21c13f.jpg`
- `valid/images/IMG_20250822_231238_1_jpg.rf.6b34d7600d1352cab6f4fa768e169697.jpg`
- `valid/images/IMG_20250822_231240_jpg.rf.0be8aa7a50a7759b7663c4d15fee22ee.jpg`
- `valid/images/IMG_20250822_231241_jpg.rf.d6ccb4191c62c428828b1cebbd503123.jpg`
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.9f96d6100cf9c4dab496702706a56721.jpg` (pHash=8, ORB inliers=7/7, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231254_jpg.rf.b3dfdd482b858b6fb25f930221870e93.jpg` (pHash=8, ORB inliers=7/9, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.1b55959316611efa584bdc9a073bc60b.jpg` (pHash=30, ORB inliers=0/3, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.ff7c20ffa1b97e6f85e4860b6c2eefbc.jpg` (pHash=28, ORB inliers=151/167, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231255_jpg.rf.36ef9257a4993f8d263c5dd4cb7bf7f5.jpg` (pHash=10, ORB inliers=4/4, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.9f96d6100cf9c4dab496702706a56721.jpg` (pHash=8, ORB inliers=6/6, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231254_jpg.rf.b3dfdd482b858b6fb25f930221870e93.jpg` (pHash=8, ORB inliers=8/10, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231254_jpg.rf.b3dfdd482b858b6fb25f930221870e93.jpg` (pHash=10, ORB inliers=5/5, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250822_231256_1_jpg.rf.aea99bc61d2369c19f56241e3f0876cf.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.1b55959316611efa584bdc9a073bc60b.jpg` (pHash=30, ORB inliers=0/0, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.ff7c20ffa1b97e6f85e4860b6c2eefbc.jpg` (pHash=28, ORB inliers=131/139, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231255_jpg.rf.36ef9257a4993f8d263c5dd4cb7bf7f5.jpg` (pHash=10, ORB inliers=5/6, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.9f96d6100cf9c4dab496702706a56721.jpg` (pHash=8, ORB inliers=6/6, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231254_jpg.rf.b3dfdd482b858b6fb25f930221870e93.jpg` (pHash=6, ORB inliers=0/5, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231240_1_jpg.rf.29e3db85d081590d2c545b6a25bf448f.jpg` (pHash=28, ORB inliers=0/2, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231240_1_jpg.rf.29e3db85d081590d2c545b6a25bf448f.jpg` (pHash=30, ORB inliers=0/3, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231240_1_jpg.rf.c599e12109d4ea69fd2c676c98f032ff.jpg` (pHash=28, ORB inliers=86/86, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231240_1_jpg.rf.c599e12109d4ea69fd2c676c98f032ff.jpg` (pHash=28, ORB inliers=81/86, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.9f96d6100cf9c4dab496702706a56721.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.9f96d6100cf9c4dab496702706a56721.jpg` (pHash=10, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231254_jpg.rf.b3dfdd482b858b6fb25f930221870e93.jpg` (pHash=8, ORB inliers=0/6, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231254_jpg.rf.b3dfdd482b858b6fb25f930221870e93.jpg` (pHash=10, ORB inliers=8/14, same timestamp=False, conflict=none).

### AUTO-G0035 — 6 images; classes=Battery_Waste; original splits=test,train,valid; status=provisional_strong_component
- `test/images/IMG_20250822_231256_1_jpg.rf.aea99bc61d2369c19f56241e3f0876cf.jpg`
- `train/images/IMG_20250822_231255_1_jpg.rf.5f530548951cb35377a1987db865ecc0.jpg`
- `train/images/IMG_20250822_231255_jpg.rf.36ef9257a4993f8d263c5dd4cb7bf7f5.jpg`
- `train/images/IMG_20250822_231256_jpg.rf.eb79bcac78e20f8fb24d570dc248ba25.jpg`
- `train/images/IMG_20250822_231257_jpg.rf.9f96d6100cf9c4dab496702706a56721.jpg`
- `valid/images/IMG_20250822_231254_jpg.rf.b3dfdd482b858b6fb25f930221870e93.jpg`
- Weak bridge to `test/images/IMG_20250822_231238_jpg.rf.3c4cd2453b779bbf5224c35672da6c9d.jpg` (pHash=8, ORB inliers=7/7, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250822_231238_jpg.rf.3c4cd2453b779bbf5224c35672da6c9d.jpg` (pHash=8, ORB inliers=7/9, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250822_231239_jpg.rf.a708ada26d463e1fc8125634fc78cbf5.jpg` (pHash=10, ORB inliers=4/4, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250822_231239_jpg.rf.a708ada26d463e1fc8125634fc78cbf5.jpg` (pHash=8, ORB inliers=6/6, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250822_231239_jpg.rf.a708ada26d463e1fc8125634fc78cbf5.jpg` (pHash=8, ORB inliers=8/10, same timestamp=False, conflict=none).
- Weak bridge to `test/images/IMG_20250822_231241_1_jpg.rf.08f4254c1a6be413d72d48bc233c8175.jpg` (pHash=10, ORB inliers=5/5, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231256_jpg.rf.77db81774004ef79c28d543c812b7e54.jpg` (pHash=30, ORB inliers=8/10, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231256_jpg.rf.b5c48e1da6f07d1a4053ac81da1a3b2c.jpg` (pHash=24, ORB inliers=191/193, same timestamp=True, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231240_jpg.rf.0be8aa7a50a7759b7663c4d15fee22ee.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.d7c89c449cafe2a0f21c3245793671f9.jpg` (pHash=10, ORB inliers=5/6, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.d7c89c449cafe2a0f21c3245793671f9.jpg` (pHash=8, ORB inliers=6/6, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.d7c89c449cafe2a0f21c3245793671f9.jpg` (pHash=6, ORB inliers=0/5, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231255_1_jpg.rf.e32021e200cdfed4065135f35ee4bca8.jpg` (pHash=36, ORB inliers=8/9, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231255_1_jpg.rf.e3ebfec65816d16aff257a0db04a3372.jpg` (pHash=26, ORB inliers=7/7, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231255_jpg.rf.7c431d715c6666390961f2fd29ff327e.jpg` (pHash=32, ORB inliers=6/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231255_jpg.rf.874ad060677087d05bc1987392adbfbb.jpg` (pHash=32, ORB inliers=4/4, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231255_1_jpg.rf.e32021e200cdfed4065135f35ee4bca8.jpg` (pHash=32, ORB inliers=0/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231255_1_jpg.rf.e3ebfec65816d16aff257a0db04a3372.jpg` (pHash=24, ORB inliers=7/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231255_jpg.rf.7c431d715c6666390961f2fd29ff327e.jpg` (pHash=28, ORB inliers=7/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231255_jpg.rf.874ad060677087d05bc1987392adbfbb.jpg` (pHash=28, ORB inliers=0/0, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231256_jpg.rf.77db81774004ef79c28d543c812b7e54.jpg` (pHash=28, ORB inliers=5/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231256_jpg.rf.b5c48e1da6f07d1a4053ac81da1a3b2c.jpg` (pHash=22, ORB inliers=224/231, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.2db20ee5a5e4fda240d26419f21c7ee8.jpg` (pHash=28, ORB inliers=6/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.faf456eebb5f026fde69eb667b9ae98e.jpg` (pHash=30, ORB inliers=155/161, same timestamp=True, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231238_1_jpg.rf.6b34d7600d1352cab6f4fa768e169697.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231241_jpg.rf.d6ccb4191c62c428828b1cebbd503123.jpg` (pHash=10, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231238_1_jpg.rf.6b34d7600d1352cab6f4fa768e169697.jpg` (pHash=8, ORB inliers=0/6, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231241_jpg.rf.d6ccb4191c62c428828b1cebbd503123.jpg` (pHash=10, ORB inliers=8/14, same timestamp=False, conflict=none).

### AUTO-G0037 — 4 images; classes=Battery_Waste; original splits=train,valid; status=provisional_strong_component
- `train/images/IMG_20250803_215648_jpg.rf.c339d9561861e32aec3649527f4423f7.jpg`
- `valid/images/IMG_20250803_215638_jpg.rf.ba4f3127b341131ea269136e3d2b124e.jpg`
- `valid/images/IMG_20250803_215639_jpg.rf.fbe999007ecf4fd037195142c1702267.jpg`
- `valid/images/IMG_20250803_215645_jpg.rf.e82c2404db3f48100de5904104aa16a5.jpg`
- Weak bridge to `train/images/IMG_20250803_215648_jpg.rf.76ccb3808e396afdab350d5bc97f6366.jpg` (pHash=30, ORB inliers=217/232, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_215648_jpg.rf.e06da5e80cf6d6815419245cec540057.jpg` (pHash=32, ORB inliers=176/196, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220242_jpg.rf.c47cf376f90ec8109c0115c6ef00398e.jpg` (pHash=8, ORB inliers=0/3, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250803_220242_jpg.rf.c47cf376f90ec8109c0115c6ef00398e.jpg` (pHash=10, ORB inliers=5/6, same timestamp=False, conflict=different_original_class_labels).

### AUTO-G0039 — 2 images; classes=Battery_Waste; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250803_215701_jpg.rf.352a524931e1295dac040a7f6381d4f1.jpg`
- `train/images/IMG_20250803_215704_jpg.rf.9e7e0d24e11f27af943cdf2c6ea3daa4.jpg`
- Weak bridge to `train/images/IMG_20250803_215701_jpg.rf.953215ef1f4ce660a9ce216ed10cd5fe.jpg` (pHash=32, ORB inliers=54/59, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_215701_jpg.rf.e582db8c0d1f0560b6d54cc15f102097.jpg` (pHash=28, ORB inliers=4/7, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_215704_jpg.rf.0d941aee39572d80f929eb96b4640aaa.jpg` (pHash=30, ORB inliers=76/81, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_215704_jpg.rf.f00cf5b73c3af9c585d0df618a428c39.jpg` (pHash=28, ORB inliers=0/0, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- **Strong graph bridge to review:** `train/images/IMG_20250803_215701_jpg.rf.352a524931e1295dac040a7f6381d4f1.jpg` ↔ `train/images/IMG_20250803_215704_jpg.rf.9e7e0d24e11f27af943cdf2c6ea3daa4.jpg`.

### AUTO-G0045 — 2 images; classes=Battery_Waste; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250803_215753_jpg.rf.6b4e8f027d529c142a005c8042a782a1.jpg`
- `train/images/IMG_20250803_215755_jpg.rf.86629ff209a27b91f91ada50042844d8.jpg`
- Weak bridge to `test/images/IMG_20250803_220141_jpg.rf.addfdb758aea6542ed458ec7e715f306.jpg` (pHash=10, ORB inliers=0/0, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250803_215753_jpg.rf.665f84f766199c759bdabee414daf5ad.jpg` (pHash=26, ORB inliers=5/7, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_215753_jpg.rf.fcef9d6acea49b62724194527689a086.jpg` (pHash=30, ORB inliers=3/12, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `valid/images/IMG_20250803_220126_jpg.rf.16a1c934ac06cbc5d2ee0f8b4c02e96b.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `valid/images/IMG_20250803_220127_jpg.rf.29454ccaf0bd8058e12a5925d3619982.jpg` (pHash=10, ORB inliers=0/3, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250803_215755_jpg.rf.02fea5708529fa5762186f65ab447c9e.jpg` (pHash=34, ORB inliers=4/7, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_215755_jpg.rf.ee180e7cf5243b0f31a037d431aac714.jpg` (pHash=36, ORB inliers=6/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `valid/images/IMG_20250803_220126_jpg.rf.16a1c934ac06cbc5d2ee0f8b4c02e96b.jpg` (pHash=8, ORB inliers=0/3, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `valid/images/IMG_20250803_220127_jpg.rf.29454ccaf0bd8058e12a5925d3619982.jpg` (pHash=8, ORB inliers=0/5, same timestamp=False, conflict=different_original_class_labels).
- **Strong graph bridge to review:** `train/images/IMG_20250803_215753_jpg.rf.6b4e8f027d529c142a005c8042a782a1.jpg` ↔ `train/images/IMG_20250803_215755_jpg.rf.86629ff209a27b91f91ada50042844d8.jpg`.

### AUTO-G0046 — 2 images; classes=Battery_Waste; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250803_215753_jpg.rf.fcef9d6acea49b62724194527689a086.jpg`
- `train/images/IMG_20250803_215755_jpg.rf.02fea5708529fa5762186f65ab447c9e.jpg`
- Weak bridge to `train/images/IMG_20250803_215753_jpg.rf.665f84f766199c759bdabee414daf5ad.jpg` (pHash=26, ORB inliers=53/55, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_215753_jpg.rf.6b4e8f027d529c142a005c8042a782a1.jpg` (pHash=30, ORB inliers=3/12, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_215755_jpg.rf.86629ff209a27b91f91ada50042844d8.jpg` (pHash=34, ORB inliers=4/7, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_215755_jpg.rf.ee180e7cf5243b0f31a037d431aac714.jpg` (pHash=22, ORB inliers=219/223, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `train/images/IMG_20250803_215753_jpg.rf.fcef9d6acea49b62724194527689a086.jpg` ↔ `train/images/IMG_20250803_215755_jpg.rf.02fea5708529fa5762186f65ab447c9e.jpg`.

### AUTO-G0061 — 3 images; classes=Mobile; original splits=train,valid; status=provisional_strong_component
- `train/images/IMG_20250803_220114_jpg.rf.68d25d49b60c21e990f42a77571309d7.jpg`
- `valid/images/IMG_20250803_220117_jpg.rf.0594f79bcf0bff0635472395fec31752.jpg`
- `valid/images/IMG_20250803_220119_jpg.rf.64e2198d7a2d24e7786b2bbc40de885d.jpg`
- Weak bridge to `train/images/IMG_20250803_220114_jpg.rf.402c2f2a849eda8b99f516509e711de9.jpg` (pHash=32, ORB inliers=0/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250803_220114_jpg.rf.b245c7433620d8fc6007ccff153e2fb3.jpg` (pHash=28, ORB inliers=144/151, same timestamp=True, conflict=none).

### AUTO-G0065 — 2 images; classes=Mobile; original splits=train,valid; status=provisional_strong_component
- `train/images/IMG_20250803_220207_jpg.rf.0f17df4cec5b7117887884be5ea29430.jpg`
- `valid/images/IMG_20250803_220206_jpg.rf.de998e9b904dcfe77468124a9e729352.jpg`
- Weak bridge to `train/images/IMG_20250803_220207_jpg.rf.5109122280ff4b0088d809ef5e0ea94a.jpg` (pHash=26, ORB inliers=246/247, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220207_jpg.rf.d1f6d01c4257a71048f99fea11ad18d9.jpg` (pHash=26, ORB inliers=0/1, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- **Strong graph bridge to review:** `train/images/IMG_20250803_220207_jpg.rf.0f17df4cec5b7117887884be5ea29430.jpg` ↔ `valid/images/IMG_20250803_220206_jpg.rf.de998e9b904dcfe77468124a9e729352.jpg`.

### AUTO-G0091 — 2 images; classes=Mobile; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250805_203804_jpg.rf.23acd615013748d7bdd3d547bb50935a.jpg`
- `train/images/IMG_20250805_203806_jpg.rf.a88c5e5ff1cddfe7235a919241c54687.jpg`
- Weak bridge to `train/images/IMG_20250803_215701_jpg.rf.e582db8c0d1f0560b6d54cc15f102097.jpg` (pHash=6, ORB inliers=0/2, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250803_215701_jpg.rf.e582db8c0d1f0560b6d54cc15f102097.jpg` (pHash=10, ORB inliers=0/0, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250805_203804_jpg.rf.2aa2f4ad307106ebef06541aae137975.jpg` (pHash=18, ORB inliers=179/185, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203804_jpg.rf.d0d25dd17b163cc60b1a269ea27346dd.jpg` (pHash=24, ORB inliers=205/215, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203922_jpg.rf.9772c2edbd0d2281767d124674a04042.jpg` (pHash=10, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231226_jpg.rf.f5c80448ec1e3edadd7fe58129065dd8.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250805_203806_jpg.rf.34c678b8361cf794b5af0e3b63a58a07.jpg` (pHash=26, ORB inliers=202/206, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203806_jpg.rf.7fd019ac3a6efc21e4b37227c5bb73e1.jpg` (pHash=32, ORB inliers=197/206, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `train/images/IMG_20250805_203804_jpg.rf.23acd615013748d7bdd3d547bb50935a.jpg` ↔ `train/images/IMG_20250805_203806_jpg.rf.a88c5e5ff1cddfe7235a919241c54687.jpg`.

### AUTO-G0095 — 4 images; classes=Mobile; original splits=train,valid; status=provisional_strong_component
- `train/images/IMG_20250805_203809_jpg.rf.d485d98923b15ece68c49e6175f4a60c.jpg`
- `train/images/IMG_20250805_203810_jpg.rf.a2d06f67c179d20800e307f90c6fc2b4.jpg`
- `train/images/IMG_20250805_203811_jpg.rf.699b0103080853f83b5e914cd660a1b6.jpg`
- `valid/images/IMG_20250805_203812_jpg.rf.41c07d004c67e427803ca860bc03612b.jpg`
- Weak bridge to `train/images/IMG_20250805_203809_jpg.rf.8369f309373ddafa37f990cc84d96df6.jpg` (pHash=34, ORB inliers=235/251, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203809_jpg.rf.f06f5be1cc7487eb12962a962b35eff3.jpg` (pHash=30, ORB inliers=50/56, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203810_jpg.rf.5dea20372f68da2fdbd452fec56b22e4.jpg` (pHash=30, ORB inliers=27/32, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203810_jpg.rf.be30c1ef7d353b90ba6f83aacd045063.jpg` (pHash=34, ORB inliers=18/24, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203811_jpg.rf.3aea3cdfa197fc1c8c2597ab337d7345.jpg` (pHash=38, ORB inliers=28/40, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203811_jpg.rf.cacd2c991eef295d45f4243e2a06b1c9.jpg` (pHash=32, ORB inliers=33/37, same timestamp=True, conflict=none).

### AUTO-G0107 — 3 images; classes=Mobile; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250805_203902_jpg.rf.a23fe3b1c30257e8947ef9b1f48a0746.jpg`
- `train/images/IMG_20250805_203903_1_jpg.rf.a847a14922cb7973b83d4e2f3c655fb5.jpg`
- `train/images/IMG_20250805_203903_jpg.rf.e34bfc57526f72ced179b5397bc93f15.jpg`
- Weak bridge to `test/images/IMG_20250803_220228_jpg.rf.8e94e7ce7153325b8e181e6a1d331ef0.jpg` (pHash=10, ORB inliers=7/8, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250803_220226_jpg.rf.ff911118579e3a71da019d386eb1d7eb.jpg` (pHash=10, ORB inliers=0/5, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203902_jpg.rf.327b0dd828626fc4dc5867660d12285a.jpg` (pHash=26, ORB inliers=216/236, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203902_jpg.rf.bb8f8213eb4744ef892e1fd9c46edc8d.jpg` (pHash=18, ORB inliers=230/238, same timestamp=True, conflict=none).
- Weak bridge to `valid/images/IMG_20250803_220227_jpg.rf.c42373ef7c64d08b36804cc7fcad4d15.jpg` (pHash=10, ORB inliers=0/0, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_1_jpg.rf.e6eac1f51cfb25d40b34be0a5ec6f1d5.jpg` (pHash=18, ORB inliers=232/239, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_1_jpg.rf.f5cee8b017104a9e8573eb4524f0ff98.jpg` (pHash=28, ORB inliers=213/222, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_jpg.rf.8f402e3b8d9da9885fbbcabec1f42576.jpg` (pHash=42, ORB inliers=242/262, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_jpg.rf.f23b1a96ab603bb4dbb0d5dc1b359a97.jpg` (pHash=24, ORB inliers=28/38, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_1_jpg.rf.e6eac1f51cfb25d40b34be0a5ec6f1d5.jpg` (pHash=16, ORB inliers=224/229, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_1_jpg.rf.f5cee8b017104a9e8573eb4524f0ff98.jpg` (pHash=30, ORB inliers=217/222, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_jpg.rf.8f402e3b8d9da9885fbbcabec1f42576.jpg` (pHash=40, ORB inliers=270/291, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_jpg.rf.f23b1a96ab603bb4dbb0d5dc1b359a97.jpg` (pHash=24, ORB inliers=26/32, same timestamp=True, conflict=none).

### AUTO-G0108 — 2 images; classes=Mobile; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250805_203902_jpg.rf.bb8f8213eb4744ef892e1fd9c46edc8d.jpg`
- `train/images/IMG_20250805_203903_1_jpg.rf.e6eac1f51cfb25d40b34be0a5ec6f1d5.jpg`
- Weak bridge to `train/images/IMG_20250805_203902_jpg.rf.327b0dd828626fc4dc5867660d12285a.jpg` (pHash=16, ORB inliers=190/210, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203902_jpg.rf.a23fe3b1c30257e8947ef9b1f48a0746.jpg` (pHash=18, ORB inliers=230/238, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231255_1_jpg.rf.e32021e200cdfed4065135f35ee4bca8.jpg` (pHash=10, ORB inliers=0/0, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250805_203903_1_jpg.rf.a847a14922cb7973b83d4e2f3c655fb5.jpg` (pHash=18, ORB inliers=232/239, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_1_jpg.rf.f5cee8b017104a9e8573eb4524f0ff98.jpg` (pHash=26, ORB inliers=190/196, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_jpg.rf.8f402e3b8d9da9885fbbcabec1f42576.jpg` (pHash=28, ORB inliers=225/241, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_jpg.rf.e34bfc57526f72ced179b5397bc93f15.jpg` (pHash=16, ORB inliers=224/229, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203903_jpg.rf.f23b1a96ab603bb4dbb0d5dc1b359a97.jpg` (pHash=12, ORB inliers=16/20, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `train/images/IMG_20250805_203902_jpg.rf.bb8f8213eb4744ef892e1fd9c46edc8d.jpg` ↔ `train/images/IMG_20250805_203903_1_jpg.rf.e6eac1f51cfb25d40b34be0a5ec6f1d5.jpg`.

### AUTO-G0112 — 2 images; classes=Mobile; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250805_203907_jpg.rf.842e4fc0cf7a9464305766b439d364c5.jpg`
- `train/images/IMG_20250805_203908_jpg.rf.f30f1e3780a6cd716c1f2a3b2f984f8b.jpg`
- Weak bridge to `train/images/IMG_20250805_203907_jpg.rf.bfe8f9064ad57b8bfa2534f603f5300f.jpg` (pHash=24, ORB inliers=5/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250805_203907_jpg.rf.dd69ee28911b2c95d949c538ac5799d4.jpg` (pHash=24, ORB inliers=140/147, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203908_jpg.rf.4c4645aca4c09485234d7bb650187ba9.jpg` (pHash=28, ORB inliers=4/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250805_203908_jpg.rf.eda7b9d24b1a975e131ef6c9b7f43fe2.jpg` (pHash=28, ORB inliers=195/204, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `train/images/IMG_20250805_203907_jpg.rf.842e4fc0cf7a9464305766b439d364c5.jpg` ↔ `train/images/IMG_20250805_203908_jpg.rf.f30f1e3780a6cd716c1f2a3b2f984f8b.jpg`.

### AUTO-G0121 — 2 images; classes=Mobile; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250805_203922_jpg.rf.fbc486f138d29a49b1f3de783c184c12.jpg`
- `train/images/IMG_20250805_203923_jpg.rf.5c6fc3c85b160812cf0a3d80250b24d1.jpg`
- Weak bridge to `train/images/IMG_20250805_203922_jpg.rf.9772c2edbd0d2281767d124674a04042.jpg` (pHash=30, ORB inliers=186/202, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203922_jpg.rf.b747b03d920d3d0c33908474edce093c.jpg` (pHash=26, ORB inliers=4/4, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250805_203923_jpg.rf.c391e7c791e4efc10d2f5cd5e5ffabea.jpg` (pHash=20, ORB inliers=177/186, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203923_jpg.rf.d38a90effba05a5b6fd5e7ac1cef65ec.jpg` (pHash=38, ORB inliers=164/183, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `train/images/IMG_20250805_203922_jpg.rf.fbc486f138d29a49b1f3de783c184c12.jpg` ↔ `train/images/IMG_20250805_203923_jpg.rf.5c6fc3c85b160812cf0a3d80250b24d1.jpg`.

### AUTO-G0126 — 9 images; classes=Battery_Waste; original splits=train,valid; status=provisional_strong_component
- `train/images/IMG_20250822_231134_1_jpg.rf.19857d21bf15db9b7b5a3e2669c53f03.jpg`
- `train/images/IMG_20250822_231135_1_jpg.rf.9c3779c2a1dd1c261e91e6a7e74a3e6b.jpg`
- `train/images/IMG_20250822_231136_jpg.rf.744ffa1266ae25359cf64542d5c6e0ef.jpg`
- `train/images/IMG_20250822_231137_jpg.rf.af50867b8ea72e358cc90be68f1b3739.jpg`
- `train/images/IMG_20250822_231138_1_jpg.rf.3868dbe6e9b2118cafab7f0bd82fa478.jpg`
- `train/images/IMG_20250822_231138_jpg.rf.0edbeb921cea2c27bf09a0a7aea04b04.jpg`
- `valid/images/IMG_20250822_231134_jpg.rf.76bc5a7c35544dbfd4c168ab5ec537e3.jpg`
- `valid/images/IMG_20250822_231135_jpg.rf.5eab8abfbf6ce0006e47452bb4e351aa.jpg`
- `valid/images/IMG_20250822_231136_1_jpg.rf.dfd740b56687c8a262b0ca3521ea517e.jpg`
- Weak bridge to `train/images/IMG_20250822_231134_1_jpg.rf.3f4911838e7d399e2c29009938d92851.jpg` (pHash=30, ORB inliers=170/174, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231134_1_jpg.rf.82a4dc567acd78e78794d5bba1b3300a.jpg` (pHash=24, ORB inliers=4/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231134_1_jpg.rf.3f4911838e7d399e2c29009938d92851.jpg` (pHash=26, ORB inliers=185/189, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231134_1_jpg.rf.82a4dc567acd78e78794d5bba1b3300a.jpg` (pHash=28, ORB inliers=5/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231135_1_jpg.rf.40d27b8048589bd7f6a91ff8c07f1d84.jpg` (pHash=28, ORB inliers=7/12, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231135_1_jpg.rf.40d27b8048589bd7f6a91ff8c07f1d84.jpg` (pHash=28, ORB inliers=5/10, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231135_1_jpg.rf.d42078c33df83bcadfbe932693eb51fb.jpg` (pHash=18, ORB inliers=181/191, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231135_1_jpg.rf.d42078c33df83bcadfbe932693eb51fb.jpg` (pHash=20, ORB inliers=182/186, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231136_jpg.rf.892f7aaf1ac6a2fe9c249f6744bb872e.jpg` (pHash=36, ORB inliers=4/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231136_jpg.rf.facafbb903d0e49f4895b43e2448f178.jpg` (pHash=34, ORB inliers=297/312, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231136_jpg.rf.892f7aaf1ac6a2fe9c249f6744bb872e.jpg` (pHash=36, ORB inliers=5/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231136_jpg.rf.facafbb903d0e49f4895b43e2448f178.jpg` (pHash=34, ORB inliers=271/289, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231137_jpg.rf.082d8b78b35dc56a1b88168f6e387d7d.jpg` (pHash=20, ORB inliers=4/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231137_jpg.rf.96ed63464751f15129b76e9093788b0a.jpg` (pHash=30, ORB inliers=264/273, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231138_1_jpg.rf.0ac4f5a7cb3daffeb355a4abc877a794.jpg` (pHash=40, ORB inliers=225/240, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231138_1_jpg.rf.0ac4f5a7cb3daffeb355a4abc877a794.jpg` (pHash=42, ORB inliers=205/215, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231138_1_jpg.rf.793734fc9cc4adbb935973beb2fd4034.jpg` (pHash=20, ORB inliers=5/10, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.327c80badfebf025a88bbe352e1fd3cc.jpg` (pHash=26, ORB inliers=196/202, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.c2066d8066e40dd14cf4c6af5b886414.jpg` (pHash=30, ORB inliers=4/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_1_jpg.rf.793734fc9cc4adbb935973beb2fd4034.jpg` (pHash=20, ORB inliers=6/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.327c80badfebf025a88bbe352e1fd3cc.jpg` (pHash=28, ORB inliers=207/216, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.c2066d8066e40dd14cf4c6af5b886414.jpg` (pHash=30, ORB inliers=5/7, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).

### AUTO-G0128 — 2 images; classes=Battery_Waste; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250822_231134_1_jpg.rf.82a4dc567acd78e78794d5bba1b3300a.jpg`
- `train/images/IMG_20250822_231136_jpg.rf.892f7aaf1ac6a2fe9c249f6744bb872e.jpg`
- Weak bridge to `train/images/IMG_20250803_220321_jpg.rf.18be45e7ac54d6c3c9f1bac80e64585e.jpg` (pHash=10, ORB inliers=4/5, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250805_203922_jpg.rf.9772c2edbd0d2281767d124674a04042.jpg` (pHash=10, ORB inliers=0/0, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250822_231134_1_jpg.rf.19857d21bf15db9b7b5a3e2669c53f03.jpg` (pHash=24, ORB inliers=4/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231134_1_jpg.rf.3f4911838e7d399e2c29009938d92851.jpg` (pHash=18, ORB inliers=4/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.327c80badfebf025a88bbe352e1fd3cc.jpg` (pHash=10, ORB inliers=4/8, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231226_jpg.rf.f5c80448ec1e3edadd7fe58129065dd8.jpg` (pHash=10, ORB inliers=0/5, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231134_jpg.rf.76bc5a7c35544dbfd4c168ab5ec537e3.jpg` (pHash=28, ORB inliers=5/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231136_jpg.rf.744ffa1266ae25359cf64542d5c6e0ef.jpg` (pHash=36, ORB inliers=4/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231136_jpg.rf.facafbb903d0e49f4895b43e2448f178.jpg` (pHash=12, ORB inliers=4/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231222_jpg.rf.6da281988f9a7ad6de6a237c773340b0.jpg` (pHash=10, ORB inliers=0/2, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231226_jpg.rf.f5c80448ec1e3edadd7fe58129065dd8.jpg` (pHash=10, ORB inliers=4/4, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.ff7c20ffa1b97e6f85e4860b6c2eefbc.jpg` (pHash=6, ORB inliers=0/0, same timestamp=False, conflict=none).
- Weak bridge to `valid/images/IMG_20250822_231136_1_jpg.rf.dfd740b56687c8a262b0ca3521ea517e.jpg` (pHash=36, ORB inliers=5/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- **Strong graph bridge to review:** `train/images/IMG_20250822_231134_1_jpg.rf.82a4dc567acd78e78794d5bba1b3300a.jpg` ↔ `train/images/IMG_20250822_231136_jpg.rf.892f7aaf1ac6a2fe9c249f6744bb872e.jpg`.

### AUTO-G0135 — 2 images; classes=Battery_Waste; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250822_231138_1_jpg.rf.793734fc9cc4adbb935973beb2fd4034.jpg`
- `train/images/IMG_20250822_231138_jpg.rf.c2066d8066e40dd14cf4c6af5b886414.jpg`
- Weak bridge to `train/images/IMG_20250803_220022_jpg.rf.1b9bf10def62ebe2424913192f6033b6.jpg` (pHash=10, ORB inliers=0/0, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250822_231138_1_jpg.rf.0ac4f5a7cb3daffeb355a4abc877a794.jpg` (pHash=28, ORB inliers=0/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_1_jpg.rf.0ac4f5a7cb3daffeb355a4abc877a794.jpg` (pHash=24, ORB inliers=4/8, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_1_jpg.rf.3868dbe6e9b2118cafab7f0bd82fa478.jpg` (pHash=20, ORB inliers=5/10, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_1_jpg.rf.3868dbe6e9b2118cafab7f0bd82fa478.jpg` (pHash=30, ORB inliers=4/9, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.0edbeb921cea2c27bf09a0a7aea04b04.jpg` (pHash=20, ORB inliers=6/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.327c80badfebf025a88bbe352e1fd3cc.jpg` (pHash=20, ORB inliers=4/7, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.0edbeb921cea2c27bf09a0a7aea04b04.jpg` (pHash=30, ORB inliers=5/7, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.327c80badfebf025a88bbe352e1fd3cc.jpg` (pHash=22, ORB inliers=7/16, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- **Strong graph bridge to review:** `train/images/IMG_20250822_231138_1_jpg.rf.793734fc9cc4adbb935973beb2fd4034.jpg` ↔ `train/images/IMG_20250822_231138_jpg.rf.c2066d8066e40dd14cf4c6af5b886414.jpg`.

### AUTO-G0141 — 4 images; classes=Battery_Waste; original splits=train,valid; status=provisional_strong_component
- `train/images/IMG_20250822_231200_jpg.rf.0f6db813145281c72c353be21cbd72aa.jpg`
- `train/images/IMG_20250822_231202_jpg.rf.1763ce102999536a602ed04b81bf0435.jpg`
- `valid/images/IMG_20250822_231201_1_jpg.rf.3ee4562ffda16d4de26bfe881d3c0a8b.jpg`
- `valid/images/IMG_20250822_231201_jpg.rf.44bc2d8fc2946f343332124c030baa7e.jpg`
- Weak bridge to `train/images/IMG_20250822_231200_jpg.rf.81019355e8515e8e16bfa79d6ab2b910.jpg` (pHash=32, ORB inliers=0/1, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231200_jpg.rf.ff53aaba394c0304da0a909829ab90c5.jpg` (pHash=32, ORB inliers=0/2, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231202_jpg.rf.9985948cecec55daa36ed7a5c51b5073.jpg` (pHash=32, ORB inliers=122/135, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231202_jpg.rf.f9ca8ef5be33db85e95f12d237090966.jpg` (pHash=28, ORB inliers=112/128, same timestamp=True, conflict=none).

### AUTO-G0156 — 2 images; classes=Battery_Waste; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250822_231226_jpg.rf.af949f552ee40abb9ba0d2ea22afdb16.jpg`
- `train/images/IMG_20250822_231229_jpg.rf.74aa09b68e143f0d05c0449cf7200946.jpg`
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.327c80badfebf025a88bbe352e1fd3cc.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231138_jpg.rf.327c80badfebf025a88bbe352e1fd3cc.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231226_jpg.rf.a384c4024b55ba7bf2eda542344eef69.jpg` (pHash=32, ORB inliers=127/138, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231226_jpg.rf.f5c80448ec1e3edadd7fe58129065dd8.jpg` (pHash=10, ORB inliers=7/12, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231239_1_jpg.rf.ff7c20ffa1b97e6f85e4860b6c2eefbc.jpg` (pHash=8, ORB inliers=0/6, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.9286a8bea2098e273163b8a58657f4c0.jpg` (pHash=18, ORB inliers=0/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231229_jpg.rf.e7e6036503debb760b34e87faa28dc66.jpg` (pHash=34, ORB inliers=106/132, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `train/images/IMG_20250822_231226_jpg.rf.af949f552ee40abb9ba0d2ea22afdb16.jpg` ↔ `train/images/IMG_20250822_231229_jpg.rf.74aa09b68e143f0d05c0449cf7200946.jpg`.

### AUTO-G0169 — 2 images; classes=Battery_Waste; original splits=train; status=provisional_strong_component
- `train/images/IMG_20250822_231256_jpg.rf.b5c48e1da6f07d1a4053ac81da1a3b2c.jpg`
- `train/images/IMG_20250822_231257_jpg.rf.faf456eebb5f026fde69eb667b9ae98e.jpg`
- Weak bridge to `test/images/IMG_20250822_231256_1_jpg.rf.aea99bc61d2369c19f56241e3f0876cf.jpg` (pHash=24, ORB inliers=191/193, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231255_1_jpg.rf.e32021e200cdfed4065135f35ee4bca8.jpg` (pHash=10, ORB inliers=5/5, same timestamp=False, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231256_jpg.rf.77db81774004ef79c28d543c812b7e54.jpg` (pHash=30, ORB inliers=2/5, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231256_jpg.rf.eb79bcac78e20f8fb24d570dc248ba25.jpg` (pHash=22, ORB inliers=224/231, same timestamp=True, conflict=none).
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.2db20ee5a5e4fda240d26419f21c7ee8.jpg` (pHash=24, ORB inliers=0/6, same timestamp=True, conflict=timestamp_match_but_visual_features_disagree).
- Weak bridge to `train/images/IMG_20250822_231257_jpg.rf.9f96d6100cf9c4dab496702706a56721.jpg` (pHash=30, ORB inliers=155/161, same timestamp=True, conflict=none).
- **Strong graph bridge to review:** `train/images/IMG_20250822_231256_jpg.rf.b5c48e1da6f07d1a4053ac81da1a3b2c.jpg` ↔ `train/images/IMG_20250822_231257_jpg.rf.faf456eebb5f026fde69eb667b9ae98e.jpg`.

### AUTO-G0172 — 2 images; classes=Mobile; original splits=valid; status=provisional_strong_component
- `valid/images/IMG_20250803_220126_jpg.rf.16a1c934ac06cbc5d2ee0f8b4c02e96b.jpg`
- `valid/images/IMG_20250803_220127_jpg.rf.29454ccaf0bd8058e12a5925d3619982.jpg`
- Weak bridge to `train/images/IMG_20250803_215753_jpg.rf.6b4e8f027d529c142a005c8042a782a1.jpg` (pHash=10, ORB inliers=0/1, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250803_215753_jpg.rf.6b4e8f027d529c142a005c8042a782a1.jpg` (pHash=10, ORB inliers=0/3, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250803_215755_jpg.rf.86629ff209a27b91f91ada50042844d8.jpg` (pHash=8, ORB inliers=0/3, same timestamp=False, conflict=different_original_class_labels).
- Weak bridge to `train/images/IMG_20250803_215755_jpg.rf.86629ff209a27b91f91ada50042844d8.jpg` (pHash=8, ORB inliers=0/5, same timestamp=False, conflict=different_original_class_labels).
- **Strong graph bridge to review:** `valid/images/IMG_20250803_220126_jpg.rf.16a1c934ac06cbc5d2ee0f8b4c02e96b.jpg` ↔ `valid/images/IMG_20250803_220127_jpg.rf.29454ccaf0bd8058e12a5925d3619982.jpg`.

### AUTO-G0175 — 2 images; classes=Mobile; original splits=valid; status=provisional_strong_component
- `valid/images/IMG_20250805_203851_jpg.rf.30549cb0368d0ebf082a10e12534ea69.jpg`
- `valid/images/IMG_20250805_203852_jpg.rf.14242e22d41ed3ca487f99b193502f87.jpg`
- **Strong graph bridge to review:** `valid/images/IMG_20250805_203851_jpg.rf.30549cb0368d0ebf082a10e12534ea69.jpg` ↔ `valid/images/IMG_20250805_203852_jpg.rf.14242e22d41ed3ca487f99b193502f87.jpg`.

## Manifest and use

`data/processed/mobile_battery_auto_group_manifest_v2.csv` contains every target image, original class and split, exact SHA-256, timestamp token, provisional strong-group ID, weak-only and all-evidence component IDs/sizes, bridge constraints, conflict status, and any existing v2 review decision/notes. `new_split_assignment` is blank for every record.
`data/processed/mobile_battery_auto_group_edges_v2.csv` contains exact, strong, and weak pair evidence with metrics and conflict flags.

## Review cautions and limitations

- Strong connected components are candidate image-family groupings, not physical-device IDs. Bridge edges can join otherwise distinct subgroups; inspect every flagged bridge before treating a component as one indivisible physical object.
- Weak bridges are retained as unresolved split blockers but are not merged into strong groups. If all candidate edges chain into a large component, that is flagged as a split conflict rather than silently treated as one identity group.
- Filename timestamps are weak evidence only. Same timestamps can be shared by augmented exports, but do not establish device identity or capture session identity.
- Existing human decisions and notes are copied from review v2 where an image path is present. No new reviewer decisions, labels, device IDs, or capture-session IDs are created.
- This is a grouping audit, not a model evaluation. Review labels and all flagged bridges before using the manifest to create new splits.
