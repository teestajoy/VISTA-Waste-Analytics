# Mobile and Battery_Waste related-image audit (v1)

- Archive inspected in place: `~/Downloads/VISTA_mobile_dataset_raw.zip` (local user Downloads folder)
- Archive size: 72,987,704 bytes; original archive was not extracted or modified.
- Scope: target images labeled `Mobile` (YOLO class id 6) and `Battery_Waste` (class id 0), as specified by the archive `data.yaml`.
- Image identifiers below are archive-relative paths. SHA-256 values are abbreviated to 12 hex characters in tables; hashes were computed from the original archived bytes.

## Archive metadata and scope counts

The archive `README.dataset.txt` identifies the source as the Roboflow E-Waste dataset and states CC BY 4.0. `README.roboflow.txt` says it contains 2,157 images, YOLOv9 annotations, 640×640 stretched resizing, auto-contrast, and three augmentation versions per source image. `data.yaml` maps class id 0 to `Battery_Waste` and id 6 to `Mobile`. The README provides dataset-level provenance but no per-image physical-device/session identifiers.

| Class | train | valid | test | Total |
|---|---:|---:|---:|---:|
| Mobile | 99 | 26 | 26 | 151 |
| Battery_Waste | 99 | 26 | 23 | 148 |

The audit decoded 299 target-class image records successfully. No decode or image lookup failures were found.

## Findings at a glance

- Exact SHA-256 duplicates among target-class image records: **none found**.
- Filename timestamp groups crossing splits: **20 groups involving 64 records** (18 Battery_Waste groups; 2 Mobile groups).
- Perceptual shortlist: pHash Hamming distance ≤10 generated 206 cross-split pair candidates. Geometric screen using ORB + RANSAC retained **174 pair links**, joined into **35 candidate similarity components** involving **142 distinct images**.
- These are candidate related-image groups, not proof of physical-device identity. Similarity components can also connect consecutive captures of the same scene or visually similar objects. No physical-device annotations are present.
- The 20 timestamp groups and the ORB/pHash components are independent evidence views and overlap; do not add their counts together.

## Method and confidence categories

1. **Exact duplicate:** SHA-256 computed from archived image bytes. Matching hashes would confirm byte-identical files. No matching hash pairs were found.
2. **Filename/time evidence:** extracted the `YYYYMMDD_HHMMSS` token from image filenames and grouped exact timestamp matches by class. These are timestamp-associated records; timestamps alone do not prove a shared device or even the same original exposure.
3. **Likely visual near-duplicate candidate:** grayscale perceptual hash (32×32 DCT, 8×8 low-frequency signature) used to shortlist cross-split pairs at Hamming distance ≤10; ORB features were matched with a 0.72 ratio test; homography inliers were estimated using RANSAC at 4 pixels. A retained link requires ≥20 inliers and ≥0.50 inlier fraction. The listed connected components are transitive candidate groups based on those links. They need visual/manual review before being treated as one device group.
4. **Uncertain:** pHash-only candidates that did not meet the ORB/RANSAC link threshold are not elevated to likely near-duplicates. pHash can collide on repeated backgrounds or similar framing. The scan is a candidate-finding heuristic, not a semantic device recognizer.

The three-version augmentation note in the README makes related variants plausible, but it does not identify which individual files are augmentations of which source. No claim that an image pair is an augmented copy is made without per-image source linkage or manual confirmation.

## Cross-split filename timestamp groups

Every row below has an exact timestamp token shared by files in different original splits. `T/V/R` denote train/valid/test counts; file names are listed so candidates can be checked directly.

| Group | Class | Timestamp | T/V/R | Archived image identifiers (filename; SHA-256 prefix) |
|---|---|---|---:|---|
| TS-B-20250803_220320 | Battery_Waste | 20250803_220320 | 0/1/1 | `IMG_20250803_220320_jpg.rf.10f96e152fbca7bb72a58524897b976a.jpg` (3f2830dcdfef); `IMG_20250803_220320_1_jpg.rf.4004befc73e0f7a956fcd25d6de4037b.jpg` (ad42835caa15) |
| TS-B-20250803_220323 | Battery_Waste | 20250803_220323 | 0/1/1 | `IMG_20250803_220323_1_jpg.rf.573457ae657057432e59564c621a62b2.jpg` (d9db9f1b7c21); `IMG_20250803_220323_jpg.rf.dae4ff08759027feadbee436a4098da3.jpg` (64fd70ded84e) |
| TS-B-20250822_231124 | Battery_Waste | 20250822_231124 | 0/1/1 | `IMG_20250822_231124_jpg.rf.454eb5a36ebdb2ad770c60899ed1ff5d.jpg` (ee3dfb89efbf); `IMG_20250822_231124_1_jpg.rf.15f6be87893a7f3e91d86d821962f40f.jpg` (b18b0a16d80f) |
| TS-B-20250822_231125 | Battery_Waste | 20250822_231125 | 3/1/0 | `IMG_20250822_231125_1_jpg.rf.055929b00c16ff17c6d4a49b7cef58a9.jpg` (8e5aca2aca1c); `IMG_20250822_231125_1_jpg.rf.24e5f0335754fa334350f28e5fc26691.jpg` (6fc05b0c539d); `IMG_20250822_231125_1_jpg.rf.51f8307cc11b5cddecd0ca3e02555e70.jpg` (6729adf52bd6); `IMG_20250822_231125_jpg.rf.712cf5e8bb04d06029470cfebfd9db8a.jpg` (78556295e215) |
| TS-B-20250822_231134 | Battery_Waste | 20250822_231134 | 3/1/0 | `IMG_20250822_231134_1_jpg.rf.19857d21bf15db9b7b5a3e2669c53f03.jpg` (37d6d330c5d2); `IMG_20250822_231134_1_jpg.rf.3f4911838e7d399e2c29009938d92851.jpg` (c53fb020c3e8); `IMG_20250822_231134_1_jpg.rf.82a4dc567acd78e78794d5bba1b3300a.jpg` (383a67a3fdcf); `IMG_20250822_231134_jpg.rf.76bc5a7c35544dbfd4c168ab5ec537e3.jpg` (25b7cc28fe02) |
| TS-B-20250822_231135 | Battery_Waste | 20250822_231135 | 3/1/0 | `IMG_20250822_231135_1_jpg.rf.40d27b8048589bd7f6a91ff8c07f1d84.jpg` (bf5dd5a09b55); `IMG_20250822_231135_1_jpg.rf.9c3779c2a1dd1c261e91e6a7e74a3e6b.jpg` (f20c7fcbefd4); `IMG_20250822_231135_1_jpg.rf.d42078c33df83bcadfbe932693eb51fb.jpg` (6f3001e40981); `IMG_20250822_231135_jpg.rf.5eab8abfbf6ce0006e47452bb4e351aa.jpg` (d1a1618ed5c5) |
| TS-B-20250822_231136 | Battery_Waste | 20250822_231136 | 3/1/0 | `IMG_20250822_231136_jpg.rf.744ffa1266ae25359cf64542d5c6e0ef.jpg` (b675cd9fb0d9); `IMG_20250822_231136_jpg.rf.892f7aaf1ac6a2fe9c249f6744bb872e.jpg` (949ef9132a24); `IMG_20250822_231136_jpg.rf.facafbb903d0e49f4895b43e2448f178.jpg` (3c6a8c941ce7); `IMG_20250822_231136_1_jpg.rf.dfd740b56687c8a262b0ca3521ea517e.jpg` (d3479f2122be) |
| TS-B-20250822_231153 | Battery_Waste | 20250822_231153 | 3/1/0 | `IMG_20250822_231153_1_jpg.rf.055d65f1f0714e9fda6d890f79bc15ab.jpg` (f812eb53457c); `IMG_20250822_231153_1_jpg.rf.0ccc4241354b45d12d7828899bc92f05.jpg` (f7fb0eb00e9f); `IMG_20250822_231153_1_jpg.rf.f19abd752876b09f640fce636287ab42.jpg` (7e33c96a14e8); `IMG_20250822_231153_jpg.rf.c0581636936ffd3f551a62ae4e3aadb3.jpg` (b9536ef77a8d) |
| TS-B-20250822_231209 | Battery_Waste | 20250822_231209 | 3/0/1 | `IMG_20250822_231209_1_jpg.rf.fcf651145e2d95b81efe4e9904f3cfed.jpg` (4c94ae150118); `IMG_20250822_231209_jpg.rf.4c130981a9fbbd3987e2dd770795972d.jpg` (596a84bceee6); `IMG_20250822_231209_jpg.rf.5fbcecd2f828c4d73be632426a997488.jpg` (c114f4ac0691); `IMG_20250822_231209_jpg.rf.71062767ff841e54501edf3ddc7a9c7a.jpg` (e7a3178ce190) |
| TS-B-20250822_231210 | Battery_Waste | 20250822_231210 | 3/0/1 | `IMG_20250822_231210_jpg.rf.3e928d15c853583abf8b89eff296af81.jpg` (b96bf989b15f); `IMG_20250822_231210_1_jpg.rf.7e7d56093797517d49417300b1eceb44.jpg` (7dccb2b73355); `IMG_20250822_231210_1_jpg.rf.b643e8ffe7294d17bcf888a766981b1a.jpg` (b2a9190438d9); `IMG_20250822_231210_1_jpg.rf.bf5f830d9485eb8d3bf5ca84d009a28d.jpg` (f494c712980b) |
| TS-B-20250822_231211 | Battery_Waste | 20250822_231211 | 0/1/1 | `IMG_20250822_231211_jpg.rf.29439c714ba00a43492ccf5b2673e53f.jpg` (574fc925f45d); `IMG_20250822_231211_1_jpg.rf.81b7eb1b4bb750b87e50a18ac5a90583.jpg` (056314d42cc5) |
| TS-B-20250822_231220 | Battery_Waste | 20250822_231220 | 3/1/0 | `IMG_20250822_231220_jpg.rf.2b36d46a7e22d101be3c9448b72ef6f1.jpg` (1f37415e1ea3); `IMG_20250822_231220_jpg.rf.847cf565415d2346e0fabc16a3fb1f28.jpg` (d1e0c48b4c07); `IMG_20250822_231220_jpg.rf.9c02b05d7b7b49ba3c228d2dc9bf9b7e.jpg` (54c24e28914c); `IMG_20250822_231220_1_jpg.rf.32d046cd028697cf3c99f250db87866d.jpg` (d8ccc6f2a489) |
| TS-B-20250822_231221 | Battery_Waste | 20250822_231221 | 0/1/1 | `IMG_20250822_231221_jpg.rf.f1b09c45232ce7f53d9929c1a6024cc0.jpg` (861e656422db); `IMG_20250822_231221_1_jpg.rf.187ee67700b833c74c2c876f9f8165c7.jpg` (c72694f24797) |
| TS-B-20250822_231238 | Battery_Waste | 20250822_231238 | 0/1/1 | `IMG_20250822_231238_jpg.rf.3c4cd2453b779bbf5224c35672da6c9d.jpg` (deeca37e84f9); `IMG_20250822_231238_1_jpg.rf.6b34d7600d1352cab6f4fa768e169697.jpg` (ed9f34f4995e) |
| TS-B-20250822_231239 | Battery_Waste | 20250822_231239 | 3/0/1 | `IMG_20250822_231239_jpg.rf.a708ada26d463e1fc8125634fc78cbf5.jpg` (efdf104ec099); `IMG_20250822_231239_1_jpg.rf.1b55959316611efa584bdc9a073bc60b.jpg` (381c60617f6a); `IMG_20250822_231239_1_jpg.rf.d7c89c449cafe2a0f21c3245793671f9.jpg` (77f12751463c); `IMG_20250822_231239_1_jpg.rf.ff7c20ffa1b97e6f85e4860b6c2eefbc.jpg` (1dbfdd5b78c4) |
| TS-B-20250822_231240 | Battery_Waste | 20250822_231240 | 3/1/0 | `IMG_20250822_231240_1_jpg.rf.29e3db85d081590d2c545b6a25bf448f.jpg` (1d030e0b7c8b); `IMG_20250822_231240_1_jpg.rf.4547dfb9464eb55f94020eafbe21c13f.jpg` (2da16eb24afb); `IMG_20250822_231240_1_jpg.rf.c599e12109d4ea69fd2c676c98f032ff.jpg` (8917b3ebd727); `IMG_20250822_231240_jpg.rf.0be8aa7a50a7759b7663c4d15fee22ee.jpg` (df1635d682c3) |
| TS-B-20250822_231241 | Battery_Waste | 20250822_231241 | 0/1/1 | `IMG_20250822_231241_1_jpg.rf.08f4254c1a6be413d72d48bc233c8175.jpg` (591d56268f95); `IMG_20250822_231241_jpg.rf.d6ccb4191c62c428828b1cebbd503123.jpg` (838623acbc96) |
| TS-B-20250822_231256 | Battery_Waste | 20250822_231256 | 3/0/1 | `IMG_20250822_231256_1_jpg.rf.aea99bc61d2369c19f56241e3f0876cf.jpg` (84fb71e063a0); `IMG_20250822_231256_jpg.rf.77db81774004ef79c28d543c812b7e54.jpg` (06802e75b896); `IMG_20250822_231256_jpg.rf.b5c48e1da6f07d1a4053ac81da1a3b2c.jpg` (ca90df3829a2); `IMG_20250822_231256_jpg.rf.eb79bcac78e20f8fb24d570dc248ba25.jpg` (97d00182307e) |
| TS-M-20250805_203327 | Mobile | 20250805_203327 | 0/1/1 | `IMG_20250805_203327_1_jpg.rf.fa4eadf62370e6bdea8c64d3f20085bf.jpg` (97ea83615b3c); `IMG_20250805_203327_jpg.rf.32be60b70e6f9a50917a48881f5ab85a.jpg` (466ebef46446) |
| TS-M-20250805_203338 | Mobile | 20250805_203338 | 3/0/1 | `IMG_20250805_203338_1_jpg.rf.bf2025683e05750526dc29fc829b8ac5.jpg` (c9be20972a6d); `IMG_20250805_203338_jpg.rf.29bc49ddc66d6aeb9a9b51d1b216f1ec.jpg` (8984b3f63de9); `IMG_20250805_203338_jpg.rf.68cadbff708a1a27b20e57a030073a53.jpg` (621bdbb570e6); `IMG_20250805_203338_jpg.rf.967c7c7b3f77ddf047ed91906ea0686d.jpg` (da6f2adb4d61) |

## Likely visual near-duplicate candidate components spanning splits

Candidate group names are audit-local identifiers only. Every group below has at least one cross-split ORB/RANSAC edge under the stated threshold. A row lists each member’s original split, filename timestamp when parsed, filename, and SHA prefix. The presence of multiple timestamps within one component means the component can capture adjacent frames/scene variants; it does not establish same-device identity.

### ND-01 — Battery_Waste; 2 images; splits T/V/R=0/1/1; 1 retained cross-split links
Evidence range: pHash distance 6–6; ORB/RANSAC inliers 81–81; inlier fraction 0.96–0.96.
- `test/images/IMG_20250803_215727_jpg.rf.ed0c24ebb5a6e14c10a1c4d99dbf6530.jpg`; timestamp `20250803_215727`; SHA-256 `4765929cb39f…`
- `valid/images/IMG_20250803_215725_jpg.rf.7d5f701994ab8ce96ca5d49e1d276ee3.jpg`; timestamp `20250803_215725`; SHA-256 `7cb1898f1883…`

### ND-02 — Battery_Waste; 2 images; splits T/V/R=0/1/1; 1 retained cross-split links
Evidence range: pHash distance 4–4; ORB/RANSAC inliers 94–94; inlier fraction 1.00–1.00.
- `test/images/IMG_20250803_215738_jpg.rf.576b5312e98cc236b1ed1246493119c6.jpg`; timestamp `20250803_215738`; SHA-256 `4c0357359e41…`
- `valid/images/IMG_20250803_215740_jpg.rf.758acdd6ad672a13af608eafffc805dd.jpg`; timestamp `20250803_215740`; SHA-256 `4fe9a77e48b1…`

### ND-03 — Battery_Waste; 3 images; splits T/V/R=1/0/2; 2 retained cross-split links
Evidence range: pHash distance 2–8; ORB/RANSAC inliers 162–175; inlier fraction 0.98–1.00.
- `test/images/IMG_20250803_215807_jpg.rf.949f4e1a94325c40ad658940123c68d8.jpg`; timestamp `20250803_215807`; SHA-256 `b0e033fc2443…`
- `test/images/IMG_20250803_215810_jpg.rf.bf03fc4486612281e874dcbde3267f8b.jpg`; timestamp `20250803_215810`; SHA-256 `9937a9ae3037…`
- `train/images/IMG_20250803_215808_jpg.rf.8a06ce738b95b31fc02705ce61b6d0ad.jpg`; timestamp `20250803_215808`; SHA-256 `2087cbdd00d2…`

### ND-04 — Battery_Waste; 3 images; splits T/V/R=1/1/1; 3 retained cross-split links
Evidence range: pHash distance 2–8; ORB/RANSAC inliers 242–252; inlier fraction 0.96–0.98.
- `test/images/IMG_20250803_220311_jpg.rf.72c9015fb578049a797e0d93bd65aee3.jpg`; timestamp `20250803_220311`; SHA-256 `0ba29832d53d…`
- `train/images/IMG_20250803_220312_jpg.rf.07000a53f58fd8b7ca0a0d97613d25f3.jpg`; timestamp `20250803_220312`; SHA-256 `279076bebcce…`
- `valid/images/IMG_20250803_220313_jpg.rf.64f586e62091d23ee0746e2a666392b5.jpg`; timestamp `20250803_220313`; SHA-256 `475febc1be8d…`

### ND-05 — Battery_Waste; 6 images; splits T/V/R=1/3/2; 11 retained cross-split links
Evidence range: pHash distance 0–4; ORB/RANSAC inliers 252–309; inlier fraction 0.96–0.99.
- `test/images/IMG_20250803_220320_jpg.rf.10f96e152fbca7bb72a58524897b976a.jpg`; timestamp `20250803_220320`; SHA-256 `3f2830dcdfef…`
- `test/images/IMG_20250803_220323_1_jpg.rf.573457ae657057432e59564c621a62b2.jpg`; timestamp `20250803_220323`; SHA-256 `d9db9f1b7c21…`
- `train/images/IMG_20250803_220321_jpg.rf.7d603d397b0d570f006ad3ed948b1fd8.jpg`; timestamp `20250803_220321`; SHA-256 `e94064b36086…`
- `valid/images/IMG_20250803_220320_1_jpg.rf.4004befc73e0f7a956fcd25d6de4037b.jpg`; timestamp `20250803_220320`; SHA-256 `ad42835caa15…`
- `valid/images/IMG_20250803_220322_jpg.rf.4ad5d2a122e2ec7a03c34971a4269fd9.jpg`; timestamp `20250803_220322`; SHA-256 `2b57ee0f0b9a…`
- `valid/images/IMG_20250803_220323_jpg.rf.dae4ff08759027feadbee436a4098da3.jpg`; timestamp `20250803_220323`; SHA-256 `64fd70ded84e…`

### ND-06 — Battery_Waste; 5 images; splits T/V/R=1/2/2; 8 retained cross-split links
Evidence range: pHash distance 0–4; ORB/RANSAC inliers 209–297; inlier fraction 0.97–0.99.
- `test/images/IMG_20250822_231124_jpg.rf.454eb5a36ebdb2ad770c60899ed1ff5d.jpg`; timestamp `20250822_231124`; SHA-256 `ee3dfb89efbf…`
- `test/images/IMG_20250822_231126_jpg.rf.7712b4701c0e3c8ad2f513a37cac8107.jpg`; timestamp `20250822_231126`; SHA-256 `53aadecd598f…`
- `train/images/IMG_20250822_231125_1_jpg.rf.055929b00c16ff17c6d4a49b7cef58a9.jpg`; timestamp `20250822_231125`; SHA-256 `8e5aca2aca1c…`
- `valid/images/IMG_20250822_231124_1_jpg.rf.15f6be87893a7f3e91d86d821962f40f.jpg`; timestamp `20250822_231124`; SHA-256 `b18b0a16d80f…`
- `valid/images/IMG_20250822_231125_jpg.rf.712cf5e8bb04d06029470cfebfd9db8a.jpg`; timestamp `20250822_231125`; SHA-256 `78556295e215…`

### ND-07 — Battery_Waste; 5 images; splits T/V/R=2/1/2; 8 retained cross-split links
Evidence range: pHash distance 2–8; ORB/RANSAC inliers 142–182; inlier fraction 0.95–0.97.
- `test/images/IMG_20250822_231151_jpg.rf.2607ab2dc8acaf81e39a0adf8e0950f7.jpg`; timestamp `20250822_231151`; SHA-256 `d6b5f32b32a9…`
- `test/images/IMG_20250822_231152_jpg.rf.e62543112ad7bd0d4593d533a2e3f231.jpg`; timestamp `20250822_231152`; SHA-256 `8025485b99f0…`
- `train/images/IMG_20250822_231153_1_jpg.rf.055d65f1f0714e9fda6d890f79bc15ab.jpg`; timestamp `20250822_231153`; SHA-256 `f812eb53457c…`
- `train/images/IMG_20250822_231155_jpg.rf.656e96d585807828d9bd58402edf9500.jpg`; timestamp `20250822_231155`; SHA-256 `94a81cda4861…`
- `valid/images/IMG_20250822_231153_jpg.rf.c0581636936ffd3f551a62ae4e3aadb3.jpg`; timestamp `20250822_231153`; SHA-256 `b9536ef77a8d…`

### ND-08 — Battery_Waste; 6 images; splits T/V/R=2/1/3; 11 retained cross-split links
Evidence range: pHash distance 0–8; ORB/RANSAC inliers 154–212; inlier fraction 0.90–0.96.
- `test/images/IMG_20250822_231209_1_jpg.rf.fcf651145e2d95b81efe4e9904f3cfed.jpg`; timestamp `20250822_231209`; SHA-256 `4c94ae150118…`
- `test/images/IMG_20250822_231210_jpg.rf.3e928d15c853583abf8b89eff296af81.jpg`; timestamp `20250822_231210`; SHA-256 `b96bf989b15f…`
- `test/images/IMG_20250822_231211_jpg.rf.29439c714ba00a43492ccf5b2673e53f.jpg`; timestamp `20250822_231211`; SHA-256 `574fc925f45d…`
- `train/images/IMG_20250822_231209_jpg.rf.4c130981a9fbbd3987e2dd770795972d.jpg`; timestamp `20250822_231209`; SHA-256 `596a84bceee6…`
- `train/images/IMG_20250822_231210_1_jpg.rf.bf5f830d9485eb8d3bf5ca84d009a28d.jpg`; timestamp `20250822_231210`; SHA-256 `f494c712980b…`
- `valid/images/IMG_20250822_231211_1_jpg.rf.81b7eb1b4bb750b87e50a18ac5a90583.jpg`; timestamp `20250822_231211`; SHA-256 `056314d42cc5…`

### ND-09 — Battery_Waste; 6 images; splits T/V/R=3/2/1; 11 retained cross-split links
Evidence range: pHash distance 2–10; ORB/RANSAC inliers 193–263; inlier fraction 0.94–0.99.
- `test/images/IMG_20250822_231221_jpg.rf.f1b09c45232ce7f53d9929c1a6024cc0.jpg`; timestamp `20250822_231221`; SHA-256 `861e656422db…`
- `train/images/IMG_20250822_231219_jpg.rf.03d08967efba3804f23ba15aac3d4cca.jpg`; timestamp `20250822_231219`; SHA-256 `1bd627ba7b2f…`
- `train/images/IMG_20250822_231220_jpg.rf.847cf565415d2346e0fabc16a3fb1f28.jpg`; timestamp `20250822_231220`; SHA-256 `d1e0c48b4c07…`
- `train/images/IMG_20250822_231222_jpg.rf.44ed716c90773067e2cf27b0d901024b.jpg`; timestamp `20250822_231222`; SHA-256 `00ddb0d770c9…`
- `valid/images/IMG_20250822_231220_1_jpg.rf.32d046cd028697cf3c99f250db87866d.jpg`; timestamp `20250822_231220`; SHA-256 `d8ccc6f2a489…`
- `valid/images/IMG_20250822_231221_1_jpg.rf.187ee67700b833c74c2c876f9f8165c7.jpg`; timestamp `20250822_231221`; SHA-256 `c72694f24797…`

### ND-10 — Battery_Waste; 5 images; splits T/V/R=3/0/2; 6 retained cross-split links
Evidence range: pHash distance 4–10; ORB/RANSAC inliers 205–242; inlier fraction 0.95–0.98.
- `test/images/IMG_20250822_231228_1_jpg.rf.8374d1d76854c1115c33fd7b7bedb107.jpg`; timestamp `20250822_231228`; SHA-256 `a449859fbf10…`
- `test/images/IMG_20250822_231228_jpg.rf.5878b0c9eb72ca9d1cdcfc596fc61234.jpg`; timestamp `20250822_231228`; SHA-256 `eea46ec6b6aa…`
- `train/images/IMG_20250822_231226_jpg.rf.a384c4024b55ba7bf2eda542344eef69.jpg`; timestamp `20250822_231226`; SHA-256 `6a266060efc9…`
- `train/images/IMG_20250822_231227_jpg.rf.8ff7f770de4f04e022e8f12c213838c1.jpg`; timestamp `20250822_231227`; SHA-256 `239f27497c94…`
- `train/images/IMG_20250822_231229_jpg.rf.e7e6036503debb760b34e87faa28dc66.jpg`; timestamp `20250822_231229`; SHA-256 `035f5631b57e…`

### ND-11 — Battery_Waste; 8 images; splits T/V/R=2/3/3; 21 retained cross-split links
Evidence range: pHash distance 2–10; ORB/RANSAC inliers 166–237; inlier fraction 0.96–1.00.
- `test/images/IMG_20250822_231238_jpg.rf.3c4cd2453b779bbf5224c35672da6c9d.jpg`; timestamp `20250822_231238`; SHA-256 `deeca37e84f9…`
- `test/images/IMG_20250822_231239_jpg.rf.a708ada26d463e1fc8125634fc78cbf5.jpg`; timestamp `20250822_231239`; SHA-256 `efdf104ec099…`
- `test/images/IMG_20250822_231241_1_jpg.rf.08f4254c1a6be413d72d48bc233c8175.jpg`; timestamp `20250822_231241`; SHA-256 `591d56268f95…`
- `train/images/IMG_20250822_231239_1_jpg.rf.d7c89c449cafe2a0f21c3245793671f9.jpg`; timestamp `20250822_231239`; SHA-256 `77f12751463c…`
- `train/images/IMG_20250822_231240_1_jpg.rf.4547dfb9464eb55f94020eafbe21c13f.jpg`; timestamp `20250822_231240`; SHA-256 `2da16eb24afb…`
- `valid/images/IMG_20250822_231238_1_jpg.rf.6b34d7600d1352cab6f4fa768e169697.jpg`; timestamp `20250822_231238`; SHA-256 `ed9f34f4995e…`
- `valid/images/IMG_20250822_231240_jpg.rf.0be8aa7a50a7759b7663c4d15fee22ee.jpg`; timestamp `20250822_231240`; SHA-256 `df1635d682c3…`
- `valid/images/IMG_20250822_231241_jpg.rf.d6ccb4191c62c428828b1cebbd503123.jpg`; timestamp `20250822_231241`; SHA-256 `838623acbc96…`

### ND-12 — Battery_Waste; 6 images; splits T/V/R=4/1/1; 6 retained cross-split links
Evidence range: pHash distance 4–10; ORB/RANSAC inliers 137–260; inlier fraction 0.96–0.98.
- `test/images/IMG_20250822_231256_1_jpg.rf.aea99bc61d2369c19f56241e3f0876cf.jpg`; timestamp `20250822_231256`; SHA-256 `84fb71e063a0…`
- `train/images/IMG_20250822_231255_1_jpg.rf.5f530548951cb35377a1987db865ecc0.jpg`; timestamp `20250822_231255`; SHA-256 `87332ccc2836…`
- `train/images/IMG_20250822_231255_jpg.rf.36ef9257a4993f8d263c5dd4cb7bf7f5.jpg`; timestamp `20250822_231255`; SHA-256 `ba4f5b11ade4…`
- `train/images/IMG_20250822_231256_jpg.rf.eb79bcac78e20f8fb24d570dc248ba25.jpg`; timestamp `20250822_231256`; SHA-256 `97d00182307e…`
- `train/images/IMG_20250822_231257_jpg.rf.9f96d6100cf9c4dab496702706a56721.jpg`; timestamp `20250822_231257`; SHA-256 `b94e38baa205…`
- `valid/images/IMG_20250822_231254_jpg.rf.b3dfdd482b858b6fb25f930221870e93.jpg`; timestamp `20250822_231254`; SHA-256 `d1ab9c97551e…`

### ND-13 — Battery_Waste; 4 images; splits T/V/R=1/3/0; 3 retained cross-split links
Evidence range: pHash distance 4–10; ORB/RANSAC inliers 103–235; inlier fraction 0.94–0.98.
- `train/images/IMG_20250803_215648_jpg.rf.c339d9561861e32aec3649527f4423f7.jpg`; timestamp `20250803_215648`; SHA-256 `24e39c1924b8…`
- `valid/images/IMG_20250803_215638_jpg.rf.ba4f3127b341131ea269136e3d2b124e.jpg`; timestamp `20250803_215638`; SHA-256 `91942de1bdc9…`
- `valid/images/IMG_20250803_215639_jpg.rf.fbe999007ecf4fd037195142c1702267.jpg`; timestamp `20250803_215639`; SHA-256 `eccbe839337f…`
- `valid/images/IMG_20250803_215645_jpg.rf.e82c2404db3f48100de5904104aa16a5.jpg`; timestamp `20250803_215645`; SHA-256 `cd77deb41e80…`

### ND-14 — Battery_Waste; 9 images; splits T/V/R=6/3/0; 15 retained cross-split links
Evidence range: pHash distance 0–10; ORB/RANSAC inliers 132–316; inlier fraction 0.95–0.99.
- `train/images/IMG_20250822_231134_1_jpg.rf.19857d21bf15db9b7b5a3e2669c53f03.jpg`; timestamp `20250822_231134`; SHA-256 `37d6d330c5d2…`
- `train/images/IMG_20250822_231135_1_jpg.rf.9c3779c2a1dd1c261e91e6a7e74a3e6b.jpg`; timestamp `20250822_231135`; SHA-256 `f20c7fcbefd4…`
- `train/images/IMG_20250822_231136_jpg.rf.744ffa1266ae25359cf64542d5c6e0ef.jpg`; timestamp `20250822_231136`; SHA-256 `b675cd9fb0d9…`
- `train/images/IMG_20250822_231137_jpg.rf.af50867b8ea72e358cc90be68f1b3739.jpg`; timestamp `20250822_231137`; SHA-256 `7c6cc43d68b1…`
- `train/images/IMG_20250822_231138_1_jpg.rf.3868dbe6e9b2118cafab7f0bd82fa478.jpg`; timestamp `20250822_231138`; SHA-256 `453cf40ab284…`
- `train/images/IMG_20250822_231138_jpg.rf.0edbeb921cea2c27bf09a0a7aea04b04.jpg`; timestamp `20250822_231138`; SHA-256 `83abeac2c900…`
- `valid/images/IMG_20250822_231134_jpg.rf.76bc5a7c35544dbfd4c168ab5ec537e3.jpg`; timestamp `20250822_231134`; SHA-256 `25b7cc28fe02…`
- `valid/images/IMG_20250822_231135_jpg.rf.5eab8abfbf6ce0006e47452bb4e351aa.jpg`; timestamp `20250822_231135`; SHA-256 `d1a1618ed5c5…`
- `valid/images/IMG_20250822_231136_1_jpg.rf.dfd740b56687c8a262b0ca3521ea517e.jpg`; timestamp `20250822_231136`; SHA-256 `d3479f2122be…`

### ND-15 — Battery_Waste; 4 images; splits T/V/R=2/2/0; 4 retained cross-split links
Evidence range: pHash distance 4–10; ORB/RANSAC inliers 142–189; inlier fraction 0.93–0.97.
- `train/images/IMG_20250822_231200_jpg.rf.0f6db813145281c72c353be21cbd72aa.jpg`; timestamp `20250822_231200`; SHA-256 `aa5845f5a51c…`
- `train/images/IMG_20250822_231202_jpg.rf.1763ce102999536a602ed04b81bf0435.jpg`; timestamp `20250822_231202`; SHA-256 `ab4487157d90…`
- `valid/images/IMG_20250822_231201_1_jpg.rf.3ee4562ffda16d4de26bfe881d3c0a8b.jpg`; timestamp `20250822_231201`; SHA-256 `d5b7afa03359…`
- `valid/images/IMG_20250822_231201_jpg.rf.44bc2d8fc2946f343332124c030baa7e.jpg`; timestamp `20250822_231201`; SHA-256 `b5e2f2338495…`

### ND-16 — Mobile; 3 images; splits T/V/R=1/1/1; 2 retained cross-split links
Evidence range: pHash distance 6–10; ORB/RANSAC inliers 69–96; inlier fraction 0.79–0.81.
- `test/images/IMG_20250803_220025_jpg.rf.0c5fa936f319339057069716325911b2.jpg`; timestamp `20250803_220025`; SHA-256 `83138498e082…`
- `train/images/IMG_20250803_220022_jpg.rf.dc0fdee22f504920709e6321bf117c22.jpg`; timestamp `20250803_220022`; SHA-256 `4d55281add1b…`
- `valid/images/IMG_20250803_220024_jpg.rf.2fc166ace133b8664434156a034ba786.jpg`; timestamp `20250803_220024`; SHA-256 `b06d4ffb4311…`

### ND-17 — Mobile; 5 images; splits T/V/R=1/2/2; 4 retained cross-split links
Evidence range: pHash distance 4–10; ORB/RANSAC inliers 108–197; inlier fraction 0.86–0.94.
- `test/images/IMG_20250803_220034_jpg.rf.ff0c9e529ddd3dfcdb4934dc21b66ca1.jpg`; timestamp `20250803_220034`; SHA-256 `cfa4eb0ad258…`
- `test/images/IMG_20250803_220038_jpg.rf.7649ac00470edad7f06ef522f2ffa0ae.jpg`; timestamp `20250803_220038`; SHA-256 `3afee53c4651…`
- `train/images/IMG_20250803_220030_jpg.rf.1f79ea661a57dc1bd9c74db29adb4ec2.jpg`; timestamp `20250803_220030`; SHA-256 `2755f6405591…`
- `valid/images/IMG_20250803_220031_jpg.rf.980993866f095141d958d78983b61f11.jpg`; timestamp `20250803_220031`; SHA-256 `28bf1be6f6bf…`
- `valid/images/IMG_20250803_220037_jpg.rf.a1e1878bef855197a7260cca88694e51.jpg`; timestamp `20250803_220037`; SHA-256 `134dc9a0d3b0…`

### ND-18 — Mobile; 3 images; splits T/V/R=2/0/1; 2 retained cross-split links
Evidence range: pHash distance 4–4; ORB/RANSAC inliers 302–305; inlier fraction 0.97–0.98.
- `test/images/IMG_20250803_220045_jpg.rf.337135edaed79d430b95b610ec32c0db.jpg`; timestamp `20250803_220045`; SHA-256 `94a465a6621b…`
- `train/images/IMG_20250803_220046_jpg.rf.593a44abf20cee25f06845f455285742.jpg`; timestamp `20250803_220046`; SHA-256 `9e91c5b5ff27…`
- `train/images/IMG_20250803_220047_jpg.rf.f69036399cb844d4fa35545cbe8bf18c.jpg`; timestamp `20250803_220047`; SHA-256 `c2ea68e2e2c9…`

### ND-19 — Mobile; 2 images; splits T/V/R=1/0/1; 1 retained cross-split links
Evidence range: pHash distance 10–10; ORB/RANSAC inliers 209–209; inlier fraction 0.95–0.95.
- `test/images/IMG_20250803_220051_jpg.rf.39ebee3d80d991a2fd3bd1e036ae9b05.jpg`; timestamp `20250803_220051`; SHA-256 `6234bebca89c…`
- `train/images/IMG_20250803_220053_jpg.rf.143f84f401009a4e0c09c5b7c33c1ea3.jpg`; timestamp `20250803_220053`; SHA-256 `da09aa15eb1b…`

### ND-20 — Mobile; 2 images; splits T/V/R=1/0/1; 1 retained cross-split links
Evidence range: pHash distance 6–6; ORB/RANSAC inliers 176–176; inlier fraction 0.95–0.95.
- `test/images/IMG_20250803_220138_jpg.rf.c32f7dc03280433f7176d3be9f2d22a3.jpg`; timestamp `20250803_220138`; SHA-256 `b9f4ead353bd…`
- `train/images/IMG_20250803_220139_jpg.rf.36b05a61cd0dc1dc31f88bba93e1bb4a.jpg`; timestamp `20250803_220139`; SHA-256 `656090beebb7…`

### ND-21 — Mobile; 3 images; splits T/V/R=0/2/1; 2 retained cross-split links
Evidence range: pHash distance 2–4; ORB/RANSAC inliers 196–207; inlier fraction 0.92–0.95.
- `test/images/IMG_20250803_220146_jpg.rf.c669407739cff140d229870182988583.jpg`; timestamp `20250803_220146`; SHA-256 `424da110c726…`
- `valid/images/IMG_20250803_220148_jpg.rf.a4e62c59a53971b3d4a9abbcf8b459b1.jpg`; timestamp `20250803_220148`; SHA-256 `81ad0efdfd49…`
- `valid/images/IMG_20250803_220149_jpg.rf.ea36148ba2419eaf41dc5822d2e5d88a.jpg`; timestamp `20250803_220149`; SHA-256 `45ea5fe1987a…`

### ND-22 — Mobile; 3 images; splits T/V/R=1/1/1; 3 retained cross-split links
Evidence range: pHash distance 0–2; ORB/RANSAC inliers 156–224; inlier fraction 0.95–0.97.
- `test/images/IMG_20250803_220228_jpg.rf.8e94e7ce7153325b8e181e6a1d331ef0.jpg`; timestamp `20250803_220228`; SHA-256 `f9c508ba5bc0…`
- `train/images/IMG_20250803_220226_jpg.rf.ff911118579e3a71da019d386eb1d7eb.jpg`; timestamp `20250803_220226`; SHA-256 `24d1e06d8aca…`
- `valid/images/IMG_20250803_220227_jpg.rf.c42373ef7c64d08b36804cc7fcad4d15.jpg`; timestamp `20250803_220227`; SHA-256 `5e2ff2087289…`

### ND-23 — Mobile; 3 images; splits T/V/R=2/0/1; 2 retained cross-split links
Evidence range: pHash distance 0–4; ORB/RANSAC inliers 157–172; inlier fraction 0.96–0.97.
- `test/images/IMG_20250803_220244_jpg.rf.25e694fad0069d6b3720bcfeff342f09.jpg`; timestamp `20250803_220244`; SHA-256 `552cc1b0e608…`
- `train/images/IMG_20250803_220242_jpg.rf.c47cf376f90ec8109c0115c6ef00398e.jpg`; timestamp `20250803_220242`; SHA-256 `5031896cc979…`
- `train/images/IMG_20250803_220243_jpg.rf.c17d27aa58b11aa0475d5b41c8ddbd97.jpg`; timestamp `20250803_220243`; SHA-256 `81b4ecc7a15e…`

### ND-24 — Mobile; 4 images; splits T/V/R=1/0/3; 3 retained cross-split links
Evidence range: pHash distance 0–2; ORB/RANSAC inliers 322–333; inlier fraction 0.97–0.99.
- `test/images/IMG_20250803_220254_jpg.rf.11410f220a6391f2c4b55aa7acabf966.jpg`; timestamp `20250803_220254`; SHA-256 `c8e00af032b9…`
- `test/images/IMG_20250803_220256_jpg.rf.3b9f89aa439c4e4d73ed04ed18e16668.jpg`; timestamp `20250803_220256`; SHA-256 `116de6b6fb23…`
- `test/images/IMG_20250803_220257_jpg.rf.8553c912c825e21ed3ec17bc2f25ff9a.jpg`; timestamp `20250803_220257`; SHA-256 `30a965434b05…`
- `train/images/IMG_20250803_220255_jpg.rf.3511615b828854235bed04fcf894a5f5.jpg`; timestamp `20250803_220255`; SHA-256 `edabae8eae26…`

### ND-25 — Mobile; 8 images; splits T/V/R=4/3/1; 16 retained cross-split links
Evidence range: pHash distance 2–10; ORB/RANSAC inliers 232–349; inlier fraction 0.96–0.99.
- `test/images/IMG_20250803_220351_jpg.rf.450e818f2d954bda365821a90b9c2a70.jpg`; timestamp `20250803_220351`; SHA-256 `48bb27a8ac77…`
- `train/images/IMG_20250803_220350_jpg.rf.51437498308c9763e0717636cb26c374.jpg`; timestamp `20250803_220350`; SHA-256 `2e60939ea359…`
- `train/images/IMG_20250803_220352_jpg.rf.af73197979f1b112320f3f06a0058c12.jpg`; timestamp `20250803_220352`; SHA-256 `d72e3f9aff2c…`
- `train/images/IMG_20250803_220353_jpg.rf.1a768a733a3016bb320de04418c64568.jpg`; timestamp `20250803_220353`; SHA-256 `4cde723e8d62…`
- `train/images/IMG_20250803_220353_jpg.rf.38ba2d3b1f3eadc387125573185c1d16.jpg`; timestamp `20250803_220353`; SHA-256 `a88bdb5d17ae…`
- `valid/images/IMG_20250803_220348_jpg.rf.2f88a5bd0de4e00950763150d344a288.jpg`; timestamp `20250803_220348`; SHA-256 `09fc38f1a755…`
- `valid/images/IMG_20250803_220349_1_jpg.rf.c2488b27ffdb7b551091ec1aff857cce.jpg`; timestamp `20250803_220349`; SHA-256 `370b6e78ff42…`
- `valid/images/IMG_20250803_220349_jpg.rf.9a2cadc69014490f60e59e1826cdc947.jpg`; timestamp `20250803_220349`; SHA-256 `3c7731d06f66…`

### ND-26 — Mobile; 3 images; splits T/V/R=1/1/1; 3 retained cross-split links
Evidence range: pHash distance 0–2; ORB/RANSAC inliers 333–341; inlier fraction 0.99–0.99.
- `test/images/IMG_20250805_203323_jpg.rf.a182df57cbdc73a0a473a3e488e1725f.jpg`; timestamp `20250805_203323`; SHA-256 `211729b6870b…`
- `train/images/IMG_20250805_203321_jpg.rf.fda2603315b3ef543f6b3a2d3c796074.jpg`; timestamp `20250805_203321`; SHA-256 `238213d26635…`
- `valid/images/IMG_20250805_203322_jpg.rf.55299a747169161eadaecc211e423a31.jpg`; timestamp `20250805_203322`; SHA-256 `838cdce65d6a…`

### ND-27 — Mobile; 3 images; splits T/V/R=0/2/1; 2 retained cross-split links
Evidence range: pHash distance 4–6; ORB/RANSAC inliers 336–354; inlier fraction 0.99–0.99.
- `test/images/IMG_20250805_203327_1_jpg.rf.fa4eadf62370e6bdea8c64d3f20085bf.jpg`; timestamp `20250805_203327`; SHA-256 `97ea83615b3c…`
- `valid/images/IMG_20250805_203327_jpg.rf.32be60b70e6f9a50917a48881f5ab85a.jpg`; timestamp `20250805_203327`; SHA-256 `466ebef46446…`
- `valid/images/IMG_20250805_203328_jpg.rf.593c94263769d9d875503af03e97d8f7.jpg`; timestamp `20250805_203328`; SHA-256 `5e31786d8e82…`

### ND-28 — Mobile; 4 images; splits T/V/R=2/0/2; 4 retained cross-split links
Evidence range: pHash distance 0–4; ORB/RANSAC inliers 297–322; inlier fraction 0.98–1.00.
- `test/images/IMG_20250805_203336_jpg.rf.ab9901e1027c275718c876cd3fd594e8.jpg`; timestamp `20250805_203336`; SHA-256 `596c6f3483bf…`
- `test/images/IMG_20250805_203338_1_jpg.rf.bf2025683e05750526dc29fc829b8ac5.jpg`; timestamp `20250805_203338`; SHA-256 `c9be20972a6d…`
- `train/images/IMG_20250805_203337_jpg.rf.619c09a827a0821443b40ad63fcaba9c.jpg`; timestamp `20250805_203337`; SHA-256 `9ee3979a4439…`
- `train/images/IMG_20250805_203338_jpg.rf.68cadbff708a1a27b20e57a030073a53.jpg`; timestamp `20250805_203338`; SHA-256 `621bdbb570e6…`

### ND-29 — Mobile; 4 images; splits T/V/R=2/1/1; 5 retained cross-split links
Evidence range: pHash distance 2–6; ORB/RANSAC inliers 255–353; inlier fraction 0.97–0.99.
- `test/images/IMG_20250805_203805_jpg.rf.ecf4905797f5b1dbc5887402d335664a.jpg`; timestamp `20250805_203805`; SHA-256 `5dd9cb0ed970…`
- `train/images/IMG_20250805_203804_jpg.rf.d0d25dd17b163cc60b1a269ea27346dd.jpg`; timestamp `20250805_203804`; SHA-256 `7080b62cac04…`
- `train/images/IMG_20250805_203806_jpg.rf.7fd019ac3a6efc21e4b37227c5bb73e1.jpg`; timestamp `20250805_203806`; SHA-256 `606cbffd4202…`
- `valid/images/IMG_20250805_203803_jpg.rf.46ed623c40ba0974562c6bbf6953d2cf.jpg`; timestamp `20250805_203803`; SHA-256 `3162b7ac6aa6…`

### ND-30 — Mobile; 3 images; splits T/V/R=1/1/1; 3 retained cross-split links
Evidence range: pHash distance 0–8; ORB/RANSAC inliers 273–351; inlier fraction 0.98–0.99.
- `test/images/IMG_20250805_203845_jpg.rf.8c15db31b863733a22036fd715b6cc3a.jpg`; timestamp `20250805_203845`; SHA-256 `ace6eea4f3d7…`
- `train/images/IMG_20250805_203846_jpg.rf.66d7b40dd70798c0e6a723d0bdc26269.jpg`; timestamp `20250805_203846`; SHA-256 `626d2f8d3b4f…`
- `valid/images/IMG_20250805_203847_jpg.rf.0b7b578193561a5e7b7297568d1862c0.jpg`; timestamp `20250805_203847`; SHA-256 `a2ee57fdae5f…`

### ND-31 — Mobile; 2 images; splits T/V/R=1/0/1; 1 retained cross-split links
Evidence range: pHash distance 6–6; ORB/RANSAC inliers 274–274; inlier fraction 0.96–0.96.
- `test/images/IMG_20250805_203918_jpg.rf.0320c79bd3505204e3c7b6d35a1d716a.jpg`; timestamp `20250805_203918`; SHA-256 `acc30670db2b…`
- `train/images/IMG_20250805_203917_jpg.rf.41ede18eba5e4814a68b0143d4b637b7.jpg`; timestamp `20250805_203917`; SHA-256 `f717080a8bd2…`

### ND-32 — Mobile; 4 images; splits T/V/R=0/3/1; 3 retained cross-split links
Evidence range: pHash distance 2–6; ORB/RANSAC inliers 270–314; inlier fraction 0.97–0.99.
- `test/images/IMG_20250805_203950_jpg.rf.e20c8cf0ae6b59ee38b2a9c2dadbe0eb.jpg`; timestamp `20250805_203950`; SHA-256 `1a1f0c1b6fa1…`
- `valid/images/IMG_20250805_203947_jpg.rf.d549f89d842b21852845f82bff9abb09.jpg`; timestamp `20250805_203947`; SHA-256 `c103adba30da…`
- `valid/images/IMG_20250805_203948_jpg.rf.dbcb8f50ff9c3d8ab30e1018b3995fa7.jpg`; timestamp `20250805_203948`; SHA-256 `6eb9a21feb55…`
- `valid/images/IMG_20250805_203949_jpg.rf.c78fa0f75f1aea8e269f90a2c79abeaa.jpg`; timestamp `20250805_203949`; SHA-256 `f804eb20d21b…`

### ND-33 — Mobile; 3 images; splits T/V/R=1/2/0; 2 retained cross-split links
Evidence range: pHash distance 2–10; ORB/RANSAC inliers 50–179; inlier fraction 0.70–0.97.
- `train/images/IMG_20250803_220114_jpg.rf.68d25d49b60c21e990f42a77571309d7.jpg`; timestamp `20250803_220114`; SHA-256 `9b5f703740f7…`
- `valid/images/IMG_20250803_220117_jpg.rf.0594f79bcf0bff0635472395fec31752.jpg`; timestamp `20250803_220117`; SHA-256 `e88770d021f1…`
- `valid/images/IMG_20250803_220119_jpg.rf.64e2198d7a2d24e7786b2bbc40de885d.jpg`; timestamp `20250803_220119`; SHA-256 `fb1aa4c43348…`

### ND-34 — Mobile; 2 images; splits T/V/R=1/1/0; 1 retained cross-split links
Evidence range: pHash distance 6–6; ORB/RANSAC inliers 309–309; inlier fraction 0.98–0.98.
- `train/images/IMG_20250803_220207_jpg.rf.0f17df4cec5b7117887884be5ea29430.jpg`; timestamp `20250803_220207`; SHA-256 `d017b90375af…`
- `valid/images/IMG_20250803_220206_jpg.rf.de998e9b904dcfe77468124a9e729352.jpg`; timestamp `20250803_220206`; SHA-256 `5d3cdb87393b…`

### ND-35 — Mobile; 4 images; splits T/V/R=3/1/0; 3 retained cross-split links
Evidence range: pHash distance 0–2; ORB/RANSAC inliers 234–254; inlier fraction 0.97–0.97.
- `train/images/IMG_20250805_203809_jpg.rf.d485d98923b15ece68c49e6175f4a60c.jpg`; timestamp `20250805_203809`; SHA-256 `fbe6db9cb292…`
- `train/images/IMG_20250805_203810_jpg.rf.a2d06f67c179d20800e307f90c6fc2b4.jpg`; timestamp `20250805_203810`; SHA-256 `2a8f12395e2e…`
- `train/images/IMG_20250805_203811_jpg.rf.699b0103080853f83b5e914cd660a1b6.jpg`; timestamp `20250805_203811`; SHA-256 `824332baf1dc…`
- `valid/images/IMG_20250805_203812_jpg.rf.41c07d004c67e427803ca860bc03612b.jpg`; timestamp `20250805_203812`; SHA-256 `68c251c62066…`

## Split-crossing conclusion

- **20 timestamp-associated groups** cross the original split boundaries. These include train↔valid, train↔test, valid↔test and, in some groups, all three splits.
- **35 visual-similarity candidate components** cross split boundaries, containing 142 images. In the component inventory, split members occur across train/validation/test combinations; this is strong evidence that the original image-level split is not independent with respect to related captures for these two classes.
- Exact duplicates were not found, which does not remove leakage risk: visually related captures and augmentation siblings can differ at the byte level.
- Since the archive states that three augmentation versions were made per source image, the original splits should not be treated as clean group-independent splits until source families are reconciled.

## Recommended grouping before new splits

1. Keep all original archive files unchanged. Build a separate candidate-family table with archive-relative path, SHA-256, class id/label, original split, timestamp token, pHash, ORB/RANSAC links, and a human-reviewed family/group id.
2. Have a reviewer inspect each candidate component (especially components spanning original test and train/validation). Confirm whether files are augmentation variants, adjacent captures of one physical item/session, or false similarity links. Timestamp equality should be evidence, not the sole group assignment.
3. Group confirmed source-image augmentation siblings together; where physical device identity is observable, group all views of the same physical device together. Keep uncertain edges flagged and resolve conservatively before assigning splits.
4. Create a new group-aware split only after group review. Preserve any existing locked test cohort untouched; do not move any member of an existing test family into training or validation. Because related captures cross the source splits here, do not claim the archive’s original test split is leakage-free for these classes.
5. Keep label review separate from grouping. Similar-image evidence does not validate `Mobile` or `Battery_Waste` annotations; review labels independently before training.

## Limitations

- The archive has no supplied physical-device IDs, capture-session IDs or per-image augmentation-parent links. Matching timestamps do not prove a shared physical device.
- The perceptual and keypoint thresholds are screening thresholds, not a trained duplicate classifier. They can over-group a repeated scene/background or under-detect large crops, rotations, or unrelated views.
- No images were modified or extracted. This report does not assess whether the labels are semantically correct, and it makes no model-performance claims.
