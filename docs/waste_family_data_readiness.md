# MSW versus E-waste classifier readiness

> Historical snapshot from before the separate folder-derived baseline was
> trained. The current experimental status and results are documented in
> [provisional_family_baseline.md](provisional_family_baseline.md). This does
> not change the status of the human-reviewed, verified family classifier.

## Evidence currently available

The MWCD audit records 35,168 files in nine category folders. These are raw
inventory counts, not verified labels. The taxonomy maps paper, cardboard,
plastic, glass, and organic to MSW and maps battery to E-waste, but every image
still needs human review. Organic means food waste under the initial VISTA
policy. The battery folder's meaning is undocumented. Generic metal is not
evidence of E-waste; trash has not been established as non-recyclable; textile
has no supported VISTA target. PCB, mobile, laptop, cable, and component have
no MWCD source folder.

The paper/cardboard pilot contains 36 explicitly reviewed canonical images
(17 paper and 19 cardboard). Those two categories map to MSW, so they are a
small possible MSW contribution. The separate family-review decision CSV
currently records five rows as `include / E_WASTE` and none as MSW. Those five
decisions still need the preparation workflow's source hash, path, image, and
duplicate-group checks. They do not make a usable family split, and the
paper/cardboard pilot's test set does not evaluate the family task.

The duplicate review documents two unresolved exact-byte groups shared by
plastic and glass. Keep both groups out of any family dataset until labels are
verified. The existing exact-duplicate checker covers five MSW candidate
folders; a future family dataset must hash all included MSW and E-waste sources
together before splitting.

## Conservative source policy

| MWCD folder | Family candidate | Current handling |
|---|---|---|
| paper, cardboard | MSW | Candidate; 36 pilot images have explicit fine labels. |
| plastic, glass | MSW | Candidate after image review; exclude unresolved cross-folder duplicate groups. |
| organic | MSW | Candidate only for food waste, after image review. |
| battery | E-waste | Candidate only after defining qualifying batteries and reviewing each image. |
| metal | None by default | Hold unless review confirms an electronic context. |
| trash | None by default | Hold until a written rule establishes the intended family. |
| textile | None | Exclude under the current taxonomy. |

These are review routes, not automatic folder-to-family labels. Do not derive a
family from a YOLO class name or from an unreviewed source folder.

## Readiness and next step

The current family decision queue has five explicit E_WASTE includes and zero
MSW includes. A two-family classifier cannot be trained or evaluated honestly
yet. Review candidate suggestions are generated separately from human labels;
the review UI supports explicit batch decisions and logs changes. The next data
step is to review representative images from both families, preserve ambiguous
items as pending/held, and then prepare a separate family manifest. Exact
SHA-256 groups must remain in one split, and any group with conflicting family
decisions must be held or excluded.

The dashboard now has a cached ResNet18 family-model loader, but the family
checkpoint and held-out evaluation artifacts do not exist. It therefore still
reports Uncertain. The guarded training script will stop until a prepared
manifest contains both families in train, validation, and test. The
paper/cardboard classifier remains a separate supporting pilot. No family
model was trained in this setup step.
