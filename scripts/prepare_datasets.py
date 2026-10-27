"""Merge MWCD + the Custom Bangladeshi E-Waste Image Dataset into one ImageFolder used by train_classifier.py.

  python scripts/prepare_datasets.py --mwcd D:/MWCD --ewaste D:/BD_EWaste --out data/merged [--trashnet data/trashnet]

* MWCD (Mendeley Data, doi 10.17632/x863v66fv3.1, CC BY 4.0): 35,168 images, folders battery/cardboard/glass/metal/organic/paper/plastic/textile/trash.
* Bangladeshi e-waste (2,157 images, 12 classes): Battery Waste, Glass Waste, Keyboard, Light Bulb, Medical Waste, Metal Waste, Mobile, Mouse,
  Organic Waste, Paper Waste, PCB, Plastic Waste.  Folder names are normalised via CLASS_MAP below - EDIT IT if your folder names differ.
* Taxonomies are NOT merged silently: every file is prefixed with its source (mwcd_ / bdew_) and a mapping report is written to
  <out>/MERGE_REPORT.json. Only identical-meaning classes are merged; the stream of each class is fixed in vista.app.hier.E_WASTE_LABELS.
"""
import argparse, json, shutil
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".webp"}
# normalised source-folder name -> target class (None = skip)
CLASS_MAP = {
    "battery": "battery", "battery waste": "battery", "keyboard": "keyboard", "mouse": "mouse", "mobile": "mobile", "pcb": "PCB",
    "light bulb": "light_bulb", "lightbulb": "light_bulb", "medical waste": "medical_waste", "metal waste": "metal", "metal": "metal",
    "glass waste": "glass", "glass": "glass", "organic waste": "organic", "organic": "organic", "paper waste": "paper", "paper": "paper",
    "plastic waste": "plastic", "plastic": "plastic", "cardboard": "cardboard", "textile": "textile", "trash": "trash",
}


def copy_tree(src: Path, prefix: str, out: Path, rep: dict):
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        key = d.name.lower().replace("_", " ").strip()
        tgt = CLASS_MAP.get(key)
        if tgt is None:
            rep["skipped_folders"].append(f"{prefix}{d.name}"); continue
        (out / tgt).mkdir(parents=True, exist_ok=True); n = 0
        for f in d.rglob("*"):
            if f.suffix.lower() in EXT:
                shutil.copy2(f, out / tgt / f"{prefix}{d.name.replace(' ', '')}_{f.name}"); n += 1
        rep["mapping"].append({"source": f"{prefix}{d.name}", "target": tgt, "images": n})


ap = argparse.ArgumentParser()
ap.add_argument("--mwcd"); ap.add_argument("--ewaste"); ap.add_argument("--trashnet"); ap.add_argument("--out", default="data/merged"); a = ap.parse_args()
out = Path(a.out); rep = {"mapping": [], "skipped_folders": [], "warnings": []}
if a.mwcd: copy_tree(Path(a.mwcd), "mwcd_", out, rep)
if a.ewaste: copy_tree(Path(a.ewaste), "bdew_", out, rep)
if a.trashnet:
    rep["warnings"].append("TrashNet is already inside MWCD: adding it again duplicates images and leaks train/test.")
    copy_tree(Path(a.trashnet), "tn_", out, rep)
if not (a.mwcd or a.ewaste or a.trashnet): raise SystemExit("give at least one of --mwcd / --ewaste / --trashnet")
out.mkdir(parents=True, exist_ok=True); (out / "MERGE_REPORT.json").write_text(json.dumps(rep, indent=2))
for c in sorted(p for p in out.iterdir() if p.is_dir()): print(f"{c.name:15s}{sum(1 for _ in c.iterdir()):6d}")
print("Review MERGE_REPORT.json (ambiguous: metal, medical_waste, light_bulb) before training.")
