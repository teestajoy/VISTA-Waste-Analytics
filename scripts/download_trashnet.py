"""Download TrashNet (Thung & Yang, 2016; resized 512x384 version, MIT-licensed) and unpack it as an ImageFolder.
  python scripts/download_trashnet.py --out data/trashnet
Classes: cardboard, glass, metal, paper, plastic, trash (2,527 images). NOTE: TrashNet is already one of the four sources inside
MWCD, so do not add it on top of MWCD unless you de-duplicate (prepare_datasets.py does not add it by default)."""
import argparse, io, urllib.request, zipfile
from pathlib import Path

URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/trashnet"); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
print("downloading", URL)
data = urllib.request.urlopen(URL, timeout=120).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for n in z.namelist():
        parts = Path(n).parts
        if n.endswith("/") or "__MACOSX" in n or len(parts) < 2 or Path(n).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        dest = out / parts[-2] / parts[-1]; dest.parent.mkdir(exist_ok=True); dest.write_bytes(z.read(n))
n = sum(1 for _ in out.rglob("*.jpg"))
print(f"done: {n} images in {out}", sorted(p.name for p in out.iterdir() if p.is_dir()))
