"""Compare the hierarchical dual-stream pipeline with the flat 9-class baseline on the SAME untouched test split.
Run after:  train_classifier.py --mode stream   and   --mode fine   (the flat baseline is the 'fine' model used without routing)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vista.app.models import PREP, load_resnet
from vista.app.hier import family_of

root = Path(__file__).resolve().parents[1]; ev = root / "models" / "eval"
meta = json.loads((ev / "split.json").read_text()); test = meta["split"]["test"]
fine = load_resnet(root / "models" / "resnet18_fine.pt", "fine")
stream = load_resnet(root / "models" / "resnet18_stream.pt", "stream") if (root / "models" / "resnet18_stream.pt").exists() else None
flat_ok = hier_ok = s_ok = routed_wrong = routed_wrong_final_ok = 0
for p, y in test:
    x = PREP(Image.open(p).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        lf = fine.model(x)[0] / fine.temperature
        pf = torch.softmax(lf, 0)
        if stream:
            ps = torch.softmax(stream.model(x)[0] / stream.temperature, 0); sname = stream.classes[int(ps.argmax())]
        else:
            agg = {"MSW": 0.0, "E_WASTE": 0.0}
            for i, c in enumerate(fine.classes): agg[family_of(c)] += float(pf[i])
            sname = max(agg, key=agg.get)
        mask = torch.tensor([family_of(c) == sname for c in fine.classes])
        ph = torch.softmax(lf.masked_fill(~mask, float("-inf")), 0)
    flat_ok += fine.classes[int(pf.argmax())] == y
    hier_ok += fine.classes[int(ph.argmax())] == y
    ok_stream = sname == family_of(y); s_ok += ok_stream
    if not ok_stream:
        routed_wrong += 1
out = {"n_test": len(test), "stream_accuracy": s_ok / len(test), "flat_accuracy": flat_ok / len(test), "hierarchical_accuracy": hier_ok / len(test),
       "misrouted_samples": routed_wrong, "misrouted_fraction": routed_wrong / len(test), "stream_source": "dedicated" if stream else "provisional aggregate",
       "note": "Hierarchical errors from wrong routing are unrecoverable by construction; compare numbers, do not assume hierarchy wins."}
(ev / "hierarchical_vs_flat").mkdir(exist_ok=True); (ev / "hierarchical_vs_flat" / "metrics.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
