"""Train + evaluate a ResNet18 classifier on an ImageFolder-style dataset (one sub-folder per class, e.g. MWCD).

  python scripts/train_classifier.py --data D:/MWCD --mode fine      # 9-class (battery, cardboard, ..., trash)
  python scripts/train_classifier.py --data D:/MWCD --mode stream    # MSW vs E-waste gate (labels derived from the taxonomy)
  python scripts/train_classifier.py --data D:/MWCD --mode flat      # same labels as 'fine' but trained/evaluated as the flat baseline

Reproducible: fixed seed, near-duplicate-aware (dHash groups) 70/15/15 split saved to models/eval/split.json.
The TEST split is only used once, at the end. Nothing here invents numbers: metrics are computed from the saved model.
"""
import argparse, json, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vista.app.hier import family_of  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp"}
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_T = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(.2, .2, .2, .02), transforms.RandomRotation(12), transforms.ToTensor(), NORM])
EVAL_T = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


def dhash(path: Path) -> int:
    try:
        a = np.asarray(Image.open(path).convert("L").resize((9, 8)), dtype=np.int16)
    except Exception:
        return -1
    return int("".join("1" if v else "0" for v in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def make_split(root: Path, seed: int, out: Path):
    rng = random.Random(seed)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    items, corrupt = [], 0
    for ci, c in enumerate(classes):
        for p in sorted((root / c).rglob("*")):
            if p.suffix.lower() in EXT:
                h = dhash(p)
                if h < 0:
                    corrupt += 1
                    continue
                items.append((str(p), c, h))
    groups = defaultdict(list)
    for it in items:
        groups[it[2]].append(it)             # identical dHash = near-duplicate group: kept together
    split = {"train": [], "val": [], "test": []}
    by_cls = defaultdict(list)
    for g in groups.values():
        by_cls[g[0][1]].append(g)
    for c, gl in by_cls.items():
        rng.shuffle(gl)
        n = len(gl); a, b = int(.7 * n), int(.85 * n)
        for i, g in enumerate(gl):
            split["train" if i < a else "val" if i < b else "test"].extend((p, cl) for p, cl, _ in g)
    meta = {"seed": seed, "classes": classes, "corrupt_skipped": corrupt, "images": len(items), "duplicate_groups_with_>1": sum(len(g) > 1 for g in groups.values()),
            "strategy": "group-aware (exact dHash) stratified 70/15/15; near-duplicates beyond identical hashes may still leak", "split": split}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta))
    return meta


class DS(Dataset):
    def __init__(self, items, label_fn, classes, tf):
        self.items, self.tf, self.y = items, tf, [classes.index(label_fn(c)) for _, c in items]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        return self.tf(Image.open(self.items[i][0]).convert("RGB")), self.y[i]


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1); acc = (pred == y).astype(float); e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def predict(model, loader, T=1.0):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            P.append(torch.softmax(model(x) / T, 1)); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def fit_temperature(model, loader):
    model.eval(); L, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            L.append(model(x)); Y.append(y)
    L, Y = torch.cat(L), torch.cat(Y)
    t = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([t], lr=.05, max_iter=100)
    def closure():
        opt.zero_grad(); loss = nn.functional.cross_entropy(L / t.clamp(min=.05), Y); loss.backward(); return loss
    opt.step(closure)
    return float(t.clamp(min=.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--mode", choices=["fine", "stream", "flat"], default="fine")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true"); ap.add_argument("--out", default="models"); ap.add_argument("--limit", type=int, default=0, help="debug: cap images per split")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); ev = out / "eval"; sp_path = ev / "split.json"
    meta = json.loads(sp_path.read_text()) if sp_path.exists() else make_split(Path(a.data), a.seed, sp_path)
    folders = meta["classes"]
    label_fn = (lambda c: family_of(c)) if a.mode == "stream" else (lambda c: c)
    classes = ["MSW", "E_WASTE"] if a.mode == "stream" else folders
    sp = {k: (v[:a.limit] if a.limit else v) for k, v in meta["split"].items()}
    dl = {k: DataLoader(DS(v, label_fn, classes, TRAIN_T if k == "train" else EVAL_T), batch_size=a.batch, shuffle=False, num_workers=a.workers) for k, v in sp.items() if k != "train"}
    tr = DS(sp["train"], label_fn, classes, TRAIN_T)
    cnt = np.bincount(tr.y, minlength=len(classes)); w = (1.0 / np.maximum(cnt, 1))[tr.y]
    train_dl = DataLoader(tr, batch_size=a.batch, sampler=WeightedRandomSampler(w, len(tr), generator=torch.Generator().manual_seed(a.seed)), num_workers=a.workers)
    try:
        model = models.resnet18(weights=None if a.no_pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception as e:
        print("WARNING: ImageNet weights unavailable (%s); training from scratch" % e); model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}; best, bad = -1, 0; ck = out / f"resnet18_{a.mode}.pt"; out.mkdir(exist_ok=True)
    for ep in range(a.epochs):
        model.train(); tl = tc = n = 0
        for x, y in train_dl:
            opt.zero_grad(); lo = model(x); loss = nn.functional.cross_entropy(lo, y, label_smoothing=0.05); loss.backward(); opt.step()
            tl += loss.item() * len(y); tc += (lo.argmax(1) == y).sum().item(); n += len(y)
        sched.step(); P, Y = predict(model, dl["val"])
        vl = float(nn.functional.nll_loss(torch.log(torch.tensor(P) + 1e-9), torch.tensor(Y))); va = float((P.argmax(1) == Y).mean())
        for k, v in zip(hist, (tl / n, vl, tc / n, va)): hist[k].append(v)
        print(f"epoch {ep+1}/{a.epochs} train_loss {tl/n:.4f} val_loss {vl:.4f} val_acc {va:.4f}")
        if va > best:
            best, bad = va, 0; torch.save({"state_dict": model.state_dict(), "classes": classes, "val_acc": va, "epochs": ep + 1, "seed": a.seed, "version": a.mode}, ck)
        else:
            bad += 1
            if bad >= a.patience: print("early stop"); break
    state = torch.load(ck, weights_only=False); model.load_state_dict(state["state_dict"])
    T = fit_temperature(model, dl["val"]); state["temperature"] = T; torch.save(state, ck)       # calibration on VAL only
    t0 = time.perf_counter(); P, Y = predict(model, dl["test"], T); lat = (time.perf_counter() - t0) / len(Y) * 1000   # TEST, once
    pred = P.argmax(1); pr, rc, f1, sup = precision_recall_fscore_support(Y, pred, labels=range(len(classes)), zero_division=0)
    m = {"accuracy": float(accuracy_score(Y, pred)), "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()), "macro_f1": float(f1.mean()),
         "weighted_f1": float(precision_recall_fscore_support(Y, pred, average="weighted", zero_division=0)[2]), "ece": ece(P, Y), "temperature": T,
         "brier": float(np.mean(np.sum((P - np.eye(len(classes))[Y]) ** 2, 1))), "latency_ms_per_image_cpu": lat, "model_size_mb": ck.stat().st_size / 1e6,
         "n_test": int(len(Y)), "classes": classes, "confusion_matrix": confusion_matrix(Y, pred, labels=range(len(classes))).tolist(),
         "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])} for i, c in enumerate(classes)},
         "split_note": meta["strategy"], "seed": a.seed}
    try:
        m["roc_auc_ovr_macro"] = float(roc_auc_score(Y, P[:, 1] if len(classes) == 2 else P, multi_class="ovr")) if len(set(Y)) > 1 else None
    except ValueError:
        m["roc_auc_ovr_macro"] = None
    d = ev / a.mode; d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(m, indent=2)); (d / "history.json").write_text(json.dumps(hist))
    np.save(d / "test_probs.npy", P); np.save(d / "test_labels.npy", Y)
    print(json.dumps({k: v for k, v in m.items() if isinstance(v, float)}, indent=1))


if __name__ == "__main__":
    main()
