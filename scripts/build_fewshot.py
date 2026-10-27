"""Few-shot fallback: ImageNet-pretrained ResNet18 + class prototypes (cosine classifier) from a SMALL support set. NO training involved.

  models/support/<class>/*.jpg   (a handful of images per class; class folders use the same names as the taxonomy)
  python scripts/build_fewshot.py --support models/support --out models/resnet18_fine.pt

The result is a normal checkpoint (backbone = ImageNet weights, fc = normalised class-mean 512-d features), so the app, Grad-CAM and the
hierarchical routing work unchanged. It is flagged 'FEW-SHOT PROTOTYPES (not trained)' in the UI and its accuracy is NOT validated:
use it only until train_classifier.py has produced a properly fine-tuned model (which then replaces this file)."""
import argparse, sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import models
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vista.app.models import PREP

ap = argparse.ArgumentParser(); ap.add_argument("--support", default="models/support"); ap.add_argument("--out", default="models/resnet18_fine.pt"); ap.add_argument("--no-pretrained", action="store_true", help="plumbing test only"); a = ap.parse_args()
m = models.resnet18(weights=None if a.no_pretrained else models.ResNet18_Weights.IMAGENET1K_V1).eval()   # needs internet once (download.pytorch.org)
fc = m.fc; m.fc = torch.nn.Identity()
classes = sorted(d.name for d in Path(a.support).iterdir() if d.is_dir())
protos, n_img = [], 0
for c in classes:
    fs = [f for f in (Path(a.support) / c).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not fs: raise SystemExit(f"class {c} has no images")
    with torch.no_grad():
        feats = torch.cat([m(PREP(Image.open(f).convert("RGB")).unsqueeze(0)) for f in fs]); n_img += len(fs)
    protos.append(feats.mean(0))
P = torch.stack(protos); scale = 10.0 / P.norm(dim=1).mean()
head = torch.nn.Linear(512, len(classes), bias=True); head.weight.data = (P / P.norm(dim=1, keepdim=True)) * scale; head.bias.data.zero_()
m.fc = head
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": m.state_dict(), "classes": classes, "few_shot": True, "support_images": n_img, "version": "few-shot-prototypes"}, a.out)
print(f"saved {a.out}: {len(classes)} classes from {n_img} support images (few-shot, unvalidated)")
