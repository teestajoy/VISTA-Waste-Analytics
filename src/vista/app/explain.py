"""Grad-CAM for ResNet18 (layer4). Hooks are always removed; failures raise instead of returning fake maps."""
import cv2
import numpy as np
import torch
from PIL import Image


class ExplainError(RuntimeError):
    pass


def gradcam(model: torch.nn.Module, batch: torch.Tensor, class_idx: int) -> dict:
    layer = model.layer4
    store: dict = {}

    def fwd(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd)
    try:
        model.eval()
        with torch.enable_grad():
            x = batch.clone().requires_grad_(True)
            logits = model(x)
            score = logits[0, class_idx]
            model.zero_grad(set_to_none=True)
            score.backward()
        if "grad" not in store:
            raise ExplainError("no gradient reached layer4")
        act, grad = store["act"].detach(), store["grad"].detach()
        w = grad.mean(dim=(2, 3), keepdim=True)                      # alpha_k
        cam = torch.relu((w * act).sum(1))[0].numpy()                # ReLU(sum_k alpha_k A^k)
        if not np.isfinite(cam).all():
            raise ExplainError("non-finite Grad-CAM values")
        raw_max = float(cam.max())
        if raw_max <= 0:
            raise ExplainError("Grad-CAM is all zero for this target (degenerate gradients)")
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        cam = cv2.resize(cam, (batch.shape[3], batch.shape[2]), interpolation=cv2.INTER_LINEAR)
        return {"cam": np.clip(cam, 0, 1).astype(np.float32), "target_layer": "layer4 (last residual stage)",
                "normalization": "min-max to [0,1] after ReLU", "target_score": float(score.detach()), "raw_max": raw_max}
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def render(crop: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> tuple[Image.Image, Image.Image]:
    """Return (pure heat-map image, overlay on the crop), both at crop size."""
    rgb = np.asarray(crop.convert("RGB"))
    c = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.cvtColor(cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    mix = (rgb * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(heat), Image.fromarray(mix)


def describe(cam: np.ndarray, thresh: float = 0.6) -> dict:
    ys, xs = np.where(cam >= thresh)
    h, w = cam.shape
    if xs.size == 0:
        return {"summary": "No strongly activated region."}
    cy, cx = ys.mean() / h, xs.mean() / w
    v = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle"
    hz = "left" if cx < 0.4 else "right" if cx > 0.6 else "centre"
    frac = float((cam >= thresh).mean())
    return {"peak_region_xyxy": [round(xs.min() / w, 3), round(ys.min() / h, 3), round((xs.max() + 1) / w, 3), round((ys.max() + 1) / h, 3)],
            "high_activation_area_fraction": round(frac, 3),
            "summary": f"Highest activation is mostly in the {v}-{hz} part of the crop and covers about {frac*100:.0f}% of it. "
                       "Grad-CAM is a coarse spatial cue, not proof of which object component the model used."}
