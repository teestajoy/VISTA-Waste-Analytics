"""Grad-CAM for a convolutional classifier (e.g. ResNet18 layer4)."""

import cv2
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model.eval()
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _i, out):
        self._acts = out.detach()

    def _save_grads(self, _m, _gi, go):
        self._grads = go[0].detach()

    def heatmap(self, batch: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a (H, W) map in 0..1 for the chosen (or top) class of a 1-image batch."""
        self.model.zero_grad()
        logits = self.model(batch.requires_grad_(True))
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # alpha_k
        cam = torch.relu((weights * self._acts).sum(dim=1))[0].numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cv2.resize(cam, (batch.shape[3], batch.shape[2]))


def overlay(crop: Image.Image, cam: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend a 0..1 heat-map onto the crop (turbo colour map)."""
    rgb = np.asarray(crop.convert("RGB"))
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    mix = (rgb * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mix)


def top_region(cam: np.ndarray, thresh: float = 0.6) -> tuple[float, float, float, float] | None:
    """Bounding box (x0, y0, x1, y1 as 0..1 fractions) of the strongest activation area."""
    ys, xs = np.where(cam >= thresh)
    if xs.size == 0:
        return None
    h, w = cam.shape
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)
