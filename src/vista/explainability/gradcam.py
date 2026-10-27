"""Grad-CAM explanations for compatible image classifiers.

The heatmap indicates image regions that influenced a selected model logit. It
does not establish that the model's classification is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class GradCAMError(RuntimeError):
    """Raised when Grad-CAM cannot be computed for the supplied model or image."""


@dataclass(frozen=True)
class GradCAMResult:
    """Normalized heatmap and its RGB overlay for a model output class."""

    target_class_index: int
    target_class: str
    heatmap: np.ndarray
    overlay: Image.Image


def generate_gradcam(classifier, image: Image.Image, *, alpha: float = 0.42) -> GradCAMResult:
    """Generate Grad-CAM using a classifier's exact transform and final ResNet block.

    The classifier is expected to expose ``model``, ``transform``, ``device``
    and ``class_names``. The project's provisional category classifier provides
    those attributes and uses the same RGB resize/normalization as training.
    Hooks and every module's training flag are restored even when computation
    fails.
    """
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise ValueError("Grad-CAM requires a non-empty Pillow image.")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("alpha must be a number between 0 and 1.")

    model = getattr(classifier, "model", None)
    transform = getattr(classifier, "transform", None)
    device = getattr(classifier, "device", None)
    class_names = tuple(getattr(classifier, "class_names", ()))
    target_layer = getattr(model, "layer4", None)
    if not isinstance(model, nn.Module) or not isinstance(target_layer, nn.Module):
        raise GradCAMError("Grad-CAM supports models with a final convolutional layer4 block, such as ResNet18.")
    if not callable(transform) or device is None or not class_names:
        raise GradCAMError("The classifier must provide its trained transform, device, and class names.")

    try:
        input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    except Exception as error:
        raise GradCAMError(f"Could not preprocess the image for Grad-CAM: {error}") from error

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def capture_activation(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise GradCAMError("The model's layer4 output must be a four-dimensional feature map.")
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    module_states = {module: module.training for module in model.modules()}
    handle = target_layer.register_forward_hook(capture_activation)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.shape[0] != 1
                or logits.shape[1] != len(class_names)
            ):
                raise GradCAMError("The classifier must return one batch of class logits.")
            target_index = int(logits[0].argmax().item())
            if target_index >= len(class_names):
                raise GradCAMError("The model output index is missing from the saved class mapping.")
            logits[0, target_index].backward()

        if not activations or not gradients:
            raise GradCAMError("Gradients were not captured from the model's layer4 block.")
        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(image.height, image.width), mode="bilinear", align_corners=False
        )[0, 0]
        minimum = cam.min()
        maximum = cam.max()
        if not torch.isfinite(cam).all():
            raise GradCAMError("The model produced a non-finite Grad-CAM heatmap.")
        if float((maximum - minimum).item()) > 1e-12:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = torch.zeros_like(cam)
        heatmap = cam.cpu().numpy().astype(np.float32, copy=False)
        color_bgr = cv2.applyColorMap(
            np.clip(heatmap * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        original = image.convert("RGB")
        color_image = Image.fromarray(color_rgb, mode="RGB")
        overlay = Image.blend(original, color_image, float(alpha))
        return GradCAMResult(
            target_class_index=target_index,
            target_class=str(class_names[target_index]),
            heatmap=heatmap,
            overlay=overlay,
        )
    except GradCAMError:
        raise
    except Exception as error:
        raise GradCAMError(f"Grad-CAM computation failed: {error}") from error
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        for module, was_training in module_states.items():
            module.training = was_training
