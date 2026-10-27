"""Hierarchical dual-stream classification: stream gate first, then a stream-restricted fine-grained head.

Stream stage
  * dedicated 2-class ResNet18 if models/resnet18_stream.pt exists;
  * otherwise PROVISIONAL: the fine-grained softmax mass summed per family (clearly labelled in the output).
Fine stage
  * the stream decision restricts the label set (masked softmax over that stream's classes) so an item routed
    to E-waste can never come out as 'paper'.  Separate MSW / E-waste head checkpoints are used when present.
"""
import torch

from vista.app.models import PREP, STREAMS, Loaded, ModelManager
from vista.classification.loader import FOLDER_TO_CATEGORY

E_WASTE_LABELS = {"metal", "battery", "PCB", "mobile", "laptop", "cable", "component",
                  "keyboard", "mouse", "light_bulb", "medical_waste"}   # last four come from the Bangladeshi e-waste dataset (see scripts/prepare_datasets.py)
# Rule-based first-generation gate: COCO labels of the generic detector that are electronics.
RULE_EWASTE_COCO = {"cell phone", "laptop", "keyboard", "mouse", "remote", "tv", "microwave", "oven", "toaster", "refrigerator", "hair drier"}


def family_of(label: str) -> str:
    return "E_WASTE" if FOLDER_TO_CATEGORY.get(label, label) in E_WASTE_LABELS else "MSW"


def display(label: str) -> str:
    return FOLDER_TO_CATEGORY.get(label, label)


class ClassificationUnavailable(RuntimeError):
    pass


def _probs(m: Loaded, batch: torch.Tensor, mask: list[bool] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = m.model(batch)[0] / m.temperature
    if mask is not None:
        logits = logits.masked_fill(~torch.tensor(mask), float("-inf"))
    return torch.softmax(logits, 0), logits


def classify(crop, mgr: ModelManager, tau_fine: float, tau_stream: float, det_label: str | None = None, gate_mode: str | None = None) -> dict:
    from vista.app import config
    gate_mode = gate_mode or config.GATE_MODE
    batch = PREP(crop.convert("RGB")).unsqueeze(0)
    fine_full = None
    # ---- stage 1: stream -------------------------------------------------------------------
    rule_hit = det_label is not None and gate_mode in ("rule", "auto") and mgr.stream is None and \
        (gate_mode == "rule" or det_label in RULE_EWASTE_COCO)
    if mgr.fine is not None:
        fine_full, _ = _probs(mgr.fine, batch)
    if rule_hit:
        is_e = det_label in RULE_EWASTE_COCO
        sp = {"MSW": 0.0 if is_e else 1.0, "E_WASTE": 1.0 if is_e else 0.0}
        stream_source, stream_cal = "rule_based_detector_label", False
    elif mgr.stream is not None and gate_mode != "rule":
        p, _ = _probs(mgr.stream, batch)
        sp = {c: float(p[i]) for i, c in enumerate(mgr.stream.classes)}
        stream_source = "dedicated_stream_model"
        stream_cal = mgr.stream.temperature != 1.0
    elif mgr.fine is not None:
        sp = {s: 0.0 for s in STREAMS}
        for i, c in enumerate(mgr.fine.classes):
            sp[family_of(c)] += float(fine_full[i])
        stream_source, stream_cal = "provisional_aggregate_of_fine_probabilities", mgr.fine.temperature != 1.0
    else:
        raise ClassificationUnavailable("No classifier loaded: train/provide models/resnet18_fine.pt (or a stream model plus both heads).")
    stream = max(sp, key=sp.get)
    # ---- stage 2: routed fine-grained head ---------------------------------------------------
    head, idx_model = None, None
    if mgr.fine is not None:
        head = mgr.fine
        mask = [family_of(c) == stream for c in head.classes]
        if not any(mask):
            raise ClassificationUnavailable(f"The fine-grained model has no classes for stream {stream}; "
                                            "train the E-waste head (see README) or supply it as models/resnet18_ewaste.pt.")
        fp, _ = _probs(head, batch, mask)
        head_name = "combined (masked to routed stream)"
    else:
        head = mgr.msw if stream == "MSW" else mgr.ewaste
        if head is None:
            raise ClassificationUnavailable(f"Routed to {stream} but its head checkpoint is missing.")
        fp, _ = _probs(head, batch)
        head_name = f"{stream} head"
    top = fp.topk(min(3, int((fp > 0).sum())))
    labels = [head.classes[int(i)] for i in top.indices]
    confs = [float(v) for v in top.values]
    full = {display(head.classes[i]): round(float(fp[i]), 5) for i in range(len(head.classes)) if float(fp[i]) > 0}
    reasons = []
    if confs[0] < tau_fine:
        reasons.append(f"fine-grained confidence {confs[0]:.2f} < {tau_fine:.2f}")
    if sp[stream] < tau_stream:
        reasons.append(f"stream confidence {sp[stream]:.2f} < {tau_stream:.2f}")
    return {
        "stream": {"prediction": stream, "confidence": round(sp[stream], 5), "probabilities": {k: round(v, 5) for k, v in sp.items()},
                   "source": stream_source, "calibrated": stream_cal,
                   "note": {"rule_based_detector_label": "rule gate: detector COCO label mapped to a stream (confidence is a rule outcome, not a probability)", "dedicated_stream_model": "learned gate: dedicated ResNet18 stream classifier"}.get(stream_source, "provisional: derived from fine-grained probabilities, not a dedicated stream model")},
        "fine": {"prediction": display(labels[0]), "raw_label": labels[0], "confidence": round(confs[0], 5), "probabilities": full,
                 "top_k": [[display(l), round(c, 5)] for l, c in zip(labels, confs)],
                 "alternatives": [[display(l), round(c, 5)] for l, c in zip(labels[1:], confs[1:])],
                 "head": head_name, "calibrated": head.temperature != 1.0},
        "uncertain": bool(reasons), "uncertainty_reasons": reasons,
        "thresholds": {"fine_tau": tau_fine, "stream_tau": tau_stream},
        "_ctx": {"head": head, "class_idx": int(top.indices[0]), "batch": batch, "labels": head.classes,
                 "mask_family": stream if mgr.fine is not None else None},
    }
