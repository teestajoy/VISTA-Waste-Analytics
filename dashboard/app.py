"""VISTA dashboard (Streamlit). Run: PYTHONPATH=src streamlit run dashboard/app.py
Set VISTA_CHECKPOINT to a trained ResNet18 .pt/.pth to enable classification and Grad-CAM."""

import html
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vista.classification.loader import WasteClassifier  # noqa: E402
from vista.explainability.gradcam import overlay  # noqa: E402
from vista.pipeline import STAGES, analyze, strip_internal  # noqa: E402

st.set_page_config(page_title="VISTA · Waste Intelligence", page_icon="◐", layout="wide")
st.markdown(f"<style>{(ROOT / 'dashboard' / 'style.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_classifier():
    try:
        return WasteClassifier.from_env(), None
    except Exception as error:  # surfaced in the UI, never hidden
        return None, str(error)


@st.cache_resource(show_spinner=False)
def load_detector():
    try:
        from vista.detection.object_detector import YoloObjectDetector
        weights = os.environ.get("VISTA_YOLO", "yolo26n.pt")
        return YoloObjectDetector(weights), None
    except Exception as error:
        return None, str(error)


def detect_crops(image: Image.Image, detector):
    """Return [(label, confidence, crop)]. Falls back to the full frame when no detector/box."""
    if detector is None:
        return [("full frame", None, image)]
    from vista.detection.object_cropper import crop_detection
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
    try:
        dets = detector.detect(tmp.name)
    finally:
        os.unlink(tmp.name)
    crops = [(d.class_label, d.confidence, crop_detection(image, d)) for d in dets[:6]]
    return crops or [("full frame (no box)", None, image)]


clf, clf_err = load_classifier()
det, det_err = load_detector()
chip = lambda on, text: f'<span class="chip {"on" if on else "off"}"><i></i>{text}</span>'

st.markdown(f"""
<div class="topbar"><div class="brand"><div class="mark"></div><b>VISTA</b><span>waste intelligence</span></div>
<div class="chips">{chip(det is not None, "YOLO DETECTOR")}{chip(clf is not None, "RESNET18 CLASSIFIER")}
<span class="chip on"><i></i>EXPERIMENTAL</span></div></div>
<div class="hero"><h1>Waste, read <em>in full.</em></h1>
<p>Drop in a photo. VISTA finds the item, routes it to the right stream, scores how clean and intact it is,
estimates the carbon and money in recycling it, and shows you the pixels behind the decision.</p></div>
""", unsafe_allow_html=True)

left, right = st.columns([1.35, 1], gap="large")

with left:
    st.markdown('<div class="lbl"><b>01</b> Capture</div>', unsafe_allow_html=True)
    tab_up, tab_cam = st.tabs(["UPLOAD", "CAMERA"])
    with tab_up:
        up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    with tab_cam:
        cam = st.camera_input("Camera", label_visibility="collapsed")
    source = up or cam
    image = None
    if source is not None:
        try:
            image = Image.open(io.BytesIO(source.getvalue()))
            image.load()
            image = image.convert("RGB")
        except Exception:
            st.error("That file could not be read as an image.")
    if image is not None:
        st.image(image, use_container_width=True)
    go = st.button("Analyse item", type="primary", use_container_width=True, disabled=image is None)

state = st.session_state
if go and image is not None:
    with st.spinner("Running pipeline…"):
        crops = detect_crops(image, det)
        state["results"] = [(lab, conf, crop, analyze(crop, clf)) for lab, conf, crop in crops]
        state["detector_used"] = det is not None

results = state.get("results")


def rail(res):
    done = {"Validate": res is not None, "Detect": res is not None and state.get("detector_used", False),
            "Crop": res is not None, "Route": res is not None and res[3]["classification"] is not None,
            "Classify": res is not None and res[3]["classification"] is not None, "Quality": res is not None,
            "Impact": res is not None and res[3]["impact"] is not None,
            "Explain": res is not None and res[3]["gradcam"] is not None, "Report": res is not None}
    out = []
    for i, name in enumerate(STAGES, 1):
        cls = "done" if done[name] else ("skip" if res is not None else "")
        tag = "" if res is None else ("ok" if done[name] else "n/a")
        out.append(f'<div class="stage {cls}"><span class="dot"></span>{i:02d} · {name}<span class="t">{tag}</span></div>')
    return "".join(out)


with right:
    st.markdown('<div class="lbl"><b>02</b> Pipeline</div>', unsafe_allow_html=True)
    sel_res = None
    if results:
        sel = 0
        if len(results) > 1:
            names = [f"{i + 1}. {r[0]}" for i, r in enumerate(results)]
            sel = names.index(st.radio("Detected item", names, horizontal=True))
        sel_res = results[sel]
    st.markdown(f'<div class="panel"><div class="rail">{rail(sel_res)}</div>'
                '<div class="note">Dashed amber = stage unavailable in this session (no model loaded).</div></div>',
                unsafe_allow_html=True)
    if sel_res:
        st.markdown(f'<div class="panel"><div class="row"><span>End-to-end latency</span><b>{sel_res[3]["latency_ms"]} ms</b></div></div>',
                    unsafe_allow_html=True)

if not results:
    st.stop()

label, dconf, crop, rep = sel_res
cls_ = rep["classification"]
bar = lambda v, kind="": f'<div class="bar {kind}"><div style="width:{max(2, int(v * 100))}%"></div></div>'
kind = lambda v: "" if v < 0.33 else ("warn" if v < 0.66 else "bad")

st.markdown('<div class="lbl" style="margin-top:1.4rem"><b>03</b> Verdict</div>', unsafe_allow_html=True)
if cls_:
    fam_e = cls_["family"] == "E_WASTE"
    top3 = "".join(f'<div><div class="row"><span>{html.escape(n)}</span><b>{p * 100:.1f}%</b></div>{bar(p)}</div>' for n, p in cls_["top3"])
    st.markdown(f"""<div class="panel"><div class="verdict"><div>
    <span class="fam {'e' if fam_e else ''}">{'E-WASTE' if fam_e else cls_['family']} STREAM</span>
    <div class="cls">{html.escape(str(cls_['class']))}</div>
    <div class="conf">confidence {cls_['confidence'] * 100:.1f}% · raw, uncalibrated softmax</div></div>
    <div class="top3">{top3}</div></div></div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="panel"><div class="pending"><b>Classifier not loaded.</b> Quality scoring below is
    training-free and still runs. To enable family, class, carbon-economic estimates and Grad-CAM, point
    <span class="mono">VISTA_CHECKPOINT</span> at your trained ResNet18 checkpoint and restart the app.</div></div>""",
                unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown('<div class="lbl"><b>04</b> Quality scoring</div>', unsafe_allow_html=True)
    co, cd = rep["contamination"], rep["condition"]
    st.markdown(f"""<div class="panel"><h4>Contamination + condition</h4><div class="sub">Training-free image measurements · 0 best → 1 worst</div>
    <div class="row"><span>Contamination · {co['verdict']}</span><b>{co['score']:.2f}</b></div>{bar(co['score'], kind(co['score']))}
    <div class="note" style="margin:.2rem 0 1rem">HSV saturation variance + Laplacian texture energy</div>
    <div class="row"><span>Condition · {cd['verdict']}</span><b>{cd['score']:.2f}</b></div>{bar(cd['score'], kind(cd['score']))}
    <div class="note" style="margin:.2rem 0 0">Deformation from contour solidity (area ÷ convex-hull area)</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="lbl"><b>05</b> Carbon-economic estimator</div>', unsafe_allow_html=True)
    imp = rep["impact"]
    if imp:
        st.markdown(f"""<div class="panel"><h4>CO₂e + value per item</h4><div class="sub">If this item is recycled instead of discarded</div>
        <div class="metrics"><div class="metric"><small>CO₂e avoided</small><div>{imp['co2e_kg']:.3f}<span>kg</span></div></div>
        <div class="metric"><small>Recovery value</small><div>${imp['value_usd']:.3f}<span>USD</span></div></div></div>
        <div class="formula">CO₂e = f<sub>c</sub> × m<sub>c</sub> &nbsp;·&nbsp; value = p<sub>c</sub> × m<sub>c</sub> &nbsp;·&nbsp; m = {imp['assumed_mass_kg']} kg (assumed)</div>
        <div class="note">Estimate, not a measurement: mass is a class-level assumption and the reference factors are placeholders pending cited life-cycle data.</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel"><h4>CO₂e + value per item</h4><div class="pending">Needs a predicted class. Load a classifier to see the estimate.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="lbl" style="margin-top:.6rem"><b>06</b> Explainability · Grad-CAM</div>', unsafe_allow_html=True)
cam = rep.get("_cam")
if cam is not None:
    g1, g2 = st.columns([1.2, 1], gap="large")
    with g1:
        mode = st.radio("View", ["Heat-map overlay", "Original", "Heat-map only"], horizontal=True, label_visibility="collapsed")
        a = st.slider("Heat-map strength", 0.1, 0.9, 0.55, 0.05)
        if mode == "Original":
            st.image(crop, use_container_width=True)
        elif mode == "Heat-map only":
            st.image(overlay(Image.new("RGB", crop.size, (10, 11, 13)), cam, 1.0), use_container_width=True)
        else:
            st.image(overlay(crop, cam, a), use_container_width=True)
    with g2:
        reg = rep["gradcam"]["peak_region_xyxy"]
        txt = "no dominant region" if reg is None else f"x {reg[0]:.0%}–{reg[2]:.0%}, y {reg[1]:.0%}–{reg[3]:.0%} of the crop"
        st.markdown(f"""<div class="panel"><h4>Why this class?</h4><div class="sub">Region that drove the decision</div>
        <div class="formula">α<sub>k</sub> = mean ∂y<sup>c</sup>/∂A<sup>k</sup> &nbsp;→&nbsp; ReLU(Σ α<sub>k</sub>A<sup>k</sup>)</div>
        <div class="row" style="margin-top:.9rem"><span>Layer</span><b>{rep['gradcam']['layer']}</b></div>
        <div class="row"><span>Strongest region</span><b>{txt}</b></div>
        <div class="note">Warm colours mark pixels that pushed the prediction up. If the glow sits on the background instead of the item, do not trust the label.</div></div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="panel"><div class="pending"><b>Grad-CAM needs a classifier.</b> It is computed on the ResNet18 last convolutional layer, so it appears once VISTA_CHECKPOINT is set.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="lbl" style="margin-top:.6rem"><b>07</b> Per-item report · JSON</div>', unsafe_allow_html=True)
r1, r2 = st.columns([1.2, 1], gap="large")
payload = strip_internal(rep)
payload["detector"] = {"label": label, "confidence": None if dconf is None else round(float(dconf), 4)}
with r1:
    st.json(payload, expanded=2)
    st.download_button("Download report.json", json.dumps(payload, indent=2), "vista_report.json", "application/json")
with r2:
    st.markdown("""<div class="panel"><h4>FastAPI service + Streamlit dashboard</h4><div class="sub">Same pipeline, two front doors</div>
    <div class="api"><span class="m">GET</span>  /health
<span class="m">POST</span> /predict  (multipart image)

&#36; uvicorn vista.api.service:app
&#36; curl -F file=@item.jpg http://localhost:8000/predict</div>
    <div class="note">The dashboard calls the pipeline directly; the API returns this same JSON for other apps and edge devices.</div></div>""", unsafe_allow_html=True)
