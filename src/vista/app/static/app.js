"use strict";
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = t => String(t ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 3) => v === null || v === undefined ? "N/A" : (typeof v === "number" ? +v.toFixed(d) : v);
const pct = v => v === null || v === undefined ? "N/A" : (v * 100).toFixed(1) + "%";
const S = { imageId: null, img: null, dets: [], sel: null, rep: null, status: null, cam: null, camTimer: null, camBusy: false, mode: "upload" };
const CATS = ["plastic","paper","cardboard","glass","organic","non-recyclable","textile","metal","battery","PCB","mobile","laptop","cable","component"];

function toast(msg, kind = "ok", ms = 4500) { const t = $("#toast"); t.textContent = msg; t.className = "toast " + kind; t.hidden = false; clearTimeout(toast.h); toast.h = setTimeout(() => t.hidden = true, ms); }
async function api(url, opt) {
  let r; try { r = await fetch(url, opt); } catch (e) { throw new Error("Backend unavailable. Is the server running?"); }
  if (!r.ok) { let m = r.statusText; try { const j = await r.json(); m = j.error || j.detail || m; } catch {} throw new Error(typeof m === "string" ? m : JSON.stringify(m)); }
  return r;
}
const jget = async u => (await api(u)).json();
function go(p) { $$(".page").forEach(e => e.classList.toggle("on", e.id === "p-" + p)); $$("#nav button").forEach(b => b.classList.toggle("on", b.dataset.p === p));
  if (p === "analytics") loadAnalytics(); if (p === "model") loadModel(); if (p === "reports") loadReports(); if (p === "overview") loadStatus(); if (p === "explain" && S.rep) renderExplain(); }
$$("#nav button").forEach(b => b.onclick = () => go(b.dataset.p)); window.go = go;

/* ---------- status ---------- */
async function loadStatus() {
  try {
    const st = S.status = await jget("/api/models/status");
    const items = [["Detector (YOLOv8)", st.detector.ready, st.detector.kind], ["Stream gate", st.stream_classifier.ready || st.fine_classifier.ready, st.stream_classifier.mode],
      ["Fine-grained ResNet18", st.fine_classifier.ready, st.fine_classifier.mode], ["Grad-CAM", st.explainability.ready, st.explainability.method]];
    $("#ovStatus").innerHTML = items.map(([n, ok, d]) => `<div class="card"><b>${esc(n)}</b><div class="${ok ? "" : "muted"}"><span class="tag ${ok ? "msw" : "unc"}">${ok ? "ready" : "not ready"}</span> ${esc(d)}</div>${ok ? "" : `<p class="small muted">${esc(st.fine_classifier.how_to_fix || st.detector.how_to_fix || "")}</p>`}</div>`).join("");
    const ok = st.detector.ready && st.fine_classifier.ready; $("#pill").textContent = ok ? "backend ready" : "backend: some models missing"; $("#pill").className = "pill " + (ok ? "ok" : "bad");
  } catch (e) { $("#pill").textContent = "backend offline"; $("#pill").className = "pill bad"; }
}

/* ---------- input modes ---------- */
$$("#modeSeg button").forEach(b => b.onclick = () => { S.mode = b.dataset.m; $$("#modeSeg button").forEach(x => x.classList.toggle("on", x === b)); $("#upBox").hidden = S.mode !== "upload"; $("#camBox").hidden = S.mode !== "camera"; if (S.mode === "upload") stopCam(); });
$("#conf").oninput = e => $("#confv").textContent = e.target.value;
$("#file").onchange = e => { const f = e.target.files[0]; if (f) submitBlob(f, f.name); };
const drop = $("#upBox"); drop.ondragover = e => { e.preventDefault(); }; drop.ondrop = e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) submitBlob(f, f.name); };

async function submitBlob(blob, name) {
  if (!/\.(jpe?g|png|webp)$/i.test(name)) return toast("Unsupported file type. Use JPEG, PNG or WEBP.", "bad");
  if (blob.size > 10 * 1024 * 1024) return toast("File is larger than 10 MB.", "bad");
  $("#stageWrap").hidden = false; $("#busy").hidden = false; $("#noDet").hidden = true; S.rep = null;
  const fd = new FormData(); fd.append("file", blob, name); fd.append("conf", $("#conf").value);
  const url = URL.createObjectURL(blob);
  try {
    const [res, img] = await Promise.all([(await api("/api/detect", { method: "POST", body: fd })).json(), loadImg(url)]);
    S.imageId = res.image_id; S.img = img; S.dets = res.detections; S.sel = res.detections.length ? 0 : null;
    $("#detInfo").textContent = `${res.detections.length} object(s) · ${res.latency_ms} ms · ${res.detector.kind}`;
    if (res.message) { $("#noDet").textContent = res.message; $("#noDet").hidden = false; }
    drawStage(); renderChips(); $("#analyzeFull").disabled = false; $("#analyzeBtn").disabled = S.sel === null;
  } catch (e) { toast(e.message, "bad"); $("#stageWrap").hidden = !S.img; }
  finally { $("#busy").hidden = true; }
}
const loadImg = u => new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = () => no(new Error("Could not display image")); i.src = u; });
$("#clearImg").onclick = () => { S.img = null; S.imageId = null; S.dets = []; S.sel = null; $("#stageWrap").hidden = true; $("#analyzeBtn").disabled = $("#analyzeFull").disabled = true; $("#file").value = ""; };
$("#showBoxes").onchange = drawStage; $("#conf").onchange = () => { if (S.img && S.mode === "upload") toast("Threshold applies to the next image / capture."); };

function drawStage() {
  if (!S.img) return; const c = $("#stage"), x = c.getContext("2d"); c.width = S.img.width; c.height = S.img.height; x.drawImage(S.img, 0, 0);
  if (!$("#showBoxes").checked) return; const lw = Math.max(2, c.width / 300), fs = Math.max(12, c.width / 50); x.font = `${fs}px sans-serif`;
  S.dets.forEach(d => { const [a, b, e, f] = d.box_xyxy, sel = d.id === S.sel; x.strokeStyle = sel ? "#ff8c00" : "#2878dc"; x.lineWidth = lw + (sel ? 2 : 0); x.strokeRect(a, b, e - a, f - b);
    const t = `#${d.id} ${d.label} ${d.confidence.toFixed(2)}`, w = x.measureText(t).width + 8; x.fillStyle = sel ? "#ff8c00" : "#2878dc"; x.fillRect(a, Math.max(0, b - fs - 4), w, fs + 4); x.fillStyle = "#fff"; x.fillText(t, a + 4, Math.max(fs, b - 5)); });
}
function renderChips() { $("#detList").innerHTML = S.dets.map(d => `<span class="chip ${d.id === S.sel ? "sel" : ""}" data-i="${d.id}">#${d.id} ${esc(d.label)} · ${d.confidence.toFixed(2)}</span>`).join("") || '<span class="muted small">No detections</span>';
  $$("#detList .chip").forEach(c => c.onclick = () => select(+c.dataset.i)); }
function select(i) { S.sel = i; $("#analyzeBtn").disabled = false; drawStage(); renderChips(); }
$("#stage").onclick = e => { const c = e.target, r = c.getBoundingClientRect(), px = (e.clientX - r.left) * c.width / r.width, py = (e.clientY - r.top) * c.height / r.height;
  const hit = S.dets.filter(d => px >= d.box_xyxy[0] && px <= d.box_xyxy[2] && py >= d.box_xyxy[1] && py <= d.box_xyxy[3]).sort((a, b) => (a.box_xyxy[2] - a.box_xyxy[0]) * (a.box_xyxy[3] - a.box_xyxy[1]) - (b.box_xyxy[2] - b.box_xyxy[0]) * (b.box_xyxy[3] - b.box_xyxy[1]));
  if (hit.length) select(hit[0].id); };

/* ---------- camera ---------- */
$("#camStart").onclick = async () => {
  if (!navigator.mediaDevices?.getUserMedia) return toast("Camera API not available (use http://localhost or https).", "bad");
  try { S.cam = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false }); }
  catch (e) { return toast(e.name === "NotAllowedError" ? "Camera permission denied. Allow camera access in the browser." : e.name === "NotFoundError" ? "No camera device found." : "Camera error: " + e.message, "bad", 7000); }
  const v = $("#vid"); v.srcObject = S.cam; await v.play(); $("#camStart").disabled = true; $("#camCap").disabled = $("#camStop").disabled = false; $("#camRetake").disabled = true; liveLoop(true);
};
function liveLoop(on) { clearInterval(S.camTimer); if (on) S.camTimer = setInterval(liveDetect, 2000); }
async function frameBlob() { const v = $("#vid"); if (!v.videoWidth) return null; const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0); return new Promise(r => c.toBlob(r, "image/jpeg", .9)); }
async function liveDetect() {
  if (S.camBusy) return; S.camBusy = true;
  try { const b = await frameBlob(); if (!b) return; const fd = new FormData(); fd.append("file", b, "live.jpg"); fd.append("conf", $("#conf").value); fd.append("live", "true");
    const res = await (await api("/api/detect", { method: "POST", body: fd })).json(); const v = $("#vid"), o = $("#vidOv"); o.width = v.videoWidth; o.height = v.videoHeight; o.style.width = v.clientWidth + "px"; o.style.height = v.clientHeight + "px";
    const x = o.getContext("2d"); x.clearRect(0, 0, o.width, o.height); x.font = `${Math.max(12, o.width / 50)}px sans-serif`; x.lineWidth = Math.max(2, o.width / 300);
    res.detections.forEach(d => { const [a, b2, e, f] = d.box_xyxy; x.strokeStyle = "#2dd4ff"; x.strokeRect(a, b2, e - a, f - b2); x.fillStyle = "#2dd4ff"; x.fillText(`${d.label} ${d.confidence.toFixed(2)}`, a + 3, Math.max(14, b2 - 4)); });
  } catch (e) { toast(e.message, "bad"); liveLoop(false); } finally { S.camBusy = false; }
}
$("#camCap").onclick = async () => { liveLoop(false); $("#vid").pause(); const b = await frameBlob(); $("#camCap").disabled = true; $("#camRetake").disabled = false; if (b) submitBlob(b, "capture.jpg"); };
$("#camRetake").onclick = async () => { await $("#vid").play(); $("#camCap").disabled = false; $("#camRetake").disabled = true; $("#stageWrap").hidden = true; liveLoop(true); };
function stopCam() { liveLoop(false); S.cam?.getTracks().forEach(t => t.stop()); S.cam = null; $("#camStart").disabled = false; $("#camCap").disabled = $("#camStop").disabled = $("#camRetake").disabled = true; const o = $("#vidOv"); o.getContext("2d").clearRect(0, 0, o.width, o.height); }
$("#camStop").onclick = stopCam;

/* ---------- analyze ---------- */
async function runAnalyze(full) {
  if (!S.imageId) return; const body = { image_id: S.imageId, detection_id: full ? null : S.sel }; const m = parseFloat($("#mass").value); if (m > 0) body.mass_kg = m;
  $("#result").innerHTML = `<div class="card"><h3>Analysis</h3><p>Running stream gate → classifier → scorers → Grad-CAM…</p></div>`; $("#analyzeBtn").disabled = true;
  try { const rep = S.rep = await (await api("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    renderResult(rep); renderExports(rep); renderLast(rep); toast("Analysis complete", "ok"); }
  catch (e) { $("#result").innerHTML = `<div class="card"><h3>Analysis</h3><div class="notice bad">${esc(e.message)}</div></div>`; toast(e.message, "bad", 7000); }
  finally { $("#analyzeBtn").disabled = S.sel === null; }
}
$("#analyzeBtn").onclick = () => runAnalyze(false); $("#analyzeFull").onclick = () => runAnalyze(true);
const bar = (v, col) => `<div class="bar"><i style="width:${Math.max(0, Math.min(1, v || 0)) * 100}%;${col ? "background:" + col : ""}"></i></div>`;
function renderResult(r) {
  const c = r.classification, st = c.stream, f = c.fine, tagc = st.prediction === "MSW" ? "msw" : "ew", ex = r.explainability, co = r.contamination, cd = r.condition, im = r.carbon_economic;
  $("#result").innerHTML = `
  <div class="card"><h3>1 · Stream gate</h3><span class="tag ${tagc}">${st.prediction === "MSW" ? "Municipal waste" : "E-waste"}</span> <span class="big">${pct(st.confidence)}</span>${bar(st.confidence)}
   <div class="small muted">${esc(st.note)}</div><div class="small">MSW ${pct(st.probabilities.MSW)} · E-waste ${pct(st.probabilities.E_WASTE)}</div></div>
  <div class="card"><h3>2 · Fine-grained category</h3><span class="big">${esc(f.prediction)}</span> ${c.uncertain ? '<span class="tag unc">review required</span>' : ""}<div>${pct(f.confidence)}</div>${bar(f.confidence, c.uncertain ? "var(--warn)" : "")}
   <div class="small muted">Head: ${esc(f.head)} · alternatives: ${f.alternatives.map(a => `${esc(a[0])} ${pct(a[1])}`).join(", ") || "none"}</div>
   ${c.uncertain ? `<div class="notice warn">Uncertain: ${esc(c.uncertainty_reasons.join("; "))}. Not forced into a class — please verify manually.</div>` : ""}</div>
  <div class="card"><h3>Guidance</h3><p>${esc(r.recommendation.guidance)}</p>${r.recommendation.handling ? `<p class="small"><b>Handling:</b> ${esc(r.recommendation.handling)}</p>` : ""}<p class="small muted">${esc(r.recommendation.note)}</p></div>
  <div class="card"><h3>Contamination &amp; condition</h3><div class="kv"><span>Contamination proxy</span><span>${fmt(co.score)} (${esc(co.verdict)})</span><span>Saturation variance</span><span>${fmt(co.saturation_variance, 5)}</span>
   <span>Laplacian energy</span><span>${fmt(co.laplacian_energy, 1)}</span><span>Condition score</span><span>${fmt(cd.score)} (${esc(cd.verdict)})</span><span>Solidity</span><span>${fmt(cd.solidity, 4)}</span><span>Contour / hull area</span><span>${fmt(cd.contour_area, 0)} / ${fmt(cd.convex_hull_area, 0)}</span></div>
   <details><summary class="small muted">What these measure</summary><p class="small muted">${esc(co.label)}. ${esc(co.formula)}. ${esc(cd.label)}. ${esc(cd.formula)}.</p></details></div>
  <div class="card"><h3>Carbon &amp; recovery value</h3>${im.status === "ok" ? `<div class="kv"><span>CO₂e avoided</span><span>${fmt(im.co2e_avoided_kg, 4)} kg</span><span>Recovery value</span><span>$${fmt(im.recovery_value_usd, 4)}</span><span>Mass</span><span>${fmt(im.mass_kg)} kg (${esc(im.mass_method)})</span><span>Formula</span><span>${esc(im.formula)}</span></div>
   <div class="notice warn small">${esc(im.source)}</div><p class="small muted">${esc(im.notice)}</p>` : `<p class="muted">Not available: ${esc(im.reason)}</p>`}</div>
  <div class="card"><h3>Grad-CAM</h3>${ex.status === "ok" ? `<div class="thumbs"><img src="${r.artifacts.crop}" alt="crop"><img src="${r.artifacts.gradcam}?t=${Date.now()}" alt="gradcam"></div><p class="small muted">${esc(ex.region?.summary || "")}</p><button onclick="go('explain')">Open Explainability Lab</button>` : `<div class="notice bad">Explainability unavailable: ${esc(ex.error)}</div>`}</div>
  ${r.warnings.length ? `<div class="card"><h3>Warnings</h3>${r.warnings.map(w => `<div class="notice warn small">${esc(w)}</div>`).join("")}</div>` : ""}
  <div class="small muted">Analysis ${r.report_id} · ${r.timings.total_ms} ms total</div>`;
}
function dl(url, label) { return `<a href="${url}" download><button>${label}</button></a>`; }
function renderExports(r) { const id = r.report_id; $("#exportBtns").innerHTML = dl(`/api/reports/${id}/json`, "Download JSON") + dl(`/api/reports/${id}/csv`, "Export CSV") + dl(`/api/reports/${id}/excel`, "Export Excel (.xlsx)") + dl(`/api/reports/${id}/annotated-image`, "Annotated image") + (r.explainability.status === "ok" ? dl(`/api/reports/${id}/gradcam`, "Grad-CAM overlay") + dl(`/api/reports/${id}/heatmap`, "Raw heatmap") : ""); }
function renderLast(r) { $("#lastAnalysis").innerHTML = `<h3>Last analysis</h3><p><b>${esc(r.classification.fine.prediction)}</b> (${pct(r.classification.fine.confidence)}) in the <b>${r.classification.stream.prediction === "MSW" ? "municipal" : "e-waste"}</b> stream · ${esc(r.timestamp_utc)}</p>`; }

/* ---------- explainability lab ---------- */
function renderExplain() {
  const r = S.rep; if (!r) return; const ex = r.explainability, c = r.classification;
  if (ex.status !== "ok") { $("#explainBody").innerHTML = `<div class="card"><div class="notice bad">Grad-CAM failed: ${esc(ex.error)}</div></div>`; return; }
  const targets = [c.fine.prediction, ...c.fine.alternatives.map(a => a[0])];
  $("#explainBody").innerHTML = `<div class="grid2"><div class="card"><h3>Original image</h3><img src="${r.artifacts["annotated-image"]}" style="max-width:100%;border-radius:8px"></div>
   <div class="card"><h3>Selected crop → Grad-CAM</h3><div class="layer" id="lay"><img src="${r.artifacts.crop}"><img id="heatImg" src="${r.artifacts.heatmap}" style="opacity:.5"></div>
    <div class="legend"></div><div class="row small muted" style="justify-content:space-between"><span>low activation</span><span>high activation</span></div>
    <label>Heat-map opacity <input type="range" id="opa" min="0" max="1" step="0.05" value="0.5"></label></div></div>
  <div class="grid2"><div class="card"><h3>Target</h3><label>Explain class <select id="tgt">${targets.map(t => `<option>${esc(t)}</option>`).join("")}<option value="stream">Stream decision (needs dedicated stream model)</option></select></label>
    <div id="exMeta" class="kv" style="margin-top:10px"></div><p id="exSum" class="muted"></p><div id="exErr"></div>
    <div class="row" id="exDl"></div></div>
   <div class="card"><h3>How to read this</h3><p class="small">Warm regions are where increasing the activation of the last convolutional stage increases the chosen class score. The map is coarse (7×7 resized) and is <b>not</b> proof of causal reasoning or of which component the model used.</p>
    <p class="small muted">Stream and fine-grained explanations are separate: the stream explanation requires a dedicated stream classifier.</p></div></div>`;
  $("#opa").oninput = e => $("#heatImg").style.opacity = e.target.value;
  const show = (meta, heat, over) => { $("#exMeta").innerHTML = `<span>Target class</span><span>${esc(meta.target)}</span><span>Target logit</span><span>${fmt(meta.target_score)}</span><span>Layer</span><span>${esc(meta.target_layer)}</span><span>Normalisation</span><span>${esc(meta.normalization)}</span>`;
    $("#exSum").textContent = meta.region?.summary || ""; $("#heatImg").src = heat; $("#exDl").innerHTML = dl(over, "Download overlay") + dl(heat, "Download heatmap"); };
  show({ target: ex.target_class, target_score: ex.target_score, target_layer: ex.target_layer, normalization: ex.normalization, region: ex.region }, r.artifacts.heatmap, r.artifacts.gradcam);
  $("#tgt").onchange = async e => { $("#exErr").innerHTML = ""; try { const m = await (await api("/api/explain/gradcam", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ analysis_id: r.report_id, target: e.target.value, opacity: +$("#opa").value }) })).json(); show(m, m.heatmap_url, m.overlay_url); }
    catch (er) { $("#exErr").innerHTML = `<div class="notice bad">${esc(er.message)}</div>`; } };
}

/* ---------- analytics ---------- */
const THEME = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#cfd8e6" }, margin: { t: 38, r: 12, b: 40, l: 44 }, xaxis:{gridcolor:"#2a3750"},yaxis:{gridcolor:"#2a3750"},colorway: ["#5aa9ff", "#4cc38a", "#f0b04a", "#ef6b6b", "#b48cff", "#5fd3d3"] };
function filterQS() { const p = new URLSearchParams(); [["stream", "#fStream"], ["category", "#fCat"], ["min_conf", "#fMin"], ["max_conf", "#fMax"], ["date_from", "#fFrom"], ["date_to", "#fTo"]].forEach(([k, s]) => { const v = $(s).value; if (v) p.set(k, k === "date_to" ? v + "T23:59:59" : v); }); return p.toString(); }
function plot(id, title, data, layout = {}) { let d = $("#" + id); if (!d) { d = document.createElement("div"); d.id = id; d.className = "card"; d.style.padding = "4px"; $("#charts").appendChild(d); } Plotly.react(d, data, { ...THEME, height: 330, title: { text: title, font: { size: 14 } }, ...layout }, { responsive: true, displaylogo: false }); }
async function loadAnalytics() {
  if (!$("#fCat").options.length || $("#fCat").options.length === 1) CATS.forEach(c => $("#fCat").add(new Option(c, c)));
  const qs = filterQS(); $("#charts").innerHTML = "";
  const ov = await jget("/api/analytics/overview?" + qs);
  if (ov.empty) { $("#kpis").innerHTML = `<div class="card" style="grid-column:1/-1"><p class="muted">${esc(ov.message)}</p></div>`; $("#basis").textContent = ""; }
  else { const k = [["Images analysed", ov.images_analyzed], ["Objects analysed", ov.objects_analyzed], ["Municipal", ov.msw], ["E-waste", ov.e_waste], ["Uncertain", ov.uncertain], ["Mean detector conf.", fmt(ov.mean_detector_conf)], ["Mean stream conf.", fmt(ov.mean_stream_conf)], ["Mean category conf.", fmt(ov.mean_fine_conf)], ["Mean contamination", fmt(ov.mean_contamination)], ["Mean condition", fmt(ov.mean_condition)], ["CO₂e avoided (kg)*", fmt(ov.total_co2e_kg, 4)], ["Recovery value ($)*", fmt(ov.total_value_usd, 4)]];
    $("#kpis").innerHTML = k.map(([a, b]) => `<div class="kpi"><small>${a}</small><b>${b}</b></div>`).join(""); $("#basis").textContent = ov.basis; }
  const d = await jget("/api/analytics/distributions?" + qs), rec = d.records;
  if (rec.length) {
    const cnt = {}; rec.forEach(r => cnt[r.stream] = (cnt[r.stream] || 0) + 1);
    plot("c1", "Municipal vs E-waste", [{ type: "pie", labels: Object.keys(cnt), values: Object.values(cnt), hole: .45 }]);
    const cats = Object.keys(d.by_category); plot("c2", "Category distribution", [{ type: "bar", x: cats, y: cats.map(c => d.by_category[c].count) }]);
    const hist = (id, t, k, lo = 0, hi = 1) => plot(id, t, [{ type: "histogram", x: rec.map(r => r[k]).filter(v => v !== null), xbins: { start: lo, end: hi, size: (hi - lo) / 10 } }], { xaxis: { range: [lo, hi] } });
    hist("c3", "Category confidence", "fine_conf"); hist("c4", "Contamination proxy", "contamination"); hist("c5", "Condition score", "condition_score");
    plot("c6", "CO₂e avoided by category (kg, placeholder factors)", [{ type: "bar", x: cats, y: cats.map(c => d.by_category[c].co2e) }]);
    plot("c7", "Recovery value by category ($, placeholder factors)", [{ type: "bar", x: cats, y: cats.map(c => d.by_category[c].value) }]);
    plot("c8", "Inference latency (ms)", [{ type: "histogram", x: rec.map(r => r.latency_ms) }]);
    const t = [...rec].reverse(); plot("c9", "Analyses over time (confidence)", [{ type: "scatter", mode: "lines+markers", x: t.map(r => r.created_at), y: t.map(r => r.fine_conf) }], { yaxis: { range: [0, 1] } });
    $("#recTable").innerHTML = `<table><tr><th>Time (UTC)</th><th>Stream</th><th>Category</th><th>Conf</th><th>Uncertain</th><th>Cont.</th><th>Cond.</th></tr>${rec.slice(0, 50).map(r => `<tr class="click" data-id="${r.id}"><td>${esc(r.created_at)}</td><td>${esc(r.stream)}</td><td>${esc(r.category)}</td><td>${fmt(r.fine_conf)}</td><td>${r.uncertain ? "yes" : "no"}</td><td>${fmt(r.contamination)}</td><td>${fmt(r.condition_score)}</td></tr>`).join("")}</table>`;
    $$("#recTable tr.click").forEach(tr => tr.onclick = async () => { S.rep = await jget("/api/history/" + tr.dataset.id); go("explain"); });
  } else $("#recTable").innerHTML = '<p class="muted">No records match.</p>';
  renderEval();
}
async function renderEval() {
  const el = $("#evalCharts"); el.innerHTML = ""; const m = await jget("/api/analytics/model-performance");
  if (m.empty) { el.innerHTML = `<div class="card"><p class="muted">${esc(m.message)}</p></div>`; return; }
  Object.entries(m.models).forEach(([name, e], i) => { const mt = e.metrics || {}, h = e.history;
    const mk = (id) => { const d = document.createElement("div"); d.className = "card"; d.id = `ev${i}${id}`; el.appendChild(d); return d; };
    const metr = Object.entries(mt).filter(([k, v]) => typeof v === "number"); const d0 = mk("m"); d0.innerHTML = `<h3>${esc(name)} — test metrics</h3><div class="kv">${metr.map(([k, v]) => `<span>${esc(k)}</span><span>${fmt(v, 4)}</span>`).join("")}</div><p class="small muted">${esc(mt.split_note || "")}</p>`;
    if (mt.confusion_matrix && mt.classes) Plotly.newPlot(mk("cm"), [{ type: "heatmap", z: mt.confusion_matrix, x: mt.classes, y: mt.classes, colorscale: "Blues" }], { ...THEME, title: { text: name + " confusion matrix", font: { size: 14 } }, xaxis: { title: "predicted" }, yaxis: { title: "true", autorange: "reversed" } }, { responsive: true });
    if (mt.per_class) { const cl = Object.keys(mt.per_class); Plotly.newPlot(mk("pc"), ["precision", "recall", "f1"].map(k => ({ type: "bar", name: k, x: cl, y: cl.map(c => mt.per_class[c][k]) })), { ...THEME, barmode: "group", title: { text: name + " per-class metrics", font: { size: 14 } } }, { responsive: true }); }
    if (h?.train_loss) Plotly.newPlot(mk("lc"), [{ y: h.train_loss, name: "train loss" }, { y: h.val_loss, name: "val loss" }], { ...THEME, title: { text: name + " loss", font: { size: 14 } } }, { responsive: true });
    if (h?.train_acc) Plotly.newPlot(mk("ac"), [{ y: h.train_acc, name: "train acc" }, { y: h.val_acc, name: "val acc" }], { ...THEME, title: { text: name + " accuracy", font: { size: 14 } } }, { responsive: true }); });
}
$("#fApply").onclick = loadAnalytics; $("#fReset").onclick = () => { $$(".filters select,.filters input").forEach(e => e.value = ""); loadAnalytics(); };
$("#fCsv").onclick = () => { const a = document.createElement("a"); a.href = "/api/history/export?format=csv&" + filterQS(); a.download = "vista_history.csv"; a.click(); };

/* ---------- model page & reports ---------- */
async function loadModel() {
  const m = await jget("/api/models/metadata"), st = m.status, ev = m.evaluation;
  const row = (n, o) => `<tr><td>${n}</td><td><span class="tag ${o.ready ? "msw" : "unc"}">${o.ready ? "ready" : "missing"}</span></td><td>${esc(o.mode || o.kind || "")}</td><td class="small">${esc(o.weights || o.path || "")}</td><td class="small">${(o.classes || []).map(esc).join(", ")}</td></tr>`;
  $("#modelBody").innerHTML = `<div class="card"><h3>Component readiness</h3><table><tr><th>Component</th><th>Status</th><th>Mode</th><th>Path</th><th>Classes</th></tr>${row("Detector", st.detector)}${row("Stream classifier", st.stream_classifier)}${row("Fine-grained classifier", st.fine_classifier)}${row("MSW head", st.msw_head)}${row("E-waste head", st.ewaste_head)}${row("Grad-CAM", { ready: st.explainability.ready, mode: st.explainability.method })}</table></div>
  <div class="grid2"><div class="card"><h3>Inference thresholds</h3><div class="kv"><span>Detector confidence</span><span>${m.thresholds.detect_conf}</span><span>NMS IoU</span><span>${m.thresholds.detect_iou}</span><span>Category confidence rule τ</span><span>${m.thresholds.fine_tau}</span><span>Stream confidence τ</span><span>${m.thresholds.stream_tau}</span><span>Calibration</span><span>${st.fine_classifier.temperature && st.fine_classifier.temperature !== 1 ? "temperature " + st.fine_classifier.temperature : "none (raw softmax — not guaranteed calibrated)"}</span></div></div>
  <div class="card"><h3>Limitations</h3><ul class="small"><li>Generic COCO detector is not a validated waste detector; fine-tune on a waste detection set.</li><li>Stream gate is provisional unless a dedicated stream model is trained.</li><li>Contamination and condition are heuristics, not validated measurements.</li><li>Carbon / value factors are placeholders until replaced with cited data.</li><li>Single-image mass is unknown; default masses are assumptions.</li></ul></div></div>
  <div class="card"><h3>Evaluation artefacts</h3>${Object.keys(ev).length ? Object.keys(ev).map(k => `<p><b>${esc(k)}</b>: ${Object.entries(ev[k].metrics || {}).filter(([, v]) => typeof v === "number").map(([a, b]) => `${esc(a)} ${fmt(b, 4)}`).join(" · ")}</p>`).join("") : '<p class="muted">No evaluation has been run yet — nothing is shown rather than made-up numbers. Run scripts/train_classifier.py then scripts/evaluate.py.</p>'}</div>`;
}
async function loadReports() {
  const h = await jget("/api/history?limit=200"), sel = $("#repSel"); sel.innerHTML = h.length ? h.map(r => `<option value="${r.id}">${esc(r.created_at)} · ${esc(r.category)} (${esc(r.stream)})</option>`).join("") : "<option value=''>no analyses yet</option>";
  const upd = async () => { const id = sel.value; if (!id) { $("#repBtns").innerHTML = ""; $("#repPre").textContent = "No analyses yet."; return; } $("#repBtns").innerHTML = ["json","csv","excel","annotated-image","gradcam","heatmap"].map(k => dl(`/api/reports/${id}/${k}`, k === "excel" ? "Excel" : k)).join(""); $("#repPre").textContent = JSON.stringify(await jget("/api/history/" + id), null, 2); };
  sel.onchange = upd; upd();
}
$("#repClear").onclick = async () => { if (!confirm("Delete all local analysis history and stored images?")) return; await api("/api/history", { method: "DELETE" }); S.rep = null; toast("History cleared", "ok"); loadReports(); };
loadStatus();
