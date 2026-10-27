# VISTA Demo Guide

## Launch

From the project root in Windows PowerShell or the VS Code integrated terminal:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The app uses the dark theme in `.streamlit/config.toml`. Its `headless = true`
setting prevents Streamlit from opening a browser automatically; it does not
disable the local web server. When startup succeeds, open the `Local URL` printed
in the terminal (normally `http://localhost:8501`).

## Image-upload demonstration

1. Start the command above and wait for Streamlit to print its local URL.
2. Open the URL and choose **Live Detection** in the sidebar.
3. Under the image input, choose **Upload an image** and select a clear image
   containing one or more ordinary objects.
4. Leave **Also classify the whole image** unchecked for the clearest separation
   of the current outputs, then select **Analyze image**.
5. Review the annotated image and **Generic object detections** table. YOLO's
   labels and scores describe its general-purpose object detector; they are not
   VISTA waste categories.
6. In **Primary waste-family result**, expect:
   **“Uncertain — verified MSW/E-waste model not available”**. No family
   confidence is available or inferred from YOLO or the paper/cardboard pilot.
7. The paper/cardboard pilot sections are separate and experimental. If enabled,
   they may show whole-image and detected-object-crop outputs. Their 0.70 cutoff
   is provisional, not calibrated, and uncertain outputs need manual review.
8. The **Explainability** panel states that genuine Grad-CAM is unavailable.

The same detection screen also offers a webcam snapshot input. Select a snapshot
and press **Analyze image**; this does not start continuous webcam detection.

## What is available

**Working:** image upload and webcam snapshots; image decoding; general-purpose
YOLO object detection; separate display of detection labels, boxes, and scores;
fail-closed MSW/E-waste family result; explicit unavailable explainability
message.

**Experimental:** the paper/cardboard ResNet18 pilot, including optional
whole-image and YOLO-crop predictions. It is a small reviewed pilot and is not a
verified general waste classifier. Its provisional confidence cutoff is not
calibrated.

**Unavailable:** verified MSW-versus-E-waste classification; the remaining
waste-category classifiers; genuine Grad-CAM; contamination or condition
assessment; recyclability decisions; carbon-impact or economic-recovery
estimates; implemented analytics and detection history. Do not present these as
working demo results.

## Troubleshooting

- **`WinError 10013` while starting or checking the local URL:** Windows denied a
  local socket operation. The Streamlit headless setting is not itself a port
  permission and this error does not establish which Windows policy denied
  access. Run the launch command from the VS Code integrated terminal on the
  project machine. If it persists, ask the device administrator to review the
  applicable local application/network policy; do not disable or bypass firewall
  or security controls. The dashboard has not been browser-verified as part of
  this guide.
- **The page does not open:** first confirm the terminal reports that Streamlit
  started and use the exact `Local URL` it prints. If it reports that the port is
  already in use, close the other local Streamlit process or ask before choosing
  another port.
- **Image upload is rejected:** try a readable JPEG or PNG and analyze again.
- **YOLO cannot load its model:** check the terminal error and the configured
  model availability. This guide does not download model weights.
- **Paper/cardboard output is unavailable:** the optional pilot checkpoint may
  be missing or fail to load. The generic YOLO detections and the family result
  remain separate; do not interpret either as a substitute pilot prediction.
