# Mobile and Battery candidate image gallery (v1)

This is a local, read-only Streamlit gallery for the review worksheet and review guide:

- `data/processed/mobile_battery_related_image_review_v1.csv`
- `docs/mobile_battery_related_image_review_v1.md`

The app reads archived images directly from `VISTA_mobile_dataset_raw.zip` using Python's
ZIP reader. It does not extract files or write to the archive, worksheet, labels, splits,
review decisions, or model artifacts. The gallery has no controls for changing decisions.

## Open the gallery

From the VISTA project root in PowerShell, run:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\outputs\mobile_battery_related_image_gallery_v1\app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`) in the terminal and usually
opens it in the default browser. Use the **Candidate group** selector in the sidebar to view
grouped images side by side. The default ZIP path is
`~/Downloads/VISTA_mobile_dataset_raw.zip`; if it is elsewhere, enter its
actual path in the sidebar. The app only reads that file. Missing archive members are shown
as unavailable, without attempting extraction.

Groups beginning with `TS-` are filename-timestamp candidates. Groups beginning with `ND-`
are perceptual-hash and ORB/RANSAC visual candidates. Neither is proof of physical-device
identity. The displayed human decision comes from the CSV and remains `pending` unless a
reviewer changes the worksheet outside this read-only app.
