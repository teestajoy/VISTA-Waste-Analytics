# Focused mobile/battery bridge review

This local Streamlit gallery displays strong bridges **#3 and #4 only**, using image bytes read directly from:

`~/Downloads/VISTA_mobile_dataset_raw.zip`

It displays each pair's archive paths, source classes and splits, SHA-256 comparison, pHash distance, ORB/RANSAC measurements, and filename-timestamp comparison. It verifies each displayed image's SHA-256 against the v2 edge CSV. It does not extract the ZIP, assign a device identity, alter labels or splits, or edit the v2 artifacts.

## Launch

From the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\outputs\mobile_battery_strong_bridge_review_v1\app.py
```

The app opens in your browser. If the ZIP is in a different location, enter its path in the **Original ZIP path** field.

Choose a decision and enter notes separately for each bridge, then click **Save reviews**. The default output is `data/processed/mobile_battery_strong_bridge_review_v1.csv`. If it already exists, the app refuses to overwrite it; enter a new versioned filename. The CSV contains the bridge ID, both image paths, decision, and notes. Saving does not change the source worksheet, labels, splits, or ZIP.
