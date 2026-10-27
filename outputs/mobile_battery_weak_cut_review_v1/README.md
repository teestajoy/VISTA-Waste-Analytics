# CAND-0002 weak-edge review gallery

Launch from the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\outputs\mobile_battery_weak_cut_review_v1\app.py
```

The gallery displays W-CUT-01 and W-CUT-03 using bytes read directly from `~/Downloads/VISTA_mobile_dataset_raw.zip`. It verifies the fixed edge IDs against the v2 edge table and checks both endpoints against the v2 manifest.

The default exclusive review output is `data/processed/mobile_battery_cand0002_weak_cut_review_v1.csv`. If that file exists, choose another versioned filename in the app. No source review CSV is modified.
