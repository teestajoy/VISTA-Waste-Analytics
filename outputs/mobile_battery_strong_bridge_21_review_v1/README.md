# Strong bridge #21 review gallery

Launch from the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\outputs\mobile_battery_strong_bridge_21_review_v1\app.py
```

The app reads the two source images directly from `~/Downloads/VISTA_mobile_dataset_raw.zip` and verifies their SHA-256 against the v2 edge row. No archive extraction occurs.

The default output path is `data/processed/mobile_battery_strong_bridge_21_review_v1.csv`. It contains bridge ID, both image paths, decision, and notes. The file is created exclusively; if it already exists, choose a new versioned filename in the app.
