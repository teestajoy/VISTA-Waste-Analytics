# Mobile and Battery candidate review gallery (v2)

This version starts from `data/processed/mobile_battery_related_image_review_v2.csv` and
provides review controls by candidate group. Images are read directly from the ZIP in memory;
the archive is never extracted or written. The original v1 gallery and v1/v2 worksheets are
not modified by this app.

## Launch

From the VISTA project root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m streamlit run .\outputs\mobile_battery_related_image_gallery_v2\app.py
```

Streamlit prints its local address (normally `http://localhost:8501`) and may open a browser.
Select a candidate group in the sidebar, choose a decision, add any notes, then click **Apply
this review to the group**. Applying changes updates the in-session copy only. Use the export
button to download it, or **Save reviewed CSV** to create the configured local output once.

The default save path is:

`data/processed/mobile_battery_related_image_review_v3.csv`

If that file already exists, saving is blocked. Choose another new versioned `.csv` filename
inside `data/processed`; existing files are never overwritten. The download button remains
available and does not write to the repository.

## Decision semantics and overlap

- `same_device` means the images show the same physical object. It does not mean the image
  files are byte-identical or that one is an augmentation of the other.
- If the evidence only suggests the same source image, an augmentation, or a near-identical
  capture, use `uncertain` and describe that suspicion in the notes.
- A group decision applies to every image row in that candidate group. Some images belong to
  multiple groups; the app warns when this occurs because the worksheet stores one decision
  and notes field per image. Reviewing an overlapping group later updates the shared row.
- Existing v2 decisions and notes are loaded as-is. New notes are appended; leaving the notes
  field blank preserves prior notes. Reviewer/date and physical-device/session IDs are not
  guessed or filled by the gallery.
