"""FastAPI service: GET /health, POST /predict (multipart image)."""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from vista.classification.loader import WasteClassifier
from vista.pipeline import analyze, strip_internal

app = FastAPI(title="VISTA API", version="0.1.0")
_classifier = WasteClassifier.from_env()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "classifier_loaded": _classifier is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="unreadable image") from error
    return strip_internal(analyze(image, _classifier, with_gradcam=False))
