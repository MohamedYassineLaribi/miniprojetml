"""Étape 3 — API REST FastAPI : lancer avec  `uvicorn main:app --reload`."""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from core import pipelines
from core.asr import ASRUnavailableError
from core.classifier import EmptyTextError, ModelNotAvailableError
from core.ocr import OCRUnavailableError
from core.schemas import PredictionResponse, TextRequest
from core.validation import read_upload, sniff_kind

app = FastAPI(
    title="Classifieur de Sentiment Multi-source",
    description="Avis clients (texte, image via OCR, audio via ASR) → sentiment positif / négatif / neutre.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Les erreurs « de service » ne sont pas la faute du client : 503 ; contenu inexploitable : 422.
_SERVICE_ERRORS = (ModelNotAvailableError, OCRUnavailableError, ASRUnavailableError)


async def _run(func, *args, **kwargs) -> dict:
    """Exécute un traitement CPU dans un thread (ne bloque pas la boucle asyncio) + traduit les erreurs."""
    try:
        return await run_in_threadpool(func, *args, **kwargs)
    except EmptyTextError as exc:
        raise HTTPException(422, str(exc))
    except _SERVICE_ERRORS as exc:
        raise HTTPException(503, str(exc))


@app.get("/", include_in_schema=False)
def root():
    return {"message": "API Sentiment Multi-source", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["Divers"])
def health():
    return {"status": "ok"}


@app.post("/predict/text", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_text(payload: TextRequest):
    """Texte brut (JSON) → sentiment."""
    return await _run(pipelines.predict_text, payload.texte)


@app.post("/predict/image", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_image(file: UploadFile = File(..., description="PNG / JPEG / WEBP… (≤ 5 Mo)")):
    """Image → OCR (pytesseract) → sentiment."""
    data = await read_upload(file, "image")
    return await _run(pipelines.predict_image, data)


@app.post("/predict/audio", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_audio(file: UploadFile = File(..., description="WAV / MP3 / OGG… (≤ 25 Mo)")):
    """Audio → transcription (Whisper) → sentiment."""
    data = await read_upload(file, "audio")
    return await _run(pipelines.predict_audio, data, suffix=Path(file.filename or "").suffix or ".wav")


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict_auto(
    text: Optional[str] = Form(None, description="Texte brut (alternative au fichier)"),
    file: Optional[UploadFile] = File(None, description="Image ou audio : le type est détecté automatiquement"),
):
    """Détection automatique : fournir SOIT `text`, SOIT `file` (image ou audio)."""
    has_text = text is not None and text.strip() != ""
    if has_text == (file is not None and bool(file.filename)):
        raise HTTPException(422, "Fournir exactement un champ : `text` ou `file`.")
    if has_text:
        TextRequest(texte=text)  # mêmes règles de validation que /predict/text
        return await _run(pipelines.predict_text, text)
    head = await file.read(12)
    await file.seek(0)
    kind = sniff_kind(file, head)
    if kind is None:
        raise HTTPException(422, "Type de fichier non reconnu (image ou audio attendu).")
    data = await read_upload(file, kind)
    if kind == "image":
        return await _run(pipelines.predict_image, data)
    return await _run(pipelines.predict_audio, data, suffix=Path(file.filename or "").suffix or ".wav")
