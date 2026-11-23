"""Architecture en cascade : image/audio -> texte -> MÊME classifieur -> réponse unifiée."""
from pathlib import Path
from typing import Union

from core import asr, ocr
from core.classifier import EmptyTextError, get_classifier


def _unified(source: str, text: str) -> dict:
    result = get_classifier().predict(text)
    return {"source": source, "texte_extrait": text, **result}


def predict_text(text: str) -> dict:
    return _unified("texte", text.strip())


def predict_image(image: Union[str, Path, bytes]) -> dict:
    text = ocr.image_to_text(image)
    if not text:
        raise EmptyTextError("Aucun texte détecté dans l'image.")
    return _unified("image", text)


def predict_audio(audio: Union[str, Path, bytes], suffix: str = ".wav") -> dict:
    text = asr.audio_to_text(audio, suffix=suffix)
    if not text:
        raise EmptyTextError("Aucune parole détectée dans l'audio.")
    return _unified("audio", text)
