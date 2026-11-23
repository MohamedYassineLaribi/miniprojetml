"""Étape 2 (audio) — transcription avec openai-whisper (chargement paresseux)."""
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Union

from core.config import ASR_LANG, WHISPER_MODEL


class ASRUnavailableError(RuntimeError):
    """openai-whisper (ou ffmpeg) n'est pas installé."""


@lru_cache(maxsize=1)
def _load_model():
    try:
        import whisper  # import tardif : l'API démarre même sans whisper
    except ImportError as exc:
        raise ASRUnavailableError("openai-whisper n'est pas installé (pip install openai-whisper).") from exc
    return whisper.load_model(WHISPER_MODEL)


def audio_to_text(audio: Union[str, Path, bytes], suffix: str = ".wav") -> str:
    """Accepte un chemin ou des octets ; retourne la transcription (whisper gère wav/mp3/… via ffmpeg)."""
    model = _load_model()
    tmp_path = None
    try:
        if isinstance(audio, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name
            path = tmp_path
        else:
            path = str(audio)
        try:
            result = model.transcribe(path, language=ASR_LANG, fp16=False)
        except FileNotFoundError as exc:  # ffmpeg absent
            raise ASRUnavailableError("ffmpeg est requis par whisper.") from exc
        return result["text"].strip()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
