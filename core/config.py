"""Configuration centrale (surchargeable par variables d'environnement)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
DATA_DIR = ROOT / "data_test_v2"

# --- Limites de validation des fichiers entrants ---
MAX_TEXT_LENGTH = 5_000
MAX_IMAGE_BYTES = 5 * 1024 * 1024      # 5 Mo
MAX_AUDIO_BYTES = 25 * 1024 * 1024     # 25 Mo

ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/tiff"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_AUDIO_MIME = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3", "audio/x-mpeg-3",
    "audio/ogg", "audio/flac", "audio/x-flac", "audio/mp4", "audio/x-m4a", "audio/webm",
}
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

# --- OCR / ASR ---
OCR_LANG = os.getenv("OCR_LANG", "fra")            # "fra+eng" possible
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny | base | small | medium
ASR_LANG = os.getenv("ASR_LANG", "fr")
