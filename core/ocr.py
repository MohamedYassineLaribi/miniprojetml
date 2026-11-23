"""Étape 2 (image) — OCR avec pytesseract + Pillow."""
import io
import os
from pathlib import Path
from typing import Union

import pytesseract
from PIL import Image, ImageOps

from core.config import OCR_LANG

# Windows : définir TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe" si tesseract n'est pas dans le PATH
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


class OCRUnavailableError(RuntimeError):
    """Tesseract n'est pas installé / langue manquante."""


def _prepare(img: Image.Image) -> Image.Image:
    """Niveaux de gris + contraste + agrandissement des petites images : améliore l'OCR."""
    img = ImageOps.exif_transpose(img)
    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    if max(gray.size) < 1000:
        gray = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)
    return gray


def image_to_text(image: Union[str, Path, bytes, Image.Image]) -> str:
    """Accepte un chemin, des octets ou un objet PIL ; retourne le texte brut extrait."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    try:
        text = pytesseract.image_to_string(_prepare(image), lang=OCR_LANG, config="--psm 6")
    except (pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as exc:
        raise OCRUnavailableError(f"OCR indisponible : {exc}") from exc
    return " ".join(text.split())
