"""Validation des fichiers entrants : type MIME, extension, signature binaire, taille."""
import io
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from core import config as cfg


def _ext(upload: UploadFile) -> str:
    return Path(upload.filename or "").suffix.lower()


def sniff_kind(upload: UploadFile, head: bytes = b"") -> Optional[str]:
    """Retourne 'image', 'audio' ou None. Ordre : MIME déclaré, extension, signature binaire."""
    mime = (upload.content_type or "").split(";")[0].strip().lower()
    ext = _ext(upload)
    if mime in cfg.ALLOWED_IMAGE_MIME:
        return "image"
    if mime in cfg.ALLOWED_AUDIO_MIME:
        return "audio"
    if mime in ("", "application/octet-stream"):
        if ext in cfg.ALLOWED_IMAGE_EXT:
            return "image"
        if ext in cfg.ALLOWED_AUDIO_EXT:
            return "audio"
        if head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"BM")):
            return "image"
        if head.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC", b"\xff\xfb")):
            return "audio"
    return None


async def read_upload(upload: UploadFile, expected: str) -> bytes:
    """Lit et valide un fichier. expected ∈ {'image', 'audio'}. Lève HTTPException 422/413."""
    max_bytes = cfg.MAX_IMAGE_BYTES if expected == "image" else cfg.MAX_AUDIO_BYTES
    data = await upload.read(max_bytes + 1)          # lecture bornée : pas de DoS mémoire
    if len(data) > max_bytes:
        raise HTTPException(413, f"Fichier trop volumineux (max {max_bytes // (1024 * 1024)} Mo).")
    if not data:
        raise HTTPException(422, "Fichier vide.")
    kind = sniff_kind(upload, data[:12])
    if kind != expected:
        raise HTTPException(
            422, f"Type de fichier non supporté ({upload.content_type or 'inconnu'}) : "
                 f"un fichier {expected} est attendu.")
    if expected == "image":
        try:
            Image.open(io.BytesIO(data)).verify()     # image réellement lisible ?
        except (UnidentifiedImageError, OSError, SyntaxError):
            raise HTTPException(422, "Image corrompue ou illisible.")
    return data
