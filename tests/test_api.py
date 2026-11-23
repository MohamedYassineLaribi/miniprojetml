"""Étape 5 — Tests unitaires et d'intégration (pytest + TestClient/httpx).

Lancer :  pytest -v
Les tests ASR (Whisper) sont *mockés* : rapides et sans GPU/téléchargement de modèle.
Les tests OCR utilisent le VRAI Tesseract et sont ignorés (skip) s'il n'est pas installé avec le pack « fra ».
"""
import io
from pathlib import Path

import pytest
from PIL import Image

from core import asr, config, ocr
from core.classifier import ModelNotAvailableError
from core.preprocessing import clean_text
from core.schemas import PredictionResponse, TextRequest

DATA = Path(__file__).resolve().parent.parent / "data_test_v2"
IMG = DATA / "images" / "I02.png"      # réclamation (négatif)
AUDIO = DATA / "audios" / "A01.mp3"

POSITIF = "Excellente qualité, livraison super rapide, je recommande vivement !"
NEGATIF = "Article reçu cassé, service client injoignable, très déçu, c'est une arnaque."


def tesseract_fra_ok() -> bool:
    try:
        import pytesseract
        return "fra" in pytesseract.get_languages()
    except Exception:
        return False


needs_ocr = pytest.mark.skipif(not tesseract_fra_ok(), reason="Tesseract + pack 'fra' non installés")


def png_bytes(size=(60, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def assert_unified(body: dict, source: str):
    """La réponse respecte le contrat JSON unifié."""
    PredictionResponse(**body)                                   # validation Pydantic complète
    assert body["source"] == source
    assert set(["source", "texte_extrait", "sentiment_predit", "confiance"]) <= set(body)
    assert body["sentiment_predit"] in {"positif", "negatif", "neutre"}
    assert 0 <= body["confiance"] <= 1
    assert abs(sum(body["probabilites"].values()) - 1) < 1e-2


@pytest.fixture
def fake_asr(monkeypatch):
    monkeypatch.setattr(asr, "audio_to_text", lambda audio, suffix=".wav": NEGATIF)


@pytest.fixture
def fake_ocr(monkeypatch):
    monkeypatch.setattr(ocr, "image_to_text", lambda image: POSITIF)


# ───────────────────────── Tests unitaires ─────────────────────────
class TestPreprocessing:
    def test_lowercase_accents_punctuation(self):
        assert clean_text("Très DÉÇU !!!") == "tres decu"

    def test_removes_stopwords_but_keeps_negation(self):
        out = clean_text("Le produit ne fonctionne pas du tout")
        assert "le" not in out.split() and "du" not in out.split()
        assert "ne" in out.split() and "pas" in out.split()

    def test_removes_noise(self):
        assert clean_text("COMMANDE #9921 ★★★ I02.png https://x.fr") == "commande"

    def test_empty_and_none(self):
        assert clean_text("") == "" and clean_text(None) == ""


class TestSchemas:
    def test_valid_and_stripped(self):
        assert TextRequest(texte="  bonjour  ").texte == "bonjour"

    def test_alias_text(self):
        assert TextRequest(text="ok").texte == "ok"

    @pytest.mark.parametrize("bad", ["", "   ", "x" * (config.MAX_TEXT_LENGTH + 1)])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            TextRequest(texte=bad)


@needs_ocr
def test_ocr_extracts_french_text():
    text = ocr.image_to_text(IMG)
    assert "RÉCLAMATION" in text and "injoignable" in text


# ───────────────────────── /health ─────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


# ───────────────────────── POST /predict/text ─────────────────────────
class TestPredictText:
    def test_positive(self, client):
        r = client.post("/predict/text", json={"texte": POSITIF})
        assert r.status_code == 200
        body = r.json()
        assert_unified(body, "texte")
        assert body["sentiment_predit"] == "positif"
        assert body["texte_extrait"] == POSITIF

    def test_negative(self, client):
        r = client.post("/predict/text", json={"texte": NEGATIF})
        assert r.status_code == 200
        assert r.json()["sentiment_predit"] == "negatif"

    def test_neutral_label_possible(self, client):
        r = client.post("/predict/text", json={"texte": "Fonctionne comme indiqué dans la notice."})
        assert r.status_code == 200
        assert_unified(r.json(), "texte")

    def test_alias_text_key(self, client):
        assert client.post("/predict/text", json={"text": POSITIF}).status_code == 200

    @pytest.mark.parametrize("payload", [
        {},                                   # champ manquant
        {"texte": ""},                        # vide
        {"texte": "     "},                   # espaces seulement
        {"texte": 123},                       # mauvais type
        {"texte": None},
        {"texte": "x" * (config.MAX_TEXT_LENGTH + 1)},   # trop long
        {"autre": "champ"},
    ])
    def test_invalid_payload_422(self, client, payload):
        r = client.post("/predict/text", json=payload)
        assert r.status_code == 422
        assert "detail" in r.json()

    def test_not_json_422(self, client):
        assert client.post("/predict/text", content="texte brut",
                           headers={"content-type": "text/plain"}).status_code == 422

    def test_only_symbols_422(self, client):
        # rien d'exploitable après nettoyage (« !!! ») -> erreur propre, pas de prédiction arbitraire
        assert client.post("/predict/text", json={"texte": "!!! ???"}).status_code == 422

    def test_wrong_method_405(self, client):
        assert client.get("/predict/text").status_code == 405

    def test_model_missing_503(self, client, monkeypatch):
        import core.pipelines as pl

        def boom():
            raise ModelNotAvailableError("absent")
        monkeypatch.setattr(pl, "get_classifier", boom)
        assert client.post("/predict/text", json={"texte": POSITIF}).status_code == 503


# ───────────────────────── POST /predict/image ─────────────────────────
class TestPredictImage:
    @needs_ocr
    def test_real_ocr_pipeline(self, client):
        r = client.post("/predict/image", files={"file": ("I02.png", IMG.read_bytes(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert_unified(body, "image")
        assert "injoignable" in body["texte_extrait"]
        assert body["sentiment_predit"] == "negatif"

    def test_with_mocked_ocr(self, client, fake_ocr):
        r = client.post("/predict/image", files={"file": ("a.png", png_bytes(), "image/png")})
        assert r.status_code == 200
        assert_unified(r.json(), "image")
        assert r.json()["sentiment_predit"] == "positif"

    def test_missing_file_422(self, client):
        assert client.post("/predict/image").status_code == 422

    def test_wrong_mime_422(self, client):
        r = client.post("/predict/image", files={"file": ("a.txt", b"hello", "text/plain")})
        assert r.status_code == 422

    def test_audio_sent_to_image_route_422(self, client):
        r = client.post("/predict/image", files={"file": ("a.mp3", b"ID3xxxxxxxx", "audio/mpeg")})
        assert r.status_code == 422

    def test_corrupt_image_422(self, client):
        r = client.post("/predict/image", files={"file": ("a.png", b"pas une image", "image/png")})
        assert r.status_code == 422

    def test_empty_file_422(self, client):
        assert client.post("/predict/image", files={"file": ("a.png", b"", "image/png")}).status_code == 422

    def test_too_large_413(self, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 10)
        r = client.post("/predict/image", files={"file": ("a.png", png_bytes(), "image/png")})
        assert r.status_code == 413

    def test_no_text_in_image_422(self, client, monkeypatch):
        monkeypatch.setattr(ocr, "image_to_text", lambda image: "")
        r = client.post("/predict/image", files={"file": ("blank.png", png_bytes(), "image/png")})
        assert r.status_code == 422
        assert "texte" in r.json()["detail"].lower()

    def test_ocr_unavailable_503(self, client, monkeypatch):
        def boom(image):
            raise ocr.OCRUnavailableError("tesseract absent")
        monkeypatch.setattr(ocr, "image_to_text", boom)
        assert client.post("/predict/image", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 503


# ───────────────────────── POST /predict/audio ─────────────────────────
class TestPredictAudio:
    def test_with_mocked_asr(self, client, fake_asr):
        r = client.post("/predict/audio", files={"file": ("A02.mp3", AUDIO.read_bytes(), "audio/mpeg")})
        assert r.status_code == 200
        body = r.json()
        assert_unified(body, "audio")
        assert body["texte_extrait"] == NEGATIF
        assert body["sentiment_predit"] == "negatif"

    def test_wav_mime(self, client, fake_asr):
        r = client.post("/predict/audio", files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")})
        assert r.status_code == 200

    def test_octet_stream_with_audio_extension_ok(self, client, fake_asr):
        r = client.post("/predict/audio", files={"file": ("a.mp3", b"ID3data", "application/octet-stream")})
        assert r.status_code == 200

    def test_missing_file_422(self, client):
        assert client.post("/predict/audio").status_code == 422

    def test_wrong_mime_422(self, client):
        assert client.post("/predict/audio", files={"file": ("a.pdf", b"%PDF", "application/pdf")}).status_code == 422

    def test_image_sent_to_audio_route_422(self, client):
        assert client.post("/predict/audio", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 422

    def test_empty_file_422(self, client):
        assert client.post("/predict/audio", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 422

    def test_too_large_413(self, client, monkeypatch):
        monkeypatch.setattr(config, "MAX_AUDIO_BYTES", 5)
        assert client.post("/predict/audio", files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}).status_code == 413

    def test_silence_empty_transcript_422(self, client, monkeypatch):
        monkeypatch.setattr(asr, "audio_to_text", lambda audio, suffix=".wav": "")
        assert client.post("/predict/audio", files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}).status_code == 422

    def test_asr_unavailable_503(self, client, monkeypatch):
        def boom(audio, suffix=".wav"):
            raise asr.ASRUnavailableError("whisper absent")
        monkeypatch.setattr(asr, "audio_to_text", boom)
        assert client.post("/predict/audio", files={"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}).status_code == 503


# ───────────────────────── POST /predict (détection automatique) ─────────────────────────
class TestPredictAuto:
    def test_text_form(self, client):
        r = client.post("/predict", data={"text": POSITIF})
        assert r.status_code == 200
        assert_unified(r.json(), "texte")

    def test_detects_image(self, client, fake_ocr):
        r = client.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 200
        assert_unified(r.json(), "image")

    def test_detects_audio(self, client, fake_asr):
        r = client.post("/predict", files={"file": ("x.mp3", b"ID3data", "audio/mpeg")})
        assert r.status_code == 200
        assert_unified(r.json(), "audio")

    def test_detects_by_extension_when_mime_generic(self, client, fake_asr):
        r = client.post("/predict", files={"file": ("x.wav", b"RIFFxxxxWAVE", "application/octet-stream")})
        assert r.status_code == 200 and r.json()["source"] == "audio"

    def test_nothing_422(self, client):
        assert client.post("/predict").status_code == 422

    def test_text_and_file_422(self, client, fake_ocr):
        r = client.post("/predict", data={"text": POSITIF}, files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 422

    def test_blank_text_422(self, client):
        assert client.post("/predict", data={"text": "   "}).status_code == 422

    def test_unknown_file_type_422(self, client):
        assert client.post("/predict", files={"file": ("x.zip", b"PK", "application/zip")}).status_code == 422


# ───────────────────────── Documentation interactive ─────────────────────────
def test_openapi_and_docs_exposed(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/predict/text", "/predict/image", "/predict/audio"} <= set(paths)
