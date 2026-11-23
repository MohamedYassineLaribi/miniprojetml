"""Chargement des artefacts (.pkl) et prédiction du sentiment à partir de texte brut."""
from functools import lru_cache

import joblib

from core.config import MODEL_PATH, VECTORIZER_PATH
from core.preprocessing import clean_text


class ModelNotAvailableError(RuntimeError):
    """Les fichiers model.pkl / vectorizer.pkl sont introuvables (lancer `python train.py`)."""


class EmptyTextError(ValueError):
    """Le texte ne contient aucun mot exploitable après nettoyage."""


class SentimentClassifier:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        if not cleaned:
            raise EmptyTextError("Aucun mot exploitable dans le texte fourni.")
        proba = self.model.predict_proba(self.vectorizer.transform([cleaned]))[0]
        best = int(proba.argmax())
        return {
            "sentiment_predit": str(self.model.classes_[best]),
            "confiance": round(float(proba[best]), 4),
            "probabilites": {str(c): round(float(p), 4) for c, p in zip(self.model.classes_, proba)},
        }


@lru_cache(maxsize=1)
def get_classifier() -> SentimentClassifier:
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
        raise ModelNotAvailableError("Modèle absent : exécutez d'abord `python train.py`.")
    return SentimentClassifier(joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH))
