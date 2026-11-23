"""Schémas Pydantic : validation des entrées et de la réponse JSON unifiée."""
from typing import Dict, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from core.config import MAX_TEXT_LENGTH

Sentiment = Literal["positif", "negatif", "neutre"]


class TextRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"texte": "Livraison rapide et produit conforme, je recommande !"}})

    texte: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LENGTH,
        validation_alias=AliasChoices("texte", "text"),   # accepte aussi la clé « text »
        description="Avis client à analyser.",
    )

    @field_validator("texte")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le texte ne doit pas être vide.")
        return v


class PredictionResponse(BaseModel):
    source: Literal["texte", "image", "audio"]
    texte_extrait: str
    sentiment_predit: Sentiment
    confiance: float = Field(..., ge=0, le=1)
    probabilites: Dict[str, float]
