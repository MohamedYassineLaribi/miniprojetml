"""Nettoyage minimal du texte (identique à l'entraînement ET à l'inférence)."""
import re
import unicodedata

# Stop-words français SANS les mots porteurs de sentiment :
# on conserve volontairement « ne, pas, non, ni, jamais, rien, aucun, plus, très, trop, sans, mais »
# car ils portent la négation / l'intensité (« ne fonctionne pas », « très déçu »).
STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou", "a", "au", "aux",
    "en", "dans", "sur", "sous", "pour", "par", "avec", "ce", "cet", "cette", "ces",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos",
    "leur", "leurs", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on", "me", "te",
    "se", "qui", "que", "qu", "est", "es", "suis", "sont", "ete", "etre", "ai", "as", "ont",
    "avoir", "y", "c", "s", "j", "m", "n", "t", "lors", "comme", "bonjour",
}

_URL = re.compile(r"https?://\S+|www\.\S+")
_FILENAME = re.compile(r"\b\w+\.(png|jpe?g|wav|mp3|webp|bmp)\b")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """minuscules -> sans URL/nom de fichier -> sans accents -> lettres seules -> sans stop-words."""
    text = (text or "").lower()
    text = _URL.sub(" ", text)
    text = _FILENAME.sub(" ", text)
    text = _strip_accents(text)
    text = _NON_LETTER.sub(" ", text)          # supprime chiffres, ponctuation, symboles
    tokens = _SPACES.sub(" ", text).strip().split(" ")
    if remove_stopwords:
        tokens = [t for t in tokens if t and t not in STOPWORDS]
    return " ".join(t for t in tokens if t)
