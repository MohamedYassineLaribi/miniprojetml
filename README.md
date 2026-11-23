# 💬 Classifieur de Sentiment Multi-source (E-commerce Customer Voice)

Un avis client arrive en **texte**, **image** (capture d'écran) ou **audio** (message vocal).
Image et audio sont convertis en texte (OCR / ASR), puis **un seul classifieur textuel** prédit
`positif`, `negatif` ou `neutre` — architecture **en cascade** exposée via une API REST FastAPI.

```
Texte ───────────────────────────────┐
Image ──► OCR (pytesseract) ─► texte ┼──► TF-IDF + Régression logistique ──► JSON unifié
Audio ──► ASR (Whisper)     ─► texte ┘
```

## 1. Structure

```
main.py                 API FastAPI (routes /predict, /predict/text|image|audio, /health)
train.py                Entraînement + évaluation + sauvegarde des .pkl
evaluate_pipeline.py    Évaluation de bout en bout sur images/ et audios/
streamlit_app.py        Mini-application web de test (Option B)
core/                   preprocessing · classifier · ocr · asr · pipelines · schemas · validation · config
models/                 model.pkl, vectorizer.pkl, metrics.json
reports/                confusion_matrix.png, classification_report.txt, pipeline_eval.json
tests/                  test_api.py (+ conftest.py)  → pytest
data_test_v2/           dataset : textes.csv, images/, audios/ (+ CSV de référence)
docs/screenshots/       captures d'écran (à ajouter, voir §7)
```

## 2. Installation

**Python ≥ 3.10** recommandé.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Deux programmes système sont nécessaires **en plus** des paquets pip :

| Outil | Rôle | Linux (apt) | macOS (brew) | Windows |
|---|---|---|---|---|
| **Tesseract + pack français** | OCR | `sudo apt install tesseract-ocr tesseract-ocr-fra` | `brew install tesseract tesseract-lang` | installeur [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) → cocher **French** ; puis `set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` |
| **ffmpeg** | lecture audio par Whisper | `sudo apt install ffmpeg` | `brew install ffmpeg` | `winget install ffmpeg` |

Vérification : `tesseract --list-langs` doit afficher `fra`, et `ffmpeg -version` doit répondre.

> Variables d'environnement optionnelles : `WHISPER_MODEL` (`tiny`/`base`/`small`, défaut `base`),
> `OCR_LANG` (défaut `fra`), `ASR_LANG` (défaut `fr`), `TESSERACT_CMD`, `API_URL` (pour Streamlit).
> La 1re transcription télécharge le modèle Whisper (≈140 Mo pour `base`).

## 3. Entraîner le modèle

```bash
python train.py
```
Génère `models/model.pkl`, `models/vectorizer.pkl`, `models/metrics.json` et les rapports dans `reports/`.
Les `.pkl` fournis ont été produits avec scikit-learn 1.9 : **relancez `python train.py`** si votre version diffère.

## 4. Lancer l'API et l'application web

```bash
uvicorn main:app --reload                 # API  → http://127.0.0.1:8000
streamlit run streamlit_app.py            # Web  → http://localhost:8501   (2e terminal)
```
Documentation interactive Swagger : **http://127.0.0.1:8000/docs** (ReDoc : `/redoc`).

### Exemples curl

```bash
curl -X POST http://127.0.0.1:8000/predict/text -H "Content-Type: application/json" \
     -d '{"texte": "Livraison rapide, produit excellent, je recommande !"}'

curl -X POST http://127.0.0.1:8000/predict/image -F "file=@data_test_v2/images/I02.png"
curl -X POST http://127.0.0.1:8000/predict/audio -F "file=@data_test_v2/audios/A01.mp3"
curl -X POST http://127.0.0.1:8000/predict       -F "file=@data_test_v2/images/I04.png"   # détection auto
curl -X POST http://127.0.0.1:8000/predict       -F "text=Très déçu, article cassé"
```

Réponse unifiée :
```json
{"source": "image", "texte_extrait": "...", "sentiment_predit": "negatif", "confiance": 0.54,
 "probabilites": {"negatif": 0.54, "neutre": 0.23, "positif": 0.23}}
```

### Codes d'erreur
| Code | Cas |
|---|---|
| 200 | Succès |
| 422 | Charge utile invalide : champ manquant/vide/trop long, mauvais type MIME, fichier vide ou corrompu, aucun texte détecté (OCR/ASR), `/predict` sans champ ou avec les deux |
| 413 | Fichier trop volumineux (image > 5 Mo, audio > 25 Mo) |
| 503 | Service indisponible côté serveur : modèle non entraîné, Tesseract ou Whisper non installés |

## 5. Tests

```bash
pytest -v          # 55 tests (les 2 tests OCR réels sont ignorés si Tesseract-fra est absent)
```
L'ASR est *mocké* dans les tests (pas de modèle à télécharger) ; l'OCR est testé pour de vrai.

## 6. Résultats et limites (à lire)

| Évaluation | Résultat |
|---|---|
| Hold-out 25 % de `textes.csv` (n = 6) | accuracy 0,50 — **trop petit pour être significatif** |
| Leave-one-out sur `textes.csv` (n = 24) | accuracy 0,21 — les avis partagent très peu de vocabulaire |
| **Images** de bout en bout (OCR réel, jamais vues) | **7/9 = 0,78** |
| **Audios** (texte de référence, jamais vus) | 8/9 = 0,89 (ASR réel : `python evaluate_pipeline.py`) |

Avec seulement 24 textes, le classifieur reste fragile sur des formulations inédites. Pistes d'amélioration :
plus de données, ou embeddings de phrases multilingues / modèle `transformers` français (CamemBERT, etc.).
Le mapping des fichiers : `audios.csv` référence des `.wav` alors que les fichiers sont des `.mp3` ; l'API accepte les deux.

## 7. Captures d'écran à fournir

Placer dans `docs/screenshots/` : `swagger.png` (page `/docs` avec une requête exécutée) et
`streamlit_texte.png`, `streamlit_image.png`, `streamlit_audio.png`.
