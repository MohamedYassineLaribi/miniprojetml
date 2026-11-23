# Guide de tests et d'explication du projet

## Partie A — Vérifier que tout marche (≈ 15 min)

| # | Commande / action | Résultat attendu |
|---|---|---|
| 1 | `pip install -r requirements.txt` puis `tesseract --list-langs` et `ffmpeg -version` | `fra` listé ; ffmpeg répond |
| 2 | `python train.py` | « Artefacts sauvegardés » ; fichiers dans `models/` et `reports/` ; test externe images ≈ 0,78 / audios ≈ 0,89 |
| 3 | `pytest -v` | **55 passed** (53 + 2 OCR réels si Tesseract-fra est installé) |
| 4 | `uvicorn main:app --reload` | « Application startup complete » |
| 5 | Ouvrir `http://127.0.0.1:8000/docs` | Swagger avec 4 routes POST + `/health` |
| 6 | Swagger → `POST /predict/text` → *Try it out* : `{"texte": "La batterie ne tient pas deux heures, une arnaque"}` | 200, `negatif`, confiance ≈ 0,83 |
| 7 | `POST /predict/text` avec `{"texte": ""}` | **422** |
| 8 | `POST /predict/image` avec `data_test_v2/images/I02.png` | 200, `texte_extrait` contient « RÉCLAMATION … injoignable », `negatif` |
| 9 | `POST /predict/audio` avec `A01.mp3` (1er appel : téléchargement Whisper, patientez) | 200, transcription française, `positif` attendu |
| 10 | `POST /predict/image` avec un fichier `.txt` | **422** |
| 11 | `POST /predict` (auto) avec une image, puis un audio, puis seulement du texte | `source` = `image`, `audio`, `texte` |
| 12 | `streamlit run streamlit_app.py` : tester les 4 onglets | Badge coloré + barre de confiance + texte extrait |
| 13 | `python evaluate_pipeline.py` | Accuracy images (≈ 0,78) et audios, matrice de confusion, liste des erreurs |

**Résultats attendus sur les 9 images (OCR réel)** — erreurs de classifieur sur I01 (positif→neutre) et I09 (neutre→négatif) ; les 7 autres sont correctes.

**Si ça échoue**
- `503 OCR indisponible` → Tesseract absent ou sans pack `fra` (Windows : variable `TESSERACT_CMD`).
- `503 openai-whisper n'est pas installé` ou erreur ffmpeg → `pip install openai-whisper` + installer ffmpeg.
- `503 Modèle absent` → lancer `python train.py`.
- Avertissement de version scikit-learn au chargement du `.pkl` → relancer `python train.py`.
- Streamlit « API injoignable » → l'API doit tourner, et l'URL dans la barre latérale doit être la bonne.

## Partie B — Explication des choix techniques (pour la soutenance)

**1. Pourquoi une architecture en cascade ?**
Une seule logique de décision (le classifieur texte) pour trois modalités : on n'entraîne pas trois modèles, on maintient un seul modèle, et chaque nouvelle modalité n'est qu'un « convertisseur vers texte ». Compromis : les erreurs d'OCR/ASR se propagent au classifieur, et on perd les informations non textuelles (ton de voix, mise en forme, étoiles ★ dans l'image).

**2. Prétraitement (`core/preprocessing.py`)**
Minuscules, suppression URL/noms de fichiers/chiffres/ponctuation, suppression des accents (robuste aux erreurs OCR), suppression des stop-words français. *Choix délibéré* : on **garde** « ne, pas, non, jamais, rien, très, trop, sans » car ils portent la négation et l'intensité (« ne fonctionne pas » ≠ « fonctionne »). La **même fonction** sert à l'entraînement et à l'inférence (évite le *train/serve skew*).

**3. Modèle : TF-IDF (1-2 grammes) + Régression logistique**
Adapté aux petits jeux de données, rapide, interprétable. La régression logistique fournit `predict_proba`, d'où le champ `confiance`. Un SVM linéaire donne la même accuracy en leave-one-out (0,21) mais sans probabilités natives. Les bigrammes capturent « pas du tout », « service client ».

**4. Évaluation — le point à maîtriser absolument**
- Le hold-out (6 exemples) est donné par exigence de l'énoncé mais **n'est pas fiable** : un seul exemple change l'accuracy de 17 points.
- Le *leave-one-out* (0,21, sous le hasard de 0,33) révèle que les 24 avis partagent presque aucun mot : le modèle ne peut pas généraliser depuis ce seul fichier à des phrases très différentes.
- Les 9 images et 9 audios jamais vus à l'entraînement forment un **vrai test externe** : ≈ 0,78 (images, OCR réel) et ≈ 0,89 (audios). Ils partagent du vocabulaire métier (livraison, service client, remboursement…) avec les textes, d'où un meilleur score.
- Je n'ai **pas** choisi les hyperparamètres en regardant ces tests externes (n = 9 : sur-ajustement garanti). La config est fixée *a priori*.
- Honnêteté : ne présentez pas 0,78/0,89 comme une performance « en production » ; c'est une estimation sur 9 exemples chacun.

**5. OCR (`core/ocr.py`)**
Pillow : niveaux de gris, auto-contraste, agrandissement ×2 des petites images ; Tesseract en `lang=fra`, `--psm 6` (bloc de texte uniforme). Sans le pack `fra`, les accents sont mal reconnus (vous pouvez comparer avec `OCR_LANG=eng`).

**6. ASR (`core/asr.py`)**
Whisper (`base` par défaut), langue forcée `fr`, chargement **paresseux** et mis en cache : l'API démarre même sans Whisper, et ne charge le modèle qu'une fois. Whisper lit le MP3/WAV via ffmpeg.

**7. API (`main.py`)**
- Routes `async` ; traitements CPU (OCR, Whisper) dans `run_in_threadpool` pour ne pas bloquer le serveur.
- Pydantic : `TextRequest` (texte 1-5000 car., non vide après `strip`), `PredictionResponse` (sentiment dans {positif, negatif, neutre}, confiance ∈ [0,1]).
- `core/validation.py` : validation du MIME déclaré, de l'extension, de la **signature binaire** (le MIME est fourni par le client, donc falsifiable), de la taille (lecture bornée), et lisibilité réelle de l'image.
- Codes : 422 (contenu invalide, comme demandé), 413 (trop gros), 503 (dépendance serveur absente) : on distingue « faute du client » et « faute du serveur ».
- `/predict` (auto) : champs de formulaire `text` **ou** `file` ; le type est déduit du MIME → extension → signature.

**8. Tests (`tests/test_api.py`)**
55 tests : unitaires (nettoyage, schémas), intégration (4 routes, cas d'erreurs 422/413/503/405). L'ASR est mocké (rapidité, pas de GPU), l'OCR est réel (ignoré si non installé). Test de sensibilité effectué : en supprimant la vérification MIME, les tests concernés échouent bien.

## Partie C — Script de démonstration (≈ 5 min)

1. *Architecture* (30 s) : schéma du README.
2. *Entraînement* : montrer la sortie de `train.py` et dire clairement les limites (partie B-4).
3. *Swagger* : texte négatif → texte positif → **erreur 422** volontaire (texte vide).
4. *Image* : `I02.png` → montrer `texte_extrait` (la preuve que l'OCR fonctionne) puis le sentiment.
5. *Audio* : `A01.mp3` → idem avec la transcription.
6. *Streamlit* : l'onglet Automatique avec une image puis un audio.
7. `pytest -v` : 55 verts. Conclusion : limites + pistes (plus de données, CamemBERT).

## Partie D — Questions probables du jury

- **Pourquoi pas de transformers ?** Jeu de données minuscule, déploiement léger et rapide ; TF-IDF+LR est un bon *baseline*. Piste : embeddings multilingues pour mieux généraliser.
- **Pourquoi le leave-one-out est si bas ?** Aucun chevauchement de vocabulaire entre avis d'une même classe ; en LOO sur classes équilibrées, un modèle sans signal tombe même sous 33 %.
- **Que se passe-t-il si l'image n'a pas de texte ?** OCR vide → 422 « Aucun texte détecté ».
- **Le MIME suffit-il pour la sécurité ?** Non, d'où l'extension, la signature binaire, `Image.verify()` et la limite de taille.
- **Pourquoi des `.pkl` séparés ?** Exigence de l'énoncé ; modèle et vectoriseur doivent être versionnés ensemble (même version de scikit-learn).
- **Comment passer à l'échelle ?** Plusieurs workers uvicorn/gunicorn, file de tâches pour l'ASR, modèle Whisper plus petit ou service dédié.
- **Pourquoi `audios.csv` mentionne `.wav` alors que les fichiers sont `.mp3` ?** Incohérence du dataset ; `evaluate_pipeline.py` apparie par nom sans extension, et l'API accepte wav, mp3, ogg, flac, m4a.
