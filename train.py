"""Étape 1 — Entraînement du classifieur de sentiment textuel.

Usage :  python train.py
Produit : models/model.pkl, models/vectorizer.pkl, models/metrics.json,
          reports/confusion_matrix.png, reports/classification_report.txt
"""
import json
import warnings

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from core.config import DATA_DIR, MODEL_PATH, MODELS_DIR, ROOT, VECTORIZER_PATH
from core.preprocessing import clean_text

warnings.filterwarnings("ignore")
LABELS = ["negatif", "neutre", "positif"]
SEED = 42
REPORTS = ROOT / "reports"


def new_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def new_model() -> LogisticRegression:
    # LogisticRegression : fournit predict_proba -> « confiance » dans la réponse de l'API
    return LogisticRegression(C=10, max_iter=2000)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_DIR / "textes.csv")
    X = df["texte"].map(clean_text)
    y = df["sentiment_ref"]
    print(f"Dataset textes.csv : {len(df)} lignes\n{y.value_counts().to_string()}\n")

    metrics: dict = {"n_samples": int(len(df))}

    # ---- 1) Hold-out stratifié 75/25 (6 exemples de test : indicatif seulement !) ----
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    vec, clf = new_vectorizer(), new_model()
    clf.fit(vec.fit_transform(X_tr), y_tr)
    pred = clf.predict(vec.transform(X_te))
    metrics["holdout"] = {"n_test": int(len(y_te)), "accuracy": accuracy_score(y_te, pred),
                          "f1_macro": f1_score(y_te, pred, average="macro")}
    report = classification_report(y_te, pred, labels=LABELS, zero_division=0)
    print(f"=== Hold-out (25 pct, n={len(y_te)}) ===\n{report}")

    cm = confusion_matrix(y_te, pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues", colorbar=False)
    plt.title("Matrice de confusion (hold-out)")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=140)
    plt.close()

    # ---- 2) Leave-one-out : estimation plus honnête sur si peu de données ----
    # (vectoriseur refit à chaque pli grâce au Pipeline -> pas de fuite d'information)
    for name, est in [("LogisticRegression", new_model()), ("LinearSVC", LinearSVC(C=1))]:
        loo_pred = cross_val_predict(make_pipeline(new_vectorizer(), est), X, y, cv=LeaveOneOut())
        acc = accuracy_score(y, loo_pred)
        metrics.setdefault("leave_one_out", {})[name] = acc
        print(f"Leave-one-out {name:20s} accuracy = {acc:.2f}")

    # ---- 3) Modèle final : réentraîné sur les 24 textes, puis sérialisé ----
    vec, clf = new_vectorizer(), new_model()
    clf.fit(vec.fit_transform(X), y)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vec, VECTORIZER_PATH)
    print(f"\nArtefacts sauvegardés : {MODEL_PATH.name}, {VECTORIZER_PATH.name}")

    # ---- 4) Jeux de test EXTERNES (jamais vus à l'entraînement) ----
    # Textes de référence des images (sortie OCR idéale) et des audios (transcription idéale).
    for name, csv, col in [("images", "images.csv", "texte_attendu_ocr"),
                           ("audios", "audios.csv", "transcription_reference")]:
        ext = pd.read_csv(DATA_DIR / csv)
        p = clf.predict(vec.transform(ext[col].map(clean_text)))
        acc = accuracy_score(ext["sentiment_ref"], p)
        metrics[f"externe_{name}_texte_ideal"] = acc
        print(f"Test externe {name:7s} (texte de référence, sans OCR/ASR) : accuracy = {acc:.2f}")

    (REPORTS / "classification_report.txt").write_text(report, encoding="utf-8")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
