"""Évaluation de BOUT EN BOUT sur images/ et audios/ (jamais vus à l'entraînement).

Usage :  python evaluate_pipeline.py            (images + audios si whisper installé)
         python evaluate_pipeline.py --no-audio
Mesure : accuracy du sentiment, qualité de l'OCR/ASR (similarité de texte), erreurs détaillées.
"""
import argparse
import difflib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from core import pipelines
from core.asr import ASRUnavailableError
from core.config import DATA_DIR, ROOT
from core.preprocessing import clean_text

LABELS = ["negatif", "neutre", "positif"]


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, clean_text(a), clean_text(b)).ratio()


def evaluate(name: str, csv: str, ref_col: str, folder: str, fn, exts) -> dict | None:
    df = pd.read_csv(DATA_DIR / csv)
    rows, y_true, y_pred = [], [], []
    for _, r in df.iterrows():
        stem = Path(r["fichier"]).stem                      # audios.csv cite .wav alors que les fichiers sont .mp3
        path = next((DATA_DIR / folder / f"{stem}{e}" for e in exts if (DATA_DIR / folder / f"{stem}{e}").exists()), None)
        if path is None:
            continue
        try:
            out = fn(path)
        except ASRUnavailableError as exc:
            print(f"[{name}] ignoré : {exc}")
            return None
        y_true.append(r["sentiment_ref"]); y_pred.append(out["sentiment_predit"])
        rows.append({"fichier": path.name, "attendu": r["sentiment_ref"], "predit": out["sentiment_predit"],
                     "confiance": out["confiance"], "similarite_texte": round(similarity(out["texte_extrait"], r[ref_col]), 3),
                     "texte_extrait": out["texte_extrait"]})
    res = pd.DataFrame(rows)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n===== {name.upper()} — accuracy sentiment = {acc:.2f} ({sum(a == b for a, b in zip(y_true, y_pred))}/{len(y_true)}) "
          f"| similarité moyenne extraction = {res.similarite_texte.mean():.2f}")
    print("Matrice de confusion (lignes = attendu, colonnes = prédit ; ordre negatif/neutre/positif)")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))
    errs = res[res.attendu != res.predit]
    if len(errs):
        print("Erreurs :\n" + errs[["fichier", "attendu", "predit", "confiance"]].to_string(index=False))
    return {"accuracy": acc, "similarite_moyenne": float(res.similarite_texte.mean()), "details": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()
    report = {"images": evaluate("images", "images.csv", "texte_attendu_ocr", "images", pipelines.predict_image, [".png", ".jpg"])}
    if not args.no_audio:
        report["audios"] = evaluate("audios", "audios.csv", "transcription_reference", "audios",
                                    pipelines.predict_audio, [".mp3", ".wav"])
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "pipeline_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
