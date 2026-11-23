"""Étape 4 (Option B) — Mini-application web de test.

Lancer :  streamlit run streamlit_app.py      (l'API doit tourner : uvicorn main:app --reload)
"""
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Sentiment Multi-source", page_icon="💬", layout="centered")

STYLE = {"positif": ("😀", "#1a9850", "Positif"),
         "negatif": ("😡", "#d73027", "Négatif"),
         "neutre": ("😐", "#6c757d", "Neutre")}

st.title("💬 Classifieur de sentiment multi-source")
st.caption("Avis client en texte, image (OCR) ou audio (ASR) → positif / négatif / neutre")

api_url = st.sidebar.text_input("URL de l'API", os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")
try:
    ok = requests.get(f"{api_url}/health", timeout=3).ok
except requests.RequestException:
    ok = False
st.sidebar.success("API connectée ✅") if ok else st.sidebar.error("API injoignable ❌ — lancez uvicorn")
st.sidebar.markdown(f"[Documentation Swagger]({api_url}/docs)")


def call_api(path: str, **kwargs):
    """Appelle l'API ; retourne le JSON ou None (erreur affichée à l'utilisateur)."""
    try:
        r = requests.post(f"{api_url}{path}", timeout=180, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Impossible de joindre l'API : {exc}")
        return None
    if r.status_code != 200:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        if isinstance(detail, list):                      # erreurs de validation Pydantic -> messages lisibles
            detail = " ; ".join(d.get("msg", str(d)).replace("Value error, ", "") for d in detail)
        st.error(f"Erreur {r.status_code} : {detail}")
        return None
    return r.json()


def show(result: dict | None):
    if not result:
        return
    emoji, color, label = STYLE[result["sentiment_predit"]]
    st.markdown(
        f"<div style='padding:1rem;border-radius:12px;background:{color}1a;border:2px solid {color};"
        f"text-align:center;font-size:2rem;font-weight:700;color:{color}'>{emoji} {label}</div>",
        unsafe_allow_html=True)
    st.write("")
    st.progress(result["confiance"], text=f"Confiance : {result['confiance']:.0%}")
    st.caption(f"Source : **{result['source']}**")
    with st.expander("Texte analysé (extrait par OCR / ASR si besoin)", expanded=result["source"] != "texte"):
        st.write(result["texte_extrait"])
    st.bar_chart(pd.Series(result["probabilites"], name="probabilité"))


tab_txt, tab_img, tab_aud, tab_auto = st.tabs(["📝 Texte", "🖼️ Image", "🎙️ Audio", "🤖 Automatique"])

with tab_txt:
    text = st.text_area("Avis client", height=130, placeholder="Livraison rapide, produit conforme, je recommande !")
    if st.button("Analyser le texte", type="primary"):
        show(call_api("/predict/text", json={"texte": text}))

with tab_img:
    img = st.file_uploader("Capture d'écran / ticket scanné", type=["png", "jpg", "jpeg", "webp"], key="img")
    if img:
        st.image(img, use_container_width=True)
        if st.button("Analyser l'image", type="primary"):
            show(call_api("/predict/image", files={"file": (img.name, img.getvalue(), img.type)}))

with tab_aud:
    aud = st.file_uploader("Message vocal", type=["wav", "mp3", "ogg", "flac", "m4a"], key="aud")
    if aud:
        st.audio(aud)
        if st.button("Transcrire et analyser", type="primary"):
            with st.spinner("Transcription en cours…"):
                show(call_api("/predict/audio", files={"file": (aud.name, aud.getvalue(), aud.type)}))

with tab_auto:
    st.write("Déposez n'importe quel fichier **ou** saisissez du texte : l'API détecte le type.")
    auto_txt = st.text_input("Texte (optionnel)", key="auto_txt")
    auto_file = st.file_uploader("Fichier image ou audio (optionnel)", key="auto_file",
                                 type=["png", "jpg", "jpeg", "webp", "wav", "mp3", "ogg", "flac", "m4a"])
    if st.button("Analyser (détection auto)", type="primary"):
        if auto_file:
            show(call_api("/predict", files={"file": (auto_file.name, auto_file.getvalue(), auto_file.type)},
                          data={"text": auto_txt} if auto_txt else None))
        else:
            show(call_api("/predict", data={"text": auto_txt}))
