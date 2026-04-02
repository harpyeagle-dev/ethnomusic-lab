import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Caribbean Sonic Humanities Lab", layout="wide")

st.title("🌴 Caribbean Sonic Humanities Lab")
st.caption("A Cognitive Interface for Sound, Culture, and Embodied Listening")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload an audio recording", type=["wav", "mp3"])

# =========================
# DEFAULT VALUES (prevents crashes)
# =========================
tempo_val = 0.0
centroid_mean = 0.0
mfcc_mean = 0.0

# =========================
# AUDIO PROCESSING
# =========================
if uploaded_file is not None:

    try:
        y, sr = librosa.load(uploaded_file, sr=None)

        # ---- Tempo (Pulse Density) ----
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        if isinstance(tempo, np.ndarray):
            tempo_val = float(tempo.item()) if tempo.size == 1 else float(np.mean(tempo))
        else:
            tempo_val = float(tempo)

        # ---- Spectral Brightness ----
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = float(np.mean(centroid))

        # ---- Timbre (MFCC proxy) ----
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfcc))

        st.success("Audio processed successfully")

    except Exception as e:
        st.error(f"Audio processing failed: {e}")

# =========================
# MACHINE HEARING LAYER
# =========================
st.subheader("🤖 Machine Hearing (Signal-Level Interpretation)")

col1, col2, col3 = st.columns(3)

col1.metric("Pulse Density (Tempo)", round(tempo_val, 2))
col2.metric("Spectral Brightness", round(centroid_mean, 2))
col3.metric("Timbral Texture", round(mfcc_mean, 2))

# =========================
# HUMAN PERCEPTION LAYER
# =========================
st.subheader("👂 Caribbean Listener Response")

with st.form("listener_form"):

    groove = st.slider("Groove Intensity (felt rhythm)", 0, 100, 50)

    movement = st.selectbox(
        "Embodied Response",
        ["Stillness", "Sway", "Dance", "Jump", "Ritual movement"]
    )

    familiarity = st.selectbox(
        "Cultural Familiarity",
        ["Strongly Caribbean", "Somewhat familiar", "Unfamiliar"]
    )

    emotion = st.selectbox(
        "Emotional Register",
        ["Joy", "Melancholy", "Spiritual", "Energetic", "Other"]
    )

    submitted = st.form_submit_button("Submit Response")

# =========================
# COGNITIVE-CULTURAL MAPPING
# =========================
if uploaded_file is not None:

    st.subheader("🌍 Cognitive-Cultural Activation")

    familiarity_score = 1.0 if familiarity == "Strongly Caribbean" else 0.5

    cognitive_df = pd.DataFrame({
        "Domain": ["Movement", "Emotion", "Cultural Memory"],
        "Activation": [
            min(tempo_val / 180, 1.0),
            min(abs(mfcc_mean) / 200, 1.0),
            familiarity_score
        ]
    })

    fig, ax = plt.subplots()
    ax.barh(cognitive_df["Domain"], cognitive_df["Activation"])
    ax.set_xlim(0, 1)
    ax.set_title("Cognitive Activation (Caribbean Frame)")
    st.pyplot(fig)

# =========================
# HUMAN vs MACHINE (TENSION MODEL)
# =========================
if submitted:

    st.subheader("⚖️ Machine vs Lived Experience")

    comparison_df = pd.DataFrame({
        "Aspect": ["Rhythmic Feel", "Energy"],
        "Machine": [
            tempo_val,
            min(tempo_val / 2, 100)
        ],
        "Human": [
            groove,
            groove
        ]
    })

    st.dataframe(comparison_df)

    fig2, ax2 = plt.subplots()
    x = np.arange(len(comparison_df["Aspect"]))

    ax2.bar(x - 0.2, comparison_df["Human"], width=0.4, label="Human")
    ax2.bar(x + 0.2, comparison_df["Machine"], width=0.4, label="Machine")

    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df["Aspect"])
    ax2.legend()

    st.pyplot(fig2)

# =========================
# EXPORT RESULTS
# =========================
st.subheader("📁 Export Analysis")

results = pd.DataFrame({
    "tempo": [tempo_val],
    "brightness": [centroid_mean],
    "timbre": [mfcc_mean]
})

st.download_button(
    "Download CSV",
    results.to_csv(index=False),
    "caribbean_sonic_analysis.csv"
)
