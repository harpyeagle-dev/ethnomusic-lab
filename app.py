import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openl3
import soundfile as sf
from sklearn.decomposition import PCA
import plotly.graph_objects as go

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
embedding_mean = np.zeros(512)

# =========================
# AUDIO PROCESSING
# =========================
if uploaded_file is not None:

    try:
        # ---- Load audio ----
        y, sr = librosa.load(uploaded_file, sr=None)

        # ---- Tempo ----
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo_val = float(tempo.item()) if tempo.size == 1 else float(np.mean(tempo))
        else:
            tempo_val = float(tempo)

        # ---- Spectral Brightness ----
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = float(np.mean(centroid))

        # ---- Timbre (MFCC) ----
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfcc))

        # ---- OpenL3 Embeddings ----
        audio, sr2 = sf.read(uploaded_file)
        embeddings, timestamps = openl3.get_audio_embedding(
            audio, sr2,
            content_type="music",
            embedding_size=512
        )

        embedding_mean = np.mean(embeddings, axis=0)

        st.success("✅ Audio processed successfully")

    except Exception as e:
        st.error(f"Audio processing failed: {e}")

# =========================
# MACHINE HEARING
# =========================
st.subheader("🤖 Machine Hearing (Signal-Level)")

col1, col2, col3 = st.columns(3)
col1.metric("Pulse Density (Tempo)", round(tempo_val, 2))
col2.metric("Spectral Brightness", round(centroid_mean, 2))
col3.metric("Timbral Texture", round(mfcc_mean, 2))

# =========================
# HUMAN RESPONSE
# =========================
st.subheader("👂 Caribbean Listener Response")

with st.form("listener_form"):
    groove = st.slider("Groove Intensity", 0, 100, 50)
    movement = st.selectbox(
        "Embodied Response",
        ["Stillness", "Sway", "Dance", "Jump", "Ritual"]
    )
    familiarity = st.selectbox(
        "Cultural Familiarity",
        ["Strongly Caribbean", "Somewhat", "Unfamiliar"]
    )
    emotion = st.selectbox(
        "Emotional Register",
        ["Joy", "Melancholy", "Spiritual", "Energetic", "Other"]
    )

    submitted = st.form_submit_button("Submit")

# =========================
# PCA EMBEDDING SPACE
# =========================
if uploaded_file is not None:

    st.subheader("🔬 Cultural Embedding Space (PCA)")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig_pca, ax_pca = plt.subplots()
    ax_pca.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    ax_pca.set_title("Audio Identity Space")
    st.pyplot(fig_pca)

# =========================
# 3D BRAIN MODEL
# =========================
if uploaded_file is not None:

    st.subheader("🧠 3D Cognitive Model")

    brain_regions = ["Motor", "Auditory", "Emotion"]
    values = [
        min(tempo_val / 180, 1.0),
        min(centroid_mean / 5000, 1.0),
        min(abs(mfcc_mean) / 200, 1.0)
    ]

    fig3d = go.Figure(data=[go.Scatter3d(
        x=[1, 2, 3],
        y=[1, 2, 3],
        z=values,
        mode='markers+text',
        text=brain_regions,
        marker=dict(size=10)
    )])

    fig3d.update_layout(title="Cognitive Activation Model")
    st.plotly_chart(fig3d)

# =========================
# HUMAN VS MACHINE
# =========================
if submitted:

    st.subheader("⚖️ Machine vs Lived Experience")

    comparison_df = pd.DataFrame({
        "Aspect": ["Rhythm", "Energy"],
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
# RESEARCH EXPORT
# =========================
st.subheader("📄 Research Export")

research_df = pd.DataFrame({
    "tempo": [tempo_val],
    "brightness": [centroid_mean],
    "timbre": [mfcc_mean],
    "embedding_1": [embedding_mean[0]],
    "embedding_2": [embedding_mean[1]]
})

st.dataframe(research_df)

st.download_button(
    "Download Research CSV",
    research_df.to_csv(index=False),
    "caribbean_sonic_research.csv"
)
