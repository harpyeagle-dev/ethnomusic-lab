import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import soundfile as sf

# Optional OpenL3
try:
    import openl3
    OPENL3_AVAILABLE = True
except:
    OPENL3_AVAILABLE = False

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Caribbean Sonic Humanities Lab", layout="wide")

st.title("🌴 Caribbean Sonic Humanities Lab")
st.caption("A Cognitive Interface for Sound, Culture, and Embodied Listening")

# =========================
# INPUT
# =========================
uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"])

# =========================
# STATE (single source of truth)
# =========================
features = {
    "tempo": 0.0,
    "brightness": 0.0,
    "timbre": 0.0
}

embedding_mean = np.zeros(512)
embeddings = None


# =========================
# AUDIO PIPELINE (ONLY PLACE WE COMPUTE)
# =========================
if uploaded_file is not None:

    try:
        # ---- Load audio ----
        y, sr = librosa.load(uploaded_file, sr=None)

        # ---- Tempo ----
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo_val = float(tempo.item()) if tempo.size == 1 else float(np.mean(tempo))
        else:
            tempo_val = float(tempo)

        # ---- Brightness ----
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness_val = float(np.mean(centroid))

        # ---- Timbre ----
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        timbre_val = float(np.mean(mfcc))

        # Save to state
        features["tempo"] = tempo_val
        features["brightness"] = brightness_val
        features["timbre"] = timbre_val

        # ---- Embeddings (optional) ----
        if OPENL3_AVAILABLE:
            audio, sr2 = sf.read(uploaded_file)
            embeddings, _ = openl3.get_audio_embedding(audio, sr2, content_type="music")
            embedding_mean = np.mean(embeddings, axis=0)

        st.success("Audio processed successfully")

    except Exception as e:
        st.error(f"Processing failed: {e}")

# =========================
# MACHINE HEARING
# =========================
st.subheader("🤖 Machine Hearing")

col1, col2, col3 = st.columns(3)
col1.metric("Pulse Density", round(features["tempo"], 2))
col2.metric("Spectral Brightness", round(features["brightness"], 2))
col3.metric("Timbral Texture", round(features["timbre"], 2))

# =========================
# HUMAN RESPONSE
# =========================
st.subheader("👂 Caribbean Listener")

groove = 50
movement = "Still"
familiarity = "Caribbean"
emotion = "Joy"

with st.form("listener"):

    groove = st.slider("Groove", 0, 100, 50)
    movement = st.selectbox("Movement", ["Still", "Sway", "Dance", "Ritual"])
    familiarity = st.selectbox("Familiarity", ["Caribbean", "Mixed", "Foreign"])
    emotion = st.selectbox("Emotion", ["Joy", "Melancholy", "Spiritual", "Energy"])

    submitted = st.form_submit_button("Submit")

if submitted:
    st.subheader("⚖️ Human vs Machine")

    compare_df = pd.DataFrame({
        "Aspect": ["Rhythm", "Energy"],
        "Machine": [features["tempo"], features["tempo"] / 2],
        "Human": [groove, groove]
    })

    st.dataframe(compare_df)

# =========================
# COGNITIVE MODEL
# =========================
st.subheader("🧠 Cognitive Mapping")

motor = min(features["tempo"] / 180, 1.0)
auditory = min(features["brightness"] / 5000, 1.0)
emotion_val = min(abs(features["timbre"]) / 200, 1.0)

brain_df = pd.DataFrame({
    "Region": ["Motor", "Auditory", "Emotion"],
    "Activation": [motor, auditory, emotion_val]
})

fig, ax = plt.subplots()
ax.barh(brain_df["Region"], brain_df["Activation"])
ax.set_xlim(0, 1)
st.pyplot(fig)

# =========================
# 3D BRAIN
# =========================
st.subheader("🧠 3D Brain Model")

fig3d = go.Figure(data=[go.Scatter3d(
    x=[1, 2, 3],
    y=[1, 2, 3],
    z=[motor, auditory, emotion_val],
    mode='markers+text',
    text=["Motor", "Auditory", "Emotion"],
    marker=dict(size=10)
)])

st.plotly_chart(fig3d)

# =========================
# PCA (if embeddings available)
# =========================
if embeddings is not None:
    st.subheader("🔬 Cultural Embedding Space")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig_pca, ax_pca = plt.subplots()
    ax_pca.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    st.pyplot(fig_pca)

# =========================
# HUMAN VS MACHINE
# =========================
if submitted:

    st.subheader("⚖️ Human vs Machine")

    compare_df = pd.DataFrame({
        "Aspect": ["Rhythm", "Energy"],
        "Machine": [features["tempo"], features["tempo"] / 2],
        "Human": [groove, groove]
    })

    st.dataframe(compare_df)

# =========================
# EXPORT
# =========================
st.subheader("📄 Export")

export_df = pd.DataFrame({
    "tempo": [features["tempo"]],
    "brightness": [features["brightness"]],
    "timbre": [features["timbre"]],
    "emb_1": [embedding_mean[0]],
    "emb_2": [embedding_mean[1]]
})

st.download_button(
    "Download CSV",
    export_df.to_csv(index=False),
    "caribbean_sonic_data.csv"
)
