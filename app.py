import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.io import wavfile

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Caribbean Sonic Humanities Lab", layout="wide")

st.title("🌴 Caribbean Sonic Humanities Lab")
st.caption("A Cognitive Interface for Sound, Culture, and Embodied Listening")

# =========================
# INPUT
# =========================
uploaded_file = st.file_uploader("Upload audio (WAV recommended)", type=["wav", "mp3"])

# =========================
# STATE (single source of truth)
# =========================
features = {
    "tempo": 0.0,
    "brightness": 0.0,
    "timbre": 0.0
}

# =========================
# PROCESSING (LIGHTWEIGHT, NO LIBROSA)
# =========================
if uploaded_file is not None:
    try:
        sr, y = wavfile.read(uploaded_file)

        y = y.astype(float)

        # Convert stereo → mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # ---- Tempo proxy (energy-based) ----
        energy = np.abs(y)
        tempo_val = float(np.mean(energy)) * 100

        # ---- Brightness (FFT) ----
        spectrum = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(spectrum))
        brightness_val = float(np.mean(np.abs(freqs)))

        # ---- Timbre proxy ----
        timbre_val = float(np.std(y))

        # Save features
        features["tempo"] = tempo_val
        features["brightness"] = brightness_val
        features["timbre"] = timbre_val

        st.success("✅ Audio processed successfully")

    except Exception as e:
        st.error(f"Processing failed: {e}")

else:
    st.info("Upload an audio file to begin analysis")

# =========================
# MACHINE HEARING (ALWAYS VISIBLE)
# =========================
st.subheader("🤖 Machine Hearing")

col1, col2, col3 = st.columns(3)
col1.metric("Pulse Density", round(features["tempo"], 2))
col2.metric("Spectral Brightness", round(features["brightness"], 4))
col3.metric("Timbral Texture", round(features["timbre"], 4))

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

# =========================
# COGNITIVE MAPPING (ALWAYS VISIBLE)
# =========================
st.subheader("🧠 Cognitive Mapping")

motor = min(features["tempo"] / 180, 1.0)
auditory = min(features["brightness"] * 1000, 1.0)
emotion_val = min(abs(features["timbre"]) / 1000, 1.0)

brain_df = pd.DataFrame({
    "Region": ["Motor", "Auditory", "Emotion"],
    "Activation": [motor, auditory, emotion_val]
})

fig, ax = plt.subplots()
ax.barh(brain_df["Region"], brain_df["Activation"])
ax.set_xlim(0, 1)
st.pyplot(fig)

# =========================
# 3D BRAIN MODEL
# =========================
st.subheader("🧠 3D Cognitive Model")

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
    "timbre": [features["timbre"]]
})

st.download_button(
    "Download CSV",
    export_df.to_csv(index=False),
    "caribbean_sonic_data.csv"
)

# =========================
# DEBUG (REMOVE LATER)
# =========================
st.write("DEBUG FEATURES:", features)
