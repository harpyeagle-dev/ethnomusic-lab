import streamlit as st
import numpy as np
import soundfile as sf
import pandas as pd
import time
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧠 Caribbean Sonic Humanities Engine")

# =========================
# 🧍 HUMAN INPUT
# =========================
st.sidebar.header("🧍 Listener Perception")

mood = st.sidebar.selectbox("Mood", ["Calm", "Energetic", "Sad", "Spiritual", "Aggressive"])
culture = st.sidebar.selectbox("Cultural Feel", ["Indigenous", "Western", "Fusion", "Unknown"])
rhythm = st.sidebar.selectbox("Rhythm Feel", ["Steady", "Free", "Dance-like"])
emotion = st.sidebar.slider("Emotional Intensity", 1, 5, 3)

# =========================
# 🎧 FILE UPLOAD (ONLY ONE)
# =========================
uploaded_file = st.file_uploader("Upload WAV file (max 10MB)", type=["wav"], key="upload1")

# =========================
# 🔥 TEMPO FUNCTION
# =========================
def estimate_tempo(signal, sr):
    diff = np.diff(signal)
    envelope = np.abs(diff)

    window = int(sr * 0.05)
    envelope = np.convolve(envelope, np.ones(window)/window, mode='same')

    peaks = np.where(envelope > np.mean(envelope) * 1.5)[0]

    if len(peaks) < 2:
        return 60

    intervals = np.diff(peaks) / sr
    avg_interval = np.mean(intervals)

    if avg_interval == 0:
        return 60

    bpm = 60 / avg_interval
    return float(min(max(bpm, 40), 200))

# =========================
# 🚀 PROCESSING BLOCK
# =========================
if uploaded_file:

    size_mb = uploaded_file.size / (1024 * 1024)

    if size_mb > 10:
        st.error(f"File too large ({round(size_mb,1)} MB). Please upload under 10MB.")
        st.stop()

    st.success(f"File uploaded ({round(size_mb,1)} MB)")

    if st.button("🔍 Analyze Audio"):

        start = time.time()
        st.info("Processing...")

        # LOAD AUDIO
        data, sr = sf.read(uploaded_file)

        # MONO
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # LIMIT LENGTH
        data = data[:sr * 30]

        # =====================
        # FEATURES
        # =====================
        energy = float(np.mean(data**2))
        brightness = float(np.mean(np.abs(np.fft.fft(data))))
        tempo = estimate_tempo(data, sr)

        # =====================
        # CLASSIFICATION
        # =====================
        if energy < 0.04 and tempo < 90:
            classification = "Indigenous / Ceremonial"
        elif energy > 0.08 and tempo > 110:
            classification = "Western / Popular"
        else:
            classification = "Hybrid / Fusion"

        # =====================
        # INTERPRETATION
        # =====================
        interpretation = []

        if energy > 0.05:
            interpretation.append("High energy — expressive or dance-driven")
        else:
            interpretation.append("Low energy — calm or reflective")

        if brightness < 100:
            interpretation.append("Warm / traditional tonal quality")
        else:
            interpretation.append("Bright / sharp tonal quality")

        if tempo > 120:
            interpretation.append("Fast tempo — dance or high activity")
        elif tempo < 80:
            interpretation.append("Slow tempo — ceremonial or reflective")

        if mood == "Spiritual":
            interpretation.append("Perceived as ceremonial or ritual")

        if culture == "Indigenous":
            interpretation.append("Strong Indigenous cultural identity")

        # =====================
        # DISPLAY FEATURES
        # =====================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎧 Machine Features")
            st.write({
                "Energy": round(energy, 4),
                "Brightness": round(brightness, 2),
                "Tempo": round(tempo, 1)
            })

        with col2:
            st.subheader("🧍 Human Perception")
            st.write({
                "Mood": mood,
                "Culture": culture,
                "Rhythm": rhythm,
                "Emotion": emotion
            })

        # =====================
        # INTERPRETATION OUTPUT
        # =====================
        st.subheader("🧠 Interpretation")
        for i in interpretation:
            st.write("•", i)

        # =====================
        # 🌍 CLASSIFICATION OUTPUT
        # =====================
        st.subheader("🌍 Cultural Classification")
        st.success(classification)

        # =====================
        # 🧠 3D BRAIN MODEL
        # =====================
        st.subheader("🧠 3D Brain Model")

        fig = go.Figure(data=[go.Scatter3d(
            x=[1, 2, 3, 4],
            y=[2, 1, 3, 2],
            z=[brightness, tempo/100, emotion, len(interpretation)],
            mode='markers+text',
            text=["Auditory", "Motor", "Limbic", "Cognitive"],
            marker=dict(size=10)
        )])

        fig.update_layout(title="Brain Activation Map")
        st.plotly_chart(fig)

        # =====================
        # ⏱ PROCESS TIME
        # =====================
        end = time.time()
        st.write(f"⏱ Processing time: {round(end - start, 2)} sec")
