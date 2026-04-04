import streamlit as st
import numpy as np
import soundfile as sf
import time
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧠 Caribbean Sonic Humanities Engine (Advanced)")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    size_mb = uploaded_file.size / (1024 * 1024)

    if size_mb > 10:
        st.error(f"File too large ({round(size_mb,1)} MB). Please upload under 10MB.")
        st.stop()

    st.success(f"File uploaded ({round(size_mb,1)} MB)")

# =========================
# HUMAN INPUT
# =========================
st.sidebar.header("🧍 Listener Perception")

mood = st.sidebar.selectbox("Mood", ["Calm", "Energetic", "Sad", "Spiritual", "Aggressive"])
culture = st.sidebar.selectbox("Cultural Feel", ["Indigenous", "Western", "Fusion", "Unknown"])
rhythm = st.sidebar.selectbox("Rhythm Feel", ["Steady", "Free", "Dance-like"])
emotion = st.sidebar.slider("Emotional Intensity", 1, 5, 3)

# =========================
# TEMPO FUNCTION
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
# MAIN PROCESS
# =========================
if uploaded_file:

    if st.button("🔍 Analyze Audio"):

        start = time.time()

        data, sr = sf.read(uploaded_file)

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        data = data[:sr * 30]

        # FEATURES
        energy = float(np.mean(data**2))
        brightness = float(np.mean(np.abs(np.fft.fft(data))))
        tempo = estimate_tempo(data, sr)

        features = np.array([energy, brightness, tempo])

        # =========================
        # CULTURAL CLASSIFICATION
        # =========================
        if energy < 0.04 and tempo < 90:
            classification = "Indigenous / Ceremonial"
        elif energy > 0.08 and tempo > 110:
            classification = "Western / Popular"
        else:
            classification = "Hybrid / Fusion"

        # =========================
        # DISPLAY
        # =========================
        st.subheader("🎧 Features")
        st.write({
            "Energy": energy,
            "Brightness": brightness,
            "Tempo": tempo
        })

        st.subheader("🌍 Cultural Classification")
        st.success(classification)

        # =========================
        # 🧠 3D BRAIN MODEL
        # =========================
        st.subheader("🧠 3D Brain Model")

        brain_x = [1, 2, 3, 4]
        brain_y = [2, 1, 3, 2]
        brain_z = [energy, tempo/100, emotion, len(features)]

        labels = ["Auditory", "Motor", "Limbic", "Cognitive"]

        fig = go.Figure(data=[go.Scatter3d(
            x=brain_x,
            y=brain_y,
            z=brain_z,
            mode='markers+text',
            text=labels,
            marker=dict(size=10)
        )])

        fig.update_layout(title="Brain Activation Map")
        st.plotly_chart(fig)

        # =========================
        # 🔬 RESEARCH MODE
        # =========================
        st.subheader("🔬 Feature Embedding")

        normalized = (features - np.mean(features)) / np.std(features)

        fig2, ax2 = plt.subplots()
        ax2.plot(normalized, marker='o')
        ax2.set_title("Feature Signature (Embedding Proxy)")
        st.pyplot(fig2)

        # =========================
        # ⏱ TIME
        # =========================
        end = time.time()
        st.write(f"⏱ Processing time: {round(end - start, 2)} sec")
