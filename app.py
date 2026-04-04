import streamlit as st
import numpy as np
import soundfile as sf
import time
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧠 Caribbean Sonic Humanities Engine")

# =========================
# 🎧 FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

# =========================
# 🧍 HUMAN QUESTIONNAIRE
# =========================
st.sidebar.header("🧍 Listener Perception")

mood = st.sidebar.selectbox("Mood", ["Calm", "Energetic", "Sad", "Spiritual", "Aggressive"])
culture = st.sidebar.selectbox("Cultural Feel", ["Indigenous", "Western", "Fusion", "Unknown"])
rhythm = st.sidebar.selectbox("Rhythm Feel", ["Steady", "Free", "Dance-like"])
emotion = st.sidebar.slider("Emotional Intensity", 1, 5, 3)

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
# PROCESS AUDIO
# =========================
if uploaded_file:

    if st.button("🔍 Analyze Audio"):

        start = time.time()

        # Load audio
        data, samplerate = sf.read(uploaded_file)

        # Convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Limit length (FAST)
        data = data[:samplerate * 30]

        # =====================
        # MACHINE FEATURES
        # =====================
        energy = float(np.mean(data**2))
        brightness = float(np.mean(np.abs(np.fft.fft(data))))
        tempo = estimate_tempo(data, samplerate)

        features = {
            "Energy": round(energy, 4),
            "Brightness": round(brightness, 2),
            "Tempo (BPM)": round(tempo, 1)
        }

        # =====================
        # INTERPRETATION ENGINE
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
        # DISPLAY PANELS
        # =====================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎧 Machine Features")
            st.write(features)

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
        # 🧠 BRAIN MODEL
        # =====================
        st.subheader("🧠 Brain Activation Model")

        brain_df = pd.DataFrame({
            "Region": ["Auditory Cortex", "Motor Cortex", "Limbic System", "Prefrontal Cortex"],
            "Value": [
                brightness,
                tempo,
                emotion,
                len(interpretation)
            ]
        })

        fig, ax = plt.subplots()
        ax.barh(brain_df["Region"], brain_df["Value"])
        ax.set_xlabel("Activation Level")
        st.pyplot(fig)

        # =====================
        # 🔄 MACHINE VS HUMAN
        # =====================
        st.subheader("🔄 Machine vs Human")

        compare_df = pd.DataFrame({
            "Feature": ["Energy", "Brightness", "Tempo"],
            "Machine": [energy, brightness, tempo],
            "Human Interpretation": [mood, culture, rhythm]
        })

        st.dataframe(compare_df)

        # =====================
        # ⏱ PROCESS TIME
        # =====================
        end = time.time()
        st.write(f"⏱ Processing time: {round(end - start, 2)} sec")
