import streamlit as st
import numpy as np
import soundfile as sf
import time
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧠 Music Interpretation Engine")

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

        # Limit to 30 sec
        data = data[:samplerate * 30]

        # =====================
        # MACHINE FEATURES
        # =====================
        energy = float(np.mean(data**2))
        brightness = float(np.mean(np.abs(np.fft.fft(data))))
        duration = len(data) / samplerate

        features = {
            "Energy": energy,
            "Brightness": brightness,
            "Duration": duration
        }

        # =====================
        # INTERPRETATION ENGINE
        # =====================
        interpretation = []

        if energy > 0.05:
            interpretation.append("High energy — likely expressive or dance-driven")
        else:
            interpretation.append("Low energy — calm or reflective")

        if brightness < 100:
            interpretation.append("Warm / traditional tonal quality")
        else:
            interpretation.append("Bright / sharp tonal quality")

        if mood == "Spiritual":
            interpretation.append("Listener perceives ceremonial or ritual elements")

        if culture == "Indigenous":
            interpretation.append("Strong Indigenous cultural identity")

        # =====================
        # DISPLAY
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
                features["Brightness"],
                features["Energy"],
                emotion,
                len(interpretation)
            ]
        })

        fig, ax = plt.subplots()
        ax.barh(brain_df["Region"], brain_df["Value"])
        st.pyplot(fig)

        # =====================
        # 🔄 MACHINE VS HUMAN
        # =====================
        st.subheader("🔄 Machine vs Human")

        compare_df = pd.DataFrame({
            "Feature": ["Energy", "Brightness"],
            "Machine": [energy, brightness],
            "Human Interpretation": [mood, culture]
        })

        st.dataframe(compare_df)

        # =====================
        # ⏱ PROCESS TIME
        # =====================
        end = time.time()
        st.write(f"⏱ Processing time: {round(end - start, 2)} sec")
