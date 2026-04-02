# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from io import BytesIO
import datetime

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="EthnoMusic Lab", layout="wide")

st.title("🎧 Ethnomusic Analysis Dashboard")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# =========================
# QUESTIONNAIRE
# =========================
st.sidebar.header("🎼 Listener Perception")

genre = st.sidebar.selectbox("What genre do you hear?", 
                            ["Indigenous", "Reggae", "Jazz", "Classical", "Other"])

mood = st.sidebar.selectbox("Mood", 
                           ["Happy", "Sad", "Energetic", "Calm", "Spiritual"])

instruments = st.sidebar.text_input("Instruments heard")

rhythm = st.sidebar.selectbox("Rhythm type",
                             ["Steady", "Syncopated", "Free", "Complex"])

submit_q = st.sidebar.button("Save Response")

# =========================
# SAVE QUESTIONNAIRE
# =========================
if submit_q:
    data = {
        "timestamp": datetime.datetime.now(),
        "genre": genre,
        "mood": mood,
        "instruments": instruments,
        "rhythm": rhythm
    }

    df = pd.DataFrame([data])

    try:
        df.to_csv("responses.csv", mode='a', header=not pd.io.common.file_exists("responses.csv"), index=False)
        st.sidebar.success("Saved!")
    except:
        st.sidebar.error("Error saving")

# =========================
# ANALYZE BUTTON
# =========================
if uploaded_file is not None:
    if st.button("🔍 Analyze this recording"):

        # LOAD AUDIO
        y, sr = librosa.load(uploaded_file, sr=22050)

        st.success("Audio Loaded!")

        # =========================
        # FEATURE EXTRACTION
        # =========================
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))

        st.subheader("📊 Audio Features")
        col1, col2, col3 = st.columns(3)

        tempo_val = float(np.mean(tempo))
        brightness_val = float(spectral_centroid)
        energy_val = float(rms)

        col1.metric("Tempo", f"{tempo_val:.2f} BPM")
        col2.metric("Brightness", f"{brightness_val:.2f}")
        col3.metric("Energy", f"{energy_val:.4f}")

        # =========================
        # WAVEFORM
        # =========================
        st.subheader("Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

        # =========================
        # SPECTROGRAM
        # =========================
        st.subheader("Spectrogram")
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))

        fig2, ax2 = plt.subplots()
        img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
        fig2.colorbar(img, ax=ax2)
        st.pyplot(fig2)

        # =========================
        # BRAIN-LIKE VIEW (SIMPLIFIED)
        # =========================
        st.subheader("🧠 Brain Interpretation View")

        tempo_val = float(np.mean(tempo))
        brightness_val = float(spectral_centroid)
        energy_val = float(rms) * 1000

        brain_data = {
        "Rhythm (Tempo)": tempo_val,
        "Timbre (Brightness)": brightness_val,
        "Energy": energy_val
}

         brain_df = pd.DataFrame({
         "Feature": list(brain_data.keys()),
         "Value": [float(v) for v in brain_data.values()]
})

         fig3, ax3 = plt.subplots()

         ax3.barh(
         brain_df["Feature"].astype(str),
         brain_df["Value"].astype(float)
)

ax3.set_title("Cognitive Audio Mapping")

st.pyplot(fig3)
        # =========================
        # COMPARE WITH USER INPUT
        # =========================
        st.subheader("🔄 Machine vs Human Interpretation")

        st.write("**User Input**")
        st.json({
            "Genre": genre,
            "Mood": mood,
            "Rhythm": rhythm,
            "Instruments": instruments
        })

        st.write("**Machine Output**")
        st.json({
            "Tempo": float(tempo),
            "Brightness": float(spectral_centroid),
            "Energy": float(rms)
        })
