import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧠 Caribbean Sonic Humanities Engine")

# =========================
# MODE TOGGLE
# =========================
mode = st.sidebar.radio("Select Mode", ["Fast Analysis", "Research Mode"])

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])

# =========================
# HUMAN QUESTIONNAIRE
# =========================
st.sidebar.header("🧍 Listener Input")

mood = st.sidebar.selectbox("Mood", ["Calm", "Energetic", "Sad", "Spiritual", "Aggressive"])
culture = st.sidebar.selectbox("Cultural Feel", ["Indigenous", "Western", "Fusion", "Unknown"])
rhythm_feel = st.sidebar.selectbox("Rhythm Feel", ["Steady", "Free", "Dance-like"])
emotion_score = st.sidebar.slider("Emotional Intensity", 1, 5, 3)

# =========================
# PROCESS AUDIO
# =========================
if uploaded_file:

    with st.spinner("Processing audio..."):

        # 🔥 FIX: limit size + speed
        y, sr = librosa.load(uploaded_file, sr=22050, duration=60)

        # =========================
        # MACHINE FEATURES
        # =========================
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        rms = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))

        # fallback safety
        tempo = float(tempo) if tempo > 0 else 60
        rms = float(rms) if rms > 0 else 0.1
        centroid = float(centroid) if centroid > 0 else 1000
        mfcc = float(mfcc)

        features = {
            "tempo": tempo,
            "energy": rms,
            "brightness": centroid,
            "timbre": mfcc
        }

    st.success("Audio processed!")

    # =========================
    # INTERPRETATION ENGINE
    # =========================
    interpretation = []

    if tempo > 120:
        interpretation.append("High energy, dance-oriented")
    elif tempo < 80:
        interpretation.append("Slow, reflective or ceremonial")

    if centroid < 2000:
        interpretation.append("Warm / traditional tonal quality")

    if mood == "Spiritual":
        interpretation.append("Perceived as ceremonial or ritual")

    if culture == "Indigenous":
        interpretation.append("Strong Indigenous identity")

    # =========================
    # DISPLAY RESULTS
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎧 Machine Features")
        st.write(features)

    with col2:
        st.subheader("🧍 Human Perception")
        st.write({
            "Mood": mood,
            "Culture": culture,
            "Rhythm": rhythm_feel,
            "Emotion": emotion_score
        })

    st.subheader("🧠 Interpretation")
    for i in interpretation:
        st.write("•", i)

    # =========================
    # BRAIN MODEL (REAL)
    # =========================
    st.subheader("🧠 Brain Activation Model")

    brain_df = pd.DataFrame({
        "Region": ["Auditory Cortex", "Motor Cortex", "Limbic System", "Prefrontal Cortex"],
        "Value": [
            features["timbre"],
            features["tempo"],
            emotion_score,
            len(interpretation)
        ]
    })

    fig, ax = plt.subplots()
    ax.barh(brain_df["Region"], brain_df["Value"])
    st.pyplot(fig)

    # =========================
    # SIDE-BY-SIDE PANEL
    # =========================
    st.subheader("🔄 Machine vs Human")

    compare_df = pd.DataFrame({
        "Feature": ["Tempo", "Energy", "Brightness"],
        "Machine": [tempo, rms, centroid],
        "Human Interpretation": [rhythm_feel, mood, culture]
    })

    st.dataframe(compare_df)

    # =========================
    # RESEARCH MODE
    # =========================
    if mode == "Research Mode":
        st.subheader("📊 Research Mode (Advanced)")

        # 🔥 PCA-like visualization (simple version)
        data = np.array([tempo, rms, centroid, mfcc])
        norm = (data - np.mean(data)) / np.std(data)

        fig2, ax2 = plt.subplots()
        ax2.plot(norm)
        ax2.set_title("Feature Pattern Signature")
        st.pyplot(fig2)

        st.write("This represents a simplified feature embedding profile.")

    # =========================
    # EXPORT
    # =========================
    st.subheader("📥 Export Results")

    export_df = pd.DataFrame([{
        **features,
        "mood": mood,
        "culture": culture,
        "rhythm": rhythm_feel,
        "emotion": emotion_score
    }])

    st.download_button("Download CSV", export_df.to_csv(index=False), "results.csv")
