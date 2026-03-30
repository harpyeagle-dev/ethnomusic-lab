import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="EthnoMusic Cognitive Lab", layout="wide")

st.title("🧠 EthnoMusic Lab: Cognitive Interpretation System")

# =========================
# AUDIO UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=None)
    
    st.audio(uploaded_file)

    st.subheader("🔊 Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    # =========================
    # FEATURE EXTRACTION
    # =========================
    st.subheader("🎛️ MIR Feature Extraction")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    rhythmic_variance = np.var(onset_env)

    features = {
        "tempo": tempo,
        "rms": rms,
        "brightness": spectral_centroid,
        "tension": spectral_bandwidth,
        "roughness": zero_crossing,
        "rhythmic_variance": rhythmic_variance
    }

    st.write(features)

    # =========================
    # PERCEPTUAL MAPPING (BRAIN LAYER)
    # =========================
    st.subheader("🧠 Cognitive Interpretation")

    def interpret(features):
        energy = (features["tempo"] * 0.3) + (features["rms"] * 100)
        brightness = features["brightness"] / 1000
        tension = features["tension"] / 1000
        stability = 1 / (features["rhythmic_variance"] + 0.01)
        groove = features["tempo"] * (1 - features["rhythmic_variance"])

        return {
            "Energy": energy,
            "Brightness": brightness,
            "Tension": tension,
            "Stability": stability,
            "Groove": groove
        }

    perception = interpret(features)
    st.write(perception)

    # =========================
    # RADAR CHART
    # =========================
    st.subheader("📊 Perceptual Radar")

    labels = list(perception.keys())
    values = list(perception.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig2, ax2 = plt.subplots(subplot_kw=dict(polar=True))
    ax2.plot(angles, values)
    ax2.fill(angles, values, alpha=0.3)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels)

    st.pyplot(fig2)

    # =========================
    # EMOTIONAL TIMELINE
    # =========================
    st.subheader("🌊 Energy Over Time")

    rms_frame = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms_frame)), sr=sr)

    fig3, ax3 = plt.subplots()
    ax3.plot(times, rms_frame)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Energy (RMS)")
    st.pyplot(fig3)

    # =========================
    # TEXT INTERPRETATION
    # =========================
    st.subheader("📝 Cognitive Summary")

    def generate_text(p):
        if p["Energy"] > 50:
            energy_desc = "high energy and strong activation"
        else:
            energy_desc = "moderate to low energy"

        if p["Stability"] > 10:
            rhythm_desc = "high rhythmic stability"
        else:
            rhythm_desc = "variable rhythmic structure"

        if p["Tension"] > 2:
            tension_desc = "increased tension and complexity"
        else:
            tension_desc = "relatively low tension"

        return f"""
        This piece exhibits {energy_desc}, suggesting strong motor engagement. 
        The rhythm shows {rhythm_desc}, indicating how predictable or syncopated the structure is. 
        The timbral qualities reflect {tension_desc}, shaping emotional intensity. 
        Overall, the piece can be cognitively interpreted as a {'dance-driven' if p['Groove'] > 20 else 'reflective'} musical experience.
        """

    st.write(generate_text(perception))
