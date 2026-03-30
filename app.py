import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd

st.set_page_config(page_title="EthnoMusic Cognitive Lab", layout="wide")

st.title("🧠 EthnoMusic Lab: Cognitive Interpretation System")

# =========================
# CULTURE SELECTOR
# =========================
st.sidebar.header("🌍 Cultural Context")

culture = st.sidebar.selectbox(
    "Select Cultural Model",
    ["Neutral", "Indigenous (Baboon Dance)", "Afro-Caribbean", "Western Classical"]
)

# =========================
# AUDIO UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if uploaded_file is not None:

    # =========================
    # LOAD AUDIO
    # =========================
    y, sr = sf.read(uploaded_file)

    # Convert to mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    st.audio(uploaded_file)

    # =========================
    # WAVEFORM
    # =========================
    st.subheader("🔊 Waveform")

    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # =========================
    # FEATURE EXTRACTION
    # =========================
    st.subheader("🎛️ MIR Feature Extraction")

    duration = len(y) / sr

    # Energy (RMS)
    rms = np.sqrt(np.mean(y**2))

    # Zero Crossing Rate (roughness)
    zcr = np.mean(np.abs(np.diff(np.sign(y))))

    # Spectral centroid (brightness)
    spectrum = np.abs(np.fft.fft(y))
    freqs = np.fft.fftfreq(len(spectrum), 1/sr)
    spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-6)

    # Tempo (very rough proxy)
    tempo = 60 / duration if duration > 0 else 0

    features = {
        "Tempo": tempo,
        "Energy (RMS)": rms,
        "Brightness": spectral_centroid,
        "Roughness": zcr
    }

    st.write(features)

    # =========================
    # PERCEPTUAL MAPPING
    # =========================
    st.subheader("🧠 Cognitive Interpretation")

    def interpret(features):
        energy = features["Tempo"] * 0.3 + features["Energy (RMS)"] * 100
        brightness = features["Brightness"] / 1000
        tension = features["Roughness"] * 10
        stability = 1 / (tension + 0.1)
        groove = energy * stability

        return {
            "Energy": energy,
            "Brightness": brightness,
            "Tension": tension,
            "Stability": stability,
            "Groove": groove
        }

    perception = interpret(features)

    # =========================
    # CULTURAL MODIFIER
    # =========================
    def apply_culture(p, culture):
        if culture == "Indigenous (Baboon Dance)":
            p["Groove"] *= 1.5
            p["Energy"] *= 1.2
        elif culture == "Afro-Caribbean":
            p["Groove"] *= 1.4
        elif culture == "Western Classical":
            p["Tension"] *= 1.3
        return p

    perception = apply_culture(perception, culture)

    st.write(perception)

    # =========================
    # RADAR CHART
    # =========================
    st.subheader("📊 Perceptual Profile")

    labels = list(perception.keys())
    values = list(perception.values())

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig2, ax2 = plt.subplots(subplot_kw=dict(polar=True))
    ax2.plot(angles, values)
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels)

    st.pyplot(fig2)

    # =========================
    # ENERGY OVER TIME
    # =========================
    st.subheader("🌊 Energy Over Time")

    frame_size = 1024
    hop = 512

    rms_frames = []
    times = []

    for i in range(0, len(y) - frame_size, hop):
        frame = y[i:i+frame_size]
        rms_frames.append(np.sqrt(np.mean(frame**2)))
        times.append(i / sr)

    fig3, ax3 = plt.subplots()
    ax3.plot(times, rms_frames)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Energy")

    st.pyplot(fig3)

    # =========================
    # TEXT INTERPRETATION
    # =========================
    st.subheader("📝 Cognitive Summary")

    def generate_text(p, culture):
        energy_desc = "high energy" if p["Energy"] > 50 else "moderate energy"
        groove_desc = "strong groove" if p["Groove"] > 20 else "low groove"
        tension_desc = "high tension" if p["Tension"] > 1 else "low tension"

        return f"""
        This piece exhibits {energy_desc}, suggesting active listener engagement.
        The rhythmic structure produces a {groove_desc}, indicating movement or dance potential.
        The timbral qualities reflect {tension_desc}, shaping emotional intensity.
        Within the {culture} framework, the piece can be interpreted as a 
        {'ritualistic' if p['Groove'] > 20 else 'reflective'} musical experience.
        """

    st.write(generate_text(perception, culture))
