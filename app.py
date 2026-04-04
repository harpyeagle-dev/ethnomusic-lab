import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import io
import time

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Caribbean Sonic Humanities Lab", layout="wide")

st.title("🌴 Caribbean Sonic Humanities Lab")
st.caption("End-to-End Cognitive Music Analysis")

# =========================
# SESSION STATE (CRITICAL)
# =========================
if "features" not in st.session_state:
    st.session_state.features = {"tempo": 0.0, "brightness": 0.0, "timbre": 0.0}

if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0.0

if "processed" not in st.session_state:
    st.session_state.processed = False

# =========================
# INPUT
# =========================
uploaded_file = st.file_uploader("Upload WAV audio (any size)", type=["wav"])

# =========================
# ANALYSIS CONTROLS
# =========================
clip_duration = st.slider("Analysis Window (seconds)", 5, 60, 20)

start_sec = 0
if uploaded_file is not None:
    # temporary estimate for slider range
    start_sec = st.slider("Start Position (seconds)", 0, 120, 0)

process = st.button("🔍 Analyze Audio")

# =========================
# PROCESSING
# =========================
if uploaded_file is not None and process:

    start_time = time.time()

    try:
        # Load audio correctly
        file_bytes = uploaded_file.getvalue()
        sr, y = wavfile.read(io.BytesIO(file_bytes))

        y = y.astype(float)

        # Convert stereo → mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # =========================
        # SMART SEGMENT SELECTION
        # =========================
        start_sample = int(start_sec * sr)
        end_sample = start_sample + int(clip_duration * sr)

        if end_sample > len(y):
            end_sample = len(y)

        y = y[start_sample:end_sample]

        st.info(f"Processing segment: {start_sec}s → {start_sec + clip_duration}s")

        # =========================
        # FEATURE EXTRACTION (ROBUST)
        # =========================
        energy = np.abs(y)
        tempo_val = float(np.mean(energy)) * 500

        spectrum = np.abs(np.fft.fft(y))
        brightness_val = float(np.mean(spectrum))

        timbre_val = float(np.std(y)) * 5

        # Save state
        st.session_state.features = {
            "tempo": max(tempo_val, 1),
            "brightness": max(brightness_val, 1),
            "timbre": max(timbre_val, 1)
        }

        st.session_state.processing_time = time.time() - start_time
        st.session_state.processed = True

        st.success(f"✅ Processed in {st.session_state.processing_time:.2f} seconds")

    except Exception as e:
        st.error(f"Processing failed: {e}")

# =========================
# ALWAYS DISPLAY RESULTS
# =========================
features = st.session_state.features

st.subheader("🤖 Machine Hearing")

c1, c2, c3 = st.columns(3)
c1.metric("Pulse Density", round(features["tempo"], 2))
c2.metric("Brightness", round(features["brightness"], 2))
c3.metric("Timbre", round(features["timbre"], 2))

# =========================
# 🧠 BRAIN MODEL (ALWAYS VISIBLE)
# =========================
st.subheader("🧠 Cognitive Brain Model")

motor = min(features["tempo"] / 200, 1.0)
auditory = min(features["brightness"] / 20000, 1.0)
emotion = min(features["timbre"] / 20, 1.0)

fig, ax = plt.subplots()

x = [0.3, 0.6, 0.5]
y_pos = [0.6, 0.7, 0.3]
labels = ["Motor", "Auditory", "Emotion"]
vals = [motor, auditory, emotion]

for i in range(3):
    ax.scatter(x[i], y_pos[i], s=vals[i]*3000 + 200)
    ax.text(x[i], y_pos[i], labels[i], ha='center')

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_title("Brain Activation Map")
ax.axis('off')

st.pyplot(fig)

# =========================
# HUMAN RESPONSE
# =========================
st.subheader("👂 Listener Response")

with st.form("listener_form"):
    groove = st.slider("Groove (felt rhythm)", 0, 100, 50)
    emotion_human = st.selectbox("Emotion", ["Joy", "Calm", "Energy", "Spiritual"])
    submit = st.form_submit_button("Submit")

if submit:
    st.subheader("⚖️ Human vs Machine")

    df = pd.DataFrame({
        "Aspect": ["Rhythm"],
        "Machine": [features["tempo"]],
        "Human": [groove]
    })

    st.dataframe(df)

# =========================
# PROCESSING TIME
# =========================
st.subheader("⏱ Processing Time")
st.metric("Seconds", round(st.session_state.processing_time, 2))

# =========================
# EXPORT
# =========================
st.subheader("📄 Export")

export_df = pd.DataFrame({
    "tempo": [features["tempo"]],
    "brightness": [features["brightness"]],
    "timbre": [features["timbre"]],
    "processing_time": [st.session_state.processing_time]
})

st.download_button(
    "Download CSV",
    export_df.to_csv(index=False),
    "caribbean_sonic_results.csv"
)
