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
# INPUT
# =========================
uploaded_file = st.file_uploader("Upload WAV audio (short clip recommended)", type=["wav"])
process = st.button("🔍 Analyze Audio")

# =========================
# STATE
# =========================
features = {"tempo": 0.0, "brightness": 0.0, "timbre": 0.0}
processing_time = 0.0

# =========================
# PROCESSING PIPELINE
# =========================
if uploaded_file is not None and process:

    start = time.time()

    try:
        # ---- Load file correctly ----
        file_bytes = uploaded_file.getvalue()
        sr, y = wavfile.read(io.BytesIO(file_bytes))

        y = y.astype(float)

        # Mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Trim for speed (30 sec max)
        max_samples = sr * 30
        if len(y) > max_samples:
            y = y[:max_samples]

        # =========================
        # FEATURE EXTRACTION (ROBUST)
        # =========================

        # ENERGY (for rhythm)
        energy = np.abs(y)
        tempo_val = float(np.mean(energy)) * 500

        # FREQUENCY CONTENT
        spectrum = np.abs(np.fft.fft(y))
        brightness_val = float(np.mean(spectrum))

        # VARIABILITY (timbre)
        timbre_val = float(np.std(y)) * 5

        # Ensure non-zero baseline
        features["tempo"] = max(tempo_val, 1)
        features["brightness"] = max(brightness_val, 1)
        features["timbre"] = max(timbre_val, 1)

        processing_time = time.time() - start

        st.success(f"✅ Processed in {processing_time:.2f}s")

    except Exception as e:
        st.error(f"Processing failed: {e}")

elif uploaded_file is None:
    st.info("Upload a WAV file and click Analyze")

# =========================
# MACHINE OUTPUT
# =========================
st.subheader("🤖 Machine Hearing")

c1, c2, c3 = st.columns(3)
c1.metric("Pulse Density", round(features["tempo"], 2))
c2.metric("Brightness", round(features["brightness"], 2))
c3.metric("Timbre", round(features["timbre"], 2))

# =========================
# NORMALIZATION (KEY FIX)
# =========================
motor = min(features["tempo"] / 200, 1.0)
auditory = min(features["brightness"] / 20000, 1.0)
emotion = min(features["timbre"] / 20, 1.0)

# =========================
# 🧠 BRAIN VISUAL (CLEAR + VISIBLE)
# =========================
st.subheader("🧠 Cognitive Brain Model")

fig, ax = plt.subplots()

regions = ["Motor", "Auditory", "Emotion"]
values = [motor, auditory, emotion]

x = [0.3, 0.6, 0.5]
y = [0.6, 0.7, 0.3]

for i, r in enumerate(regions):
    ax.scatter(x[i], y[i], s=values[i]*3000 + 200)
    ax.text(x[i], y[i], r, ha='center')

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_title("Brain Activation")
ax.axis('off')

st.pyplot(fig)

# =========================
# HUMAN INPUT
# =========================
st.subheader("👂 Listener Response")

with st.form("listener"):
    groove = st.slider("Groove", 0, 100, 50)
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
# EXPORT
# =========================
st.subheader("📄 Export")

export = pd.DataFrame({
    "tempo": [features["tempo"]],
    "brightness": [features["brightness"]],
    "timbre": [features["timbre"]],
    "processing_time": [processing_time]
})

st.download_button("Download CSV", export.to_csv(index=False), "results.csv")
