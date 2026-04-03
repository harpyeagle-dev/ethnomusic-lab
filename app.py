import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.io import wavfile
import io
import time

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Caribbean Sonic Humanities Lab", layout="wide")

st.title("🌴 Caribbean Sonic Humanities Lab")
st.caption("A Cognitive Interface for Sound, Culture, and Embodied Listening")

# =========================
# INPUT
# =========================
uploaded_file = st.file_uploader("Upload WAV audio", type=["wav"])
process = st.button("🔍 Analyze Audio")

# =========================
# STATE
# =========================
features = {
    "tempo": 0.0,
    "brightness": 0.0,
    "timbre": 0.0
}

processing_time = 0.0

# =========================
# AUDIO PROCESSING
# =========================
if uploaded_file is not None and process:
    start_time = time.time()

    try:
        bytes_data = uploaded_file.getvalue()
        sr, y = wavfile.read(io.BytesIO(bytes_data))

        y = y.astype(float)

        # Convert stereo → mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Speed optimization (important)
        if len(y) > 500000:
            y = y[:500000]

        # =========================
        # FEATURE EXTRACTION (FIXED)
        # =========================
        energy = np.abs(y)
        tempo_val = float(np.mean(energy)) * 1000

        spectrum = np.abs(np.fft.fft(y))
        brightness_val = float(np.mean(spectrum))

        timbre_val = float(np.std(y)) * 10

        features["tempo"] = tempo_val
        features["brightness"] = brightness_val
        features["timbre"] = timbre_val

        processing_time = time.time() - start_time

        st.success(f"✅ Audio processed in {processing_time:.2f} seconds")

    except Exception as e:
        st.error(f"Processing failed: {e}")

elif uploaded_file is None:
    st.info("Upload a WAV file and click Analyze")

# =========================
# MACHINE HEARING
# =========================
st.subheader("🤖 Machine Hearing")

col1, col2, col3 = st.columns(3)
col1.metric("Pulse Density", round(features["tempo"], 2))
col2.metric("Spectral Brightness", round(features["brightness"], 2))
col3.metric("Timbral Texture", round(features["timbre"], 2))

# =========================
# PROCESSING TIME DISPLAY
# =========================
st.subheader("⏱ Processing Time")
st.metric("Analysis Duration (seconds)", f"{processing_time:.2f}")

# =========================
# HUMAN RESPONSE
# =========================
st.subheader("👂 Caribbean Listener")

with st.form("listener"):
    groove = st.slider("Groove", 0, 100, 50)
    movement = st.selectbox("Movement", ["Still", "Sway", "Dance", "Ritual"])
    familiarity = st.selectbox("Familiarity", ["Caribbean", "Mixed", "Foreign"])
    emotion = st.selectbox("Emotion", ["Joy", "Melancholy", "Spiritual", "Energy"])

    submitted = st.form_submit_button("Submit")

# =========================
# NORMALIZATION (FIXED)
# =========================
motor = min(features["tempo"] / 500, 1.0)
auditory = min(features["brightness"] / 50000, 1.0)
emotion_val = min(features["timbre"] / 50, 1.0)

# =========================
# 🧠 2D BRAIN MAP
# =========================
st.subheader("🧠 Brain Activation Map")

brain_layout = {
    "Motor Cortex": (0.3, 0.6),
    "Auditory Cortex": (0.6, 0.7),
    "Limbic System": (0.5, 0.3)
}

activations = {
    "Motor Cortex": motor,
    "Auditory Cortex": auditory,
    "Limbic System": emotion_val
}

fig, ax = plt.subplots()

for region, (x, y) in brain_layout.items():
    val = activations[region]
    ax.scatter(x, y, s=val * 2000 + 100, alpha=0.7)
    ax.text(x, y, region, ha='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Cognitive Activation (Caribbean Listening)")
ax.axis('off')

st.pyplot(fig)

# =========================
# 🧠 3D BRAIN MODEL
# =========================
st.subheader("🧠 3D Cognitive Model")

x = [1, 2, 3]
y = [2, 2, 2]
z = [motor, auditory, emotion_val]

sizes = [v * 40 + 10 for v in z]

fig3d = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers+text',
    text=["Motor", "Auditory", "Emotion"],
    marker=dict(
        size=sizes,
        color=z,
        colorscale='Viridis',
        opacity=0.9
    )
)])

fig3d.update_layout(
    title="3D Cognitive Activation",
    scene=dict(zaxis=dict(range=[0, 1]))
)

st.plotly_chart(fig3d)

# =========================
# HUMAN VS MACHINE
# =========================
if submitted:
    st.subheader("⚖️ Human vs Machine")

    compare_df = pd.DataFrame({
        "Aspect": ["Rhythm", "Energy"],
        "Machine": [features["tempo"], features["tempo"] / 2],
        "Human": [groove, groove]
    })

    st.dataframe(compare_df)

# =========================
# EXPORT
# =========================
st.subheader("📄 Export")

export_df = pd.DataFrame({
    "tempo": [features["tempo"]],
    "brightness": [features["brightness"]],
    "timbre": [features["timbre"]],
    "processing_time_sec": [processing_time]
})

st.download_button(
    "Download CSV",
    export_df.to_csv(index=False),
    "caribbean_sonic_data.csv"
)
