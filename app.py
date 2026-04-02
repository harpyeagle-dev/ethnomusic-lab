import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ethnomusic Lab", layout="wide")

st.title("🎧 Ethnomusicology Lab")
st.subheader("Analyze Music + Compare Human vs Machine Interpretation")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# =========================
# DEFAULT VALUES (CRITICAL)
# =========================
tempo_val = 0.0
centroid_mean = 0.0
mfcc_mean = 0.0

# =========================
# AUDIO PROCESSING
# =========================
if uploaded_file is not None:

    try:
        # Load audio
        y, sr = librosa.load(uploaded_file, sr=None)

        # Tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo)

        # Spectral Centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = float(np.mean(centroid))

        # MFCC (timbre proxy)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfcc))

        st.success("✅ Audio processed successfully")

    except Exception as e:
        st.error(f"Audio processing failed: {e}")

# =========================
# MACHINE ANALYSIS DISPLAY
# =========================
st.subheader("🤖 Machine Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("Tempo (BPM)", round(tempo_val, 2))
col2.metric("Brightness", round(centroid_mean, 2))
col3.metric("Timbre (MFCC)", round(mfcc_mean, 2))

# =========================
# BRAIN MAPPING (SIMPLIFIED)
# =========================
st.subheader("🧠 Brain Interpretation (Machine)")

brain_data = pd.DataFrame({
    "Region": ["Motor Cortex", "Auditory Cortex", "Emotion"],
    "Activation": [
        min(tempo_val / 200, 1.0),
        min(centroid_mean / 5000, 1.0),
        min(abs(mfcc_mean) / 200, 1.0)
    ]
})

fig, ax = plt.subplots()
ax.barh(brain_data["Region"], brain_data["Activation"])
ax.set_xlim(0, 1)
ax.set_title("Brain Activation (Machine)")
st.pyplot(fig)

# =========================
# HUMAN QUESTIONNAIRE
# =========================
st.subheader("👤 Human Interpretation")

with st.form("user_input"):

    perceived_tempo = st.slider("Perceived Tempo", 0, 200, 100)
    perceived_energy = st.slider("Energy Level", 0, 100, 50)
    perceived_emotion = st.selectbox(
        "Emotion",
        ["Happy", "Sad", "Calm", "Aggressive", "Other"]
    )

    submitted = st.form_submit_button("Submit")

# =========================
# HUMAN VS MACHINE COMPARISON
# =========================
if submitted:

    st.subheader("🔄 Human vs Machine Comparison")

    human_vs_machine = pd.DataFrame({
        "Feature": ["Tempo", "Energy"],
        "Human": [perceived_tempo, perceived_energy],
        "Machine": [
            tempo_val,
            min(tempo_val / 2, 100)  # simple mapping
        ]
    })

    st.dataframe(human_vs_machine)

    fig2, ax2 = plt.subplots()
    x = np.arange(len(human_vs_machine["Feature"]))

    ax2.bar(x - 0.2, human_vs_machine["Human"], width=0.4, label="Human")
    ax2.bar(x + 0.2, human_vs_machine["Machine"], width=0.4, label="Machine")

    ax2.set_xticks(x)
    ax2.set_xticklabels(human_vs_machine["Feature"])
    ax2.legend()

    st.pyplot(fig2)

# =========================
# EXPORT DATA
# =========================
st.subheader("📁 Export")

results = pd.DataFrame({
    "tempo": [tempo_val],
    "centroid": [centroid_mean],
    "mfcc": [mfcc_mean]
})

st.download_button(
    "Download Results CSV",
    results.to_csv(index=False),
    "analysis.csv"
)
