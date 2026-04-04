import streamlit as st
import numpy as np
import soundfile as sf
import time

st.title("🧠 Music Interpretation Engine (Stable)")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file:

    if st.button("Analyze Audio"):

        start = time.time()

        # Load audio safely
        data, samplerate = sf.read(uploaded_file)

        # Convert stereo → mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # LIMIT LENGTH (fast)
        data = data[:samplerate * 30]

        # =====================
        # FEATURES (LIGHTWEIGHT)
        # =====================
        energy = float(np.mean(data**2))
        tempo_est = float(len(data) / samplerate)  # simple proxy
        brightness = float(np.mean(np.abs(np.fft.fft(data))))

        end = time.time()

        st.success("Processed!")

        st.write({
            "Energy": energy,
            "Tempo (proxy)": tempo_est,
            "Brightness": brightness
        })

        st.write(f"⏱ Processing time: {round(end - start, 2)} sec")
