# tabs/model_monitoring.py
import streamlit as st
import pandas as pd
import numpy as np
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

def run():
    st.header("📈 Model Monitoring")

    # 1. Drift Detection
    st.subheader("🔄 Data Drift")
    if "preprocessed_df" in st.session_state and "inference_inputs" in st.session_state:
        train_lengths = st.session_state.preprocessed_df["text"].apply(lambda x: len(str(x).split()))
        infer_lengths = [len(text.split()) for text in st.session_state.inference_inputs]

        fig, ax = plt.subplots()
        sns.kdeplot(train_lengths, label="Training Text Lengths", ax=ax)
        sns.kdeplot(infer_lengths, label="Inference Text Lengths", ax=ax)
        ax.legend()
        ax.set_title("Word Count Distribution")
        st.pyplot(fig)

        wd = wasserstein_distance(train_lengths, infer_lengths)
        st.write(f"📊 Drift score (Wasserstein Distance): `{wd:.4f}`")
        if wd > 5.0:
            st.warning("⚠️ Significant data drift detected!")
    else:
        st.info("Need both training and inference data to detect drift.")

    # 2. Bias Monitoring (Example: Distribution of summary lengths)
    st.subheader("⚖️ Bias Monitoring")
    if "inference_outputs" in st.session_state:
        summary_lengths = [len(summary.split()) for summary in st.session_state.inference_outputs]
        fig2, ax2 = plt.subplots()
        sns.histplot(summary_lengths, kde=True, ax=ax2)
        ax2.set_title("Summary Length Distribution")
        st.pyplot(fig2)

        if np.mean(summary_lengths) < 10:
            st.warning("🚨 Summaries may be too short — possible length bias.")
    else:
        st.info("Generate some summaries to analyze bias.")

    # 3. Resource Monitoring
    st.subheader("🖥️ Resource Utilization")
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    st.metric("CPU Usage (%)", cpu)
    st.metric("Memory Usage (%)", mem)

    # 4. Safe Temperature Read (Windows-safe)
    st.subheader("🌡️ System Temperature")
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if hasattr(entry, "current"):
                            st.metric(f"{name} Temp (°C)", entry.current)
            else:
                st.info("Temperature sensors not available on this device.")
        except Exception as e:
            st.warning(f"Unable to read temperature sensors: {e}")
    else:
        st.info("Temperature monitoring not supported on this OS.")
