# tabs/data_loading.py
import streamlit as st
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def run():
    st.header("Data Loading")

    data_source = st.radio("Choose Data Source", ["Preloaded: billsum", "Upload your own"], index=0)

    def visualize_core_plots(df):
        df["length"] = df["text"].apply(lambda x: len(str(x).split()))
        df["summary_length"] = df["summary"].apply(lambda x: len(str(x).split()))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Text Length Distribution")
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            sns.histplot(df["length"], bins=30, kde=True, ax=ax1)
            ax1.set_xlabel("Word Count")
            ax1.set_ylabel("Frequency")
            st.pyplot(fig1)

        with col2:
            st.markdown("#### Scatter: Text vs Summary Length")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.scatterplot(x="length", y="summary_length", data=df, ax=ax2)
            ax2.set_xlabel("Text Length")
            ax2.set_ylabel("Summary Length")
            st.pyplot(fig2)

    # Preloaded Dataset Logic
    if data_source == "Preloaded: billsum":
        if "raw_df" not in st.session_state and st.button("Load Preloaded Dataset"):
            try:
                ds = load_dataset("billsum", split="train")
                st.session_state.raw_df = pd.DataFrame(ds)
                st.session_state.data_saved = False
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")

        if "raw_df" in st.session_state:
            df = st.session_state.raw_df
            st.success(f"Loaded {len(df)} samples from billsum")
            st.dataframe(df.head())
            visualize_core_plots(df)

            if "data_saved" not in st.session_state:
                st.session_state.data_saved = False

            if st.button("✅ Save Loaded Data") and not st.session_state.data_saved:
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
                train_df["split"] = "train"
                test_df["split"] = "test"
                df = pd.concat([train_df, test_df]).reset_index(drop=True)

                st.session_state.cleaned_df = df
                st.session_state.data_saved = True
                st.success("✅ Dataset saved to session for next stage.")
                st.session_state.active_tab = "Preprocessing"
                st.rerun()
            elif st.session_state.data_saved:
                st.info("✅ Dataset already saved. You can proceed to the next tab.")

    # Upload CSV Logic
    else:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if "text" not in df.columns or "summary" not in df.columns:
                    st.error("CSV must contain 'text' and 'summary' columns.")
                    return
                st.session_state.raw_df = df
                st.success("Successfully loaded uploaded dataset.")
                st.dataframe(df.head())
                visualize_core_plots(df)

                if "data_saved" not in st.session_state:
                    st.session_state.data_saved = False

                if st.button("✅ Save Loaded Data") and not st.session_state.data_saved:
                    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
                    train_df["split"] = "train"
                    test_df["split"] = "test"
                    df = pd.concat([train_df, test_df]).reset_index(drop=True)

                    st.session_state.cleaned_df = df
                    st.session_state.data_saved = True
                    st.success("✅ Dataset saved to session for next stage.")
                    st.session_state.active_tab = "Preprocessing"
                elif st.session_state.data_saved:
                    st.info("✅ Dataset already saved. You can proceed to the next tab.")

            except Exception as e:
                st.error(f"Error reading CSV: {e}")
