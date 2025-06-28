# tabs/preprocessing.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run():
    st.header("Preprocessing")

    if "cleaned_df" not in st.session_state:
        st.warning("Please load and save a dataset in the Data Loading tab first.")
        return

    df = st.session_state.cleaned_df.copy()
    st.markdown("### Sample Selection")

    sample_size = st.slider("How many samples to keep?", min_value=100, max_value=len(df), value=min(1000, len(df)), step=100)

    st.markdown("### Preprocessing Options")
    options = st.multiselect("Choose Preprocessing Steps", [
        "Remove Duplicates",
        "Lowercase Text",
        "Remove Empty Rows",
        "Normalize Whitespace"
    ], default=["Remove Duplicates", "Lowercase Text"])

    if "preprocessing_applied" not in st.session_state:
        st.session_state.preprocessing_applied = False

    if st.button("Apply Preprocessing") or st.session_state.preprocessing_applied:
        st.session_state.preprocessing_applied = True

        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        if "Remove Duplicates" in options:
            df.drop_duplicates(subset=["text", "summary"], inplace=True)

        if "Remove Empty Rows" in options:
            df.dropna(subset=["text", "summary"], inplace=True)

        if "Lowercase Text" in options:
            df["text"] = df["text"].str.lower()
            df["summary"] = df["summary"].str.lower()

        if "Normalize Whitespace" in options:
            df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
            df["summary"] = df["summary"].str.replace(r"\s+", " ", regex=True).str.strip()

        # Save preprocessed
        st.session_state.preprocessed_df = df

        st.markdown("### Preview After Preprocessing")
        st.dataframe(df.head())

        st.markdown("### Visualization")
        df["text_len"] = df["text"].apply(lambda x: len(str(x).split()))
        df["summary_len"] = df["summary"].apply(lambda x: len(str(x).split()))

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(df["text_len"], bins=30, kde=True, ax=ax1)
            ax1.set_title("Text Length After Preprocessing")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.histplot(df["summary_len"], bins=30, kde=True, ax=ax2)
            ax2.set_title("Summary Length After Preprocessing")
            st.pyplot(fig2)

        if st.button("✅ Save Preprocessed Dataset"):
            st.success("Saved preprocessed dataset for model training.")
            st.session_state.active_tab = "Model Training"
