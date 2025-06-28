# tabs/deployment_inference.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import os

MODEL_DIR = "./saved_model"

def run():
    st.header("Model Deployment & Inference")

    # --- Save model if trained but not saved ---
    if "trained_model" in st.session_state and "tokenizer" in st.session_state:
        if st.button("📦 Save Model"):
            st.session_state.trained_model.save_pretrained(MODEL_DIR)
            st.session_state.tokenizer.save_pretrained(MODEL_DIR)
            st.success("Model and tokenizer saved to disk.")

    # --- Load model if not already in session state ---
    if ("trained_model" not in st.session_state or "tokenizer" not in st.session_state) and os.path.exists(MODEL_DIR):
        try:
            model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            st.session_state.trained_model = model
            st.session_state.tokenizer = tokenizer
            st.success("✅ Model and tokenizer loaded from disk.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    if "trained_model" not in st.session_state or "tokenizer" not in st.session_state:
        st.warning("Please train and save a model first.")
        return

    # --- Inference section ---
    st.markdown("### Input Text")
    user_input = st.text_area("Enter the text to summarize", height=200)

    max_len = st.slider("Max summary length", 10, 150, 60)
    min_len = st.slider("Min summary length", 5, 50, 10)
    do_sample = st.checkbox("Use sampling (creative output)", value=False)

    if st.button("🧠 Generate Summary"):
        with st.spinner("Generating summary..."):
            summarizer = pipeline(
                "summarization",
                model=st.session_state.trained_model,
                tokenizer=st.session_state.tokenizer,
                device=0 if st.session_state.get("use_gpu", False) else -1
            )
            output = summarizer("summarize: " + user_input, max_length=max_len, min_length=min_len, do_sample=do_sample)
            summary = output[0]["summary_text"]
            st.success("Generated Summary:")
            st.write(summary)
