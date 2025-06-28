# tabs/evaluation.py
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def run():
    st.header("Model Evaluation")

    if "trained_model" not in st.session_state or "tokenizer" not in st.session_state:
        st.warning("Please train the model first.")
        return

    if "dataset" not in st.session_state:
        if "preprocessed_df" in st.session_state and "split" in st.session_state.preprocessed_df.columns:
            df = st.session_state.preprocessed_df
            from datasets import Dataset, DatasetDict
            st.session_state.dataset = DatasetDict({
                "train": Dataset.from_pandas(df[df["split"] == "train"].reset_index(drop=True)),
                "test": Dataset.from_pandas(df[df["split"] == "test"].reset_index(drop=True)),
            })
        else:
            st.warning("Preprocessed dataset not found.")
            return


    model = st.session_state.trained_model
    tokenizer = st.session_state.tokenizer
    dataset = st.session_state.dataset

    st.markdown("### Generating Predictions")

    # Create test DataFrame
    test_data = pd.DataFrame(dataset["test"])

    # Initialize pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if st.session_state.get("use_gpu", False) else -1)

    # Limit number for quick evaluation
    num_samples = st.slider("Select number of test samples to evaluate", 10, 100, 20)

    inputs = ["summarize: " + text for text in test_data["text"][:num_samples]]
    predictions = summarizer(inputs, max_length=128, min_length=5, do_sample=False)
    predicted_summaries = [p["summary_text"] for p in predictions]
    reference_summaries = test_data["summary"][:num_samples].tolist()

    st.markdown("### Sample Results")
    for i in range(num_samples):
        st.write(f"**Input Text {i+1}:** {test_data['text'][i][:300]}...")
        st.success(f"**Predicted Summary:** {predicted_summaries[i]}")
        st.info(f"**Reference Summary:** {reference_summaries[i]}")
        st.divider()

    # Compute Metrics
    st.markdown("### Evaluation Metrics")

    # BLEU
    bleu_scores = [
        sentence_bleu([ref.split()], pred.split())
        for ref, pred in zip(reference_summaries, predicted_summaries)
    ]
    avg_bleu = np.mean(bleu_scores)
    st.metric("Average BLEU Score", round(avg_bleu, 4))

    # ROUGE
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, pred) for ref, pred in zip(reference_summaries, predicted_summaries)]
    avg_rouge1 = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score["rougeL"].fmeasure for score in rouge_scores])
    st.metric("Average ROUGE-1", round(avg_rouge1, 4))
    st.metric("Average ROUGE-L", round(avg_rougeL, 4))

    # Confusion Matrix (rough mapping)
    st.markdown("### Confusion Matrix (Categorical Match)")

    def coarse_match(pred, ref):
        return "Match" if pred.strip().lower() == ref.strip().lower() else "Mismatch"

    match_labels = ["Match", "Mismatch"]
    y_true = [coarse_match(ref, ref) for ref in reference_summaries]
    y_pred = [coarse_match(pred, ref) for pred, ref in zip(predicted_summaries, reference_summaries)]

    cm = confusion_matrix(y_true, y_pred, labels=match_labels)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=match_labels).plot(ax=ax, cmap='Blues')
    st.pyplot(fig)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    st.metric("Exact Match Accuracy", round(acc, 4))
