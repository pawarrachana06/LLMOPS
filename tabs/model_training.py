# tabs/model_training.py
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import DatasetDict, Dataset
import pandas as pd
import matplotlib.pyplot as plt

def run():
    st.header("Model Training")

    if "preprocessed_df" not in st.session_state:
        st.warning("Please complete data preprocessing first.")
        return

    df = st.session_state.preprocessed_df

    # Ensure 'split' column exists before proceeding
    if "split" not in df.columns:
        st.error("❌ 'split' column not found. Please ensure data was properly split in preprocessing.")
        return

    st.markdown("### Training Configuration")

    model_name = st.selectbox("Select Model", ["t5-small", "t5-base", "t5-large"], index=0)
    num_train_epochs = st.slider("Epochs", 1, 10, 3)
    batch_size = st.slider("Batch Size", 4, 32, 8, step=4)
    learning_rate = st.select_slider("Learning Rate", [5e-5, 3e-5, 2e-5, 1e-5], value=3e-5)

    if st.button("Start Training"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # Build Hugging Face DatasetDict from split DataFrame
        dataset = DatasetDict({
            "train": Dataset.from_pandas(df[df["split"] == "train"].reset_index(drop=True)),
            "test": Dataset.from_pandas(df[df["split"] == "test"].reset_index(drop=True)),
        })

        def preprocess_function(example):
            inputs = ["summarize: " + x for x in example["text"]]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

            labels = tokenizer(example["summary"], max_length=128, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Tokenization
        tokenized = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["text", "summary", "split"]
        )

        # Define training arguments
        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            report_to="none",  # Avoid WandB or other reporting unless configured
        )

        # Data Collator handles dynamic padding
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        # Trainer setup
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        st.info("Training started... This may take time depending on your system.")

        # Run training
        result = trainer.train()

        st.success("✅ Training completed!")

        # Final Metrics
        st.markdown("### Final Training Metrics")
        st.json(result.metrics)

        # Logs & Loss Curve
        st.markdown("### Training Curve")
        logs = pd.DataFrame(trainer.state.log_history)
        if "loss" in logs.columns:
            st.line_chart(logs["loss"])
        if "eval_loss" in logs.columns:
            st.line_chart(logs["eval_loss"])

        # Save to session
        st.session_state.trained_model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.active_tab = "Evaluation"
