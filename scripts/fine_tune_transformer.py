"""
Fine-Tuning Pipeline for Fashion Sentiment Analysis
====================================================
This script fine-tunes DistilBERT on our custom fashion influencer dataset.
It is a standalone script and does NOT affect the existing pre-trained model.

Fine-tuning means:
  - We take DistilBERT (already knows English from pre-training on millions of sentences)
  - We replace its final classification layer with our own (3 labels: neg/neu/pos)
  - We train it on our specific fashion dataset so it learns domain-specific nuances
  - We save the updated weights locally to models/my_custom_fashion_model/

Run this script once:
    python scripts/fine_tune_transformer.py

The saved model will automatically be detected by app.py.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_PATH         = "data/raw/synthetic_fashion_data.csv"
SAVE_DIR          = "models/my_custom_fashion_model"
BASE_MODEL        = "distilbert-base-uncased"
NUM_LABELS        = 3
MAX_TOKEN_LEN     = 128
EPOCHS            = 3
BATCH_SIZE        = 16
LEARNING_RATE     = 2e-5
WEIGHT_DECAY      = 0.01
REPORT_PATH       = "reports/fine_tuned_evaluation.json"

# Label mapping
LABEL_MAP    = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL  = {v: k for k, v in LABEL_MAP.items()}


# -------------------------------------------------------------------
# Custom PyTorch Dataset
# -------------------------------------------------------------------
class FashionSentimentDataset(torch.utils.data.Dataset):
    """
    Wraps tokenized HuggingFace encodings + integer labels
    into a format the Trainer API can consume.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# -------------------------------------------------------------------
# Compute metrics callback for the Trainer
# -------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# -------------------------------------------------------------------
# Main fine-tuning function
# -------------------------------------------------------------------
def fine_tune():
    # 1. Load Data
    print("=" * 60)
    print("  Step 1 — Loading Dataset")
    print("=" * 60)
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please run:  python scripts/prepare_data.py  first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text', 'sentiment_label'])
    df['label'] = df['sentiment_label'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"  Loaded {len(df)} records.")
    print(f"  Label distribution:\n{df['sentiment_label'].value_counts()}\n")

    # 2. Train / Validation Split (80 / 20)
    print("=" * 60)
    print("  Step 2 — Splitting into Train / Validation Sets")
    print("=" * 60)
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label'].tolist()
    )
    print(f"  Train size : {len(X_train)}")
    print(f"  Val size   : {len(X_val)}\n")

    # 3. Tokenisation
    print("=" * 60)
    print("  Step 3 — Tokenising Text with DistilBERT Tokenizer")
    print("=" * 60)
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_TOKEN_LEN)
    val_enc   = tokenizer(X_val,   truncation=True, padding=True, max_length=MAX_TOKEN_LEN)
    print("  Tokenisation complete.\n")

    train_dataset = FashionSentimentDataset(train_enc, y_train)
    val_dataset   = FashionSentimentDataset(val_enc,   y_val)

    # 4. Load Base Model
    print("=" * 60)
    print("  Step 4 — Loading Base Model (DistilBERT)")
    print("=" * 60)
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS
    )
    print("  Base model loaded. Final classification layer replaced with 3-class head.\n")

    # 5. Training Arguments (hyperparameters)
    print("=" * 60)
    print("  Step 5 — Configuring Training Hyperparameters")
    print("=" * 60)
    training_args = TrainingArguments(
        output_dir              = SAVE_DIR,
        num_train_epochs        = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate           = LEARNING_RATE,
        weight_decay            = WEIGHT_DECAY,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        logging_steps           = 50,
        report_to               = "none",
    )
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Weight decay  : {WEIGHT_DECAY}\n")

    # 6. Trainer
    print("=" * 60)
    print("  Step 6 — Starting Fine-Tuning  (this will take a few minutes)")
    print("=" * 60)
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
    )
    trainer.train()
    print("\n  Fine-tuning complete!\n")

    # 7. Save Model + Tokenizer
    print("=" * 60)
    print("  Step 7 — Saving Custom Model and Tokenizer")
    print("=" * 60)
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"  Model saved to: {SAVE_DIR}\n")

    # 8. Evaluation on Validation Set
    print("=" * 60)
    print("  Step 8 — Evaluating on Validation Set")
    print("=" * 60)
    predictions_output = trainer.predict(val_dataset)
    preds   = np.argmax(predictions_output.predictions, axis=-1)
    y_true  = np.array(y_val)

    acc     = accuracy_score(y_true, preds)
    report  = classification_report(y_true, preds,
                                    target_names=["negative", "neutral", "positive"],
                                    output_dict=True)
    cm      = confusion_matrix(y_true, preds)

    print(f"  Accuracy : {acc:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_true, preds, target_names=["negative", "neutral", "positive"]))

    # Save evaluation report
    os.makedirs("reports", exist_ok=True)
    report_data = {
        "model"       : "fine-tuned-distilbert",
        "accuracy"    : acc,
        "epochs"      : EPOCHS,
        "batch_size"  : BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "classification_report": report,
        "confusion_matrix"     : cm.tolist(),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"\n  Evaluation report saved to: {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("  FINE-TUNING PIPELINE COMPLETE")
    print("  Run  streamlit run app.py  and select 'Fine-Tuned Model'")
    print("  in the Text Analyzer tab to use your custom model.")
    print("=" * 60)


if __name__ == "__main__":
    fine_tune()
