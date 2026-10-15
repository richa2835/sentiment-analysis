"""
Fine-Tuned Model Loader
========================
Loads our locally fine-tuned DistilBERT from models/my_custom_fashion_model/.
This is completely separate from the pre-trained HuggingFace model in sentiment_model.py.
"""

import os
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

FINE_TUNED_DIR = "models/my_custom_fashion_model"
ID_TO_LABEL    = {0: "negative", 1: "neutral", 2: "positive"}


def is_fine_tuned_model_available():
    """Check if the locally fine-tuned model has been saved (config + weights must exist)."""
    has_config = os.path.exists(os.path.join(FINE_TUNED_DIR, "config.json"))
    has_weights = (
        os.path.exists(os.path.join(FINE_TUNED_DIR, "model.safetensors")) or
        os.path.exists(os.path.join(FINE_TUNED_DIR, "pytorch_model.bin"))
    )
    return has_config and has_weights


class FineTunedSentimentModel:
    """
    Loads our custom fine-tuned DistilBERT model.
    Used in the Streamlit Text Analyzer tab as the 'Our Trained Model' option.
    """

    def __init__(self):
        if not is_fine_tuned_model_available():
            raise FileNotFoundError(
                f"Fine-tuned model not found at '{FINE_TUNED_DIR}'.\n"
                "Please run:  python scripts/fine_tune_transformer.py"
            )

        print(f"Loading fine-tuned model from {FINE_TUNED_DIR} ...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(FINE_TUNED_DIR)
        self.model     = DistilBertForSequenceClassification.from_pretrained(FINE_TUNED_DIR)
        self.model.eval()
        print("Fine-tuned model loaded successfully.")

    def predict_single(self, text: str) -> dict:
        """
        Predicts sentiment for a single string.
        Returns a dict with 'label' and 'confidence'.
        """
        if not isinstance(text, str) or text.strip() == "":
            return {"label": "neutral", "confidence": 0.0}

        inputs = self.tokenizer(
            text[:512],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits  = outputs.logits
            probs   = torch.softmax(logits, dim=-1).squeeze().numpy()

        predicted_id    = int(np.argmax(probs))
        confidence      = float(probs[predicted_id])
        label           = ID_TO_LABEL[predicted_id]

        return {
            "label"      : label,
            "confidence" : round(confidence * 100, 2),
            "scores"     : {
                "negative": round(float(probs[0]) * 100, 2),
                "neutral" : round(float(probs[1]) * 100, 2),
                "positive": round(float(probs[2]) * 100, 2),
            }
        }

    def predict(self, texts: list) -> list:
        """Batch predict — returns a list of label strings."""
        return [self.predict_single(t)["label"] for t in texts]
