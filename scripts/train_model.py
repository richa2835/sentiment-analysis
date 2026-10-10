import pandas as pd
import json
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sentiment_model import SentimentModel

def main():
    data_path = "data/raw/synthetic_fashion_data.csv"
    if not os.path.exists(data_path):
        print("Data not found. Run prepare_data.py first.")
        return
        
    df = pd.parse_dates=False
    df = pd.read_csv(data_path)
    
    # We will use the raw text for TF-IDF since it does its own tokenization,
    # but in a full pipeline we might use our cleaned text.
    
    X = df['text'].fillna('')
    y = df['sentiment_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = SentimentModel()
    model.train(X_train, y_train)
    
    # Save model
    model_path = "models/sentiment_model/sentiment_pipeline.pkl"
    model.save(model_path)
    
    # Evaluate
    acc, report, cm = model.evaluate(X_test, y_test)
    
    print(f"Accuracy: {acc:.4f}")
    
    # Save evaluation report
    report_data = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_report.json", "w") as f:
        json.dump(report_data, f, indent=4)
        
    print("Evaluation report saved to reports/evaluation_report.json")

if __name__ == "__main__":
    main()
