import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sentiment_model import SentimentModel

def main():
    data_path = "data/raw/synthetic_fashion_data.csv"
    model_path = "models/sentiment_model/sentiment_pipeline.pkl"
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Data or model not found.")
        return
        
    df = pd.read_csv(data_path)
    X = df['text'].fillna('')
    y = df['sentiment_label']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = SentimentModel()
    model.load(model_path)
    
    acc, report, cm = model.evaluate(X_test, y_test)
    
    print("Test Accuracy:", acc)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    labels = sorted(list(set(y_test)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    print("Saved confusion matrix plot to reports/confusion_matrix.png")

if __name__ == "__main__":
    main()
