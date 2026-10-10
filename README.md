# Automated Text Mining Dashboard for Analyzing Fast-Fashion Trend Shifts

## 1. Problem Statement
Fast-fashion moves incredibly quickly, and traditional market research cannot keep up. This project aims to build an automated text-mining system that extracts emerging fashion trends from influencer captions and audience comments, tracks their shifts over time, and analyzes audience sentiment.

## 2. Objectives
- Automatically identify fashion trends from unstructured text data using NLP.
- Detect "trend shifts" (Emerging, Rising, Stable, Declining) over time.
- Analyze the audience sentiment (positive, neutral, negative) towards these trends.
- Present the findings in an interactive Streamlit dashboard.

## 3. Dataset
The dataset is a semi-synthetic collection of ~2,500 fashion influencer posts and comments, spanning multiple months. It contains captions (e.g., "Obsessed with this Y2K look!") and comments (e.g., "Not a fan of oversized clothing"). 
Note: Real data can be substituted by dropping any Instagram/TikTok CSV into `data/raw/`.

## 4. NLP Preprocessing & spaCy Usage
We use `spaCy` (`en_core_web_sm`) to clean the text:
- Tokenization & Lemmatization
- Stop-word & Punctuation Removal
- Noun Phrase Extraction (to identify candidate fashion terms)

## 5. Trend Extraction & Shift Algorithm
- **Extraction:** A hybrid approach using seed terms and noun phrase extraction.
- **Shift Detection:** Groups mentions by month. Calculates the growth rate: `(current - previous) / previous`.
  - >50% growth = Emerging
  - 10-50% = Rising
  - -10% to 10% = Stable
  - <-10% = Declining

## 6. Sentiment Model
We implemented a robust Machine Learning pipeline (`TF-IDF + LogisticRegression`) for sentiment analysis. This guarantees fast, local execution without requiring a GPU, making it highly defensible for an academic project baseline. (The architecture allows swapping this with a HuggingFace Transformer via the `SentimentModel` class).

## 7. Setup and Execution (How to run locally)

### Prerequisites
- Python 3.8+
- Windows/Linux/Mac

### Installation
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Pipeline Execution
Run the following commands in order:

1. **Generate the dataset (Synthetic for demo):**
```bash
python scripts/prepare_data.py
```

2. **Train the Sentiment Model:**
```bash
python scripts/train_model.py
```

3. **Evaluate the Model:**
```bash
python scripts/evaluate_model.py
```

4. **Process Data through NLP Pipeline:**
```bash
python scripts/process_data.py
```

5. **Run the Dashboard:**
```bash
streamlit run app.py
```

## 8. What to Demonstrate in a Viva/Review
1. **The Dataset:** Show `data/raw/synthetic_fashion_data.csv`.
2. **Preprocessing:** Show `src/preprocessing.py` and explain how spaCy extracts noun phrases.
3. **Training:** Show the output of `train_model.py` and the saved model in `models/`.
4. **Evaluation Metrics:** Show the Confusion Matrix in `reports/`.
5. **Dashboard:** Open Streamlit. Demonstrate the Real-time Text Analyzer to show live NLP processing and model prediction.

## 9. Future Scope
- Fine-tune a `DistilBERT` model on a larger corpus.
- Implement Latent Dirichlet Allocation (LDA) for fully unsupervised new topic discovery.
- Integrate live API fetching for real-time data.
