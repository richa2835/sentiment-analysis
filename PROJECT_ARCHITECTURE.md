# Project Architecture

```mermaid
flowchart TD
    A[Raw Data CSV] --> B[Data Preprocessing]
    
    subgraph NLP Pipeline
    B -->|spaCy| C[Tokenization & Cleaning]
    C --> D[Noun Phrase Extraction]
    end
    
    D --> E[Trend Detection & Extraction]
    
    subgraph Machine Learning
    F[Training Data] --> G[TF-IDF Vectorization]
    G --> H[Logistic Regression]
    H --> I[Saved Sentiment Model]
    end
    
    C --> I
    I --> J[Predicted Sentiment]
    
    E --> K[Trend Shift Analysis]
    J --> K
    
    K --> L[Processed Data CSV]
    
    L --> M[Streamlit Dashboard]
    M -->|Interactive Tabs| N[Trend Shifts, Sentiment, Real-time Analyzer]
```

### Components:
1. **Raw Data**: Captions and comments from fashion influencers.
2. **Data Preprocessing**: Cleans URLs, mentions, and symbols.
3. **NLP Pipeline**: Uses spaCy for tokenization, lemmatization, and noun phrase extraction.
4. **Trend Detection**: Identifies fashion trends from the extracted noun phrases.
5. **Sentiment Model**: A machine learning model (TF-IDF + LR) trained to predict positive, neutral, or negative sentiment.
6. **Trend Shift Analysis**: Mathematical grouping of trend mentions over time to calculate growth rates and classify trends as Emerging, Rising, Stable, or Declining.
7. **Streamlit Dashboard**: The final interactive UI to explore the data, view model performance, and analyze new text in real-time.
