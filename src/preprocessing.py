import spacy
import re
import pandas as pd

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove special characters but keep hashtags for now
    text = re.sub(r'[^\w\s#]', '', text)
    return text.strip()

def preprocess_text(text):
    text = clean_text(text)
    doc = nlp(text)
    
    tokens = []
    for token in doc:
        # Keep hashtags, alphanumeric, remove stop words and punctuation
        if (token.is_alpha or token.text.startswith('#')) and not token.is_stop and not token.is_punct:
            # Lemmatize
            tokens.append(token.lemma_.lower())
            
    return " ".join(tokens)

def extract_noun_phrases(text):
    doc = nlp(clean_text(text))
    return [chunk.text.lower() for chunk in doc.noun_chunks if not chunk.root.is_stop]

def preprocess_dataframe(df, text_column="text"):
    print("Preprocessing text...")
    df['clean_text'] = df[text_column].apply(preprocess_text)
    print("Extracting noun phrases...")
    df['noun_phrases'] = df[text_column].apply(extract_noun_phrases)
    return df

if __name__ == "__main__":
    # Test
    sample = "Obsessed with these oversized denim jackets! @fashion_guru #streetwear https://example.com"
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))
    print("Preprocessed:", preprocess_text(sample))
    print("Noun Phrases:", extract_noun_phrases(sample))
