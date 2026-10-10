import pandas as pd
from collections import Counter
import itertools

# Pre-defined fashion trends (Seed terms)
SEED_TRENDS = [
    "y2k", "streetwear", "quiet luxury", "coquette", "athleisure", 
    "oversized clothing", "denim", "cargo pants", "ballet flats", 
    "monochrome", "sustainable fashion", "vintage"
]

def extract_trends_from_phrases(noun_phrases_list):
    """
    Combines seed matching and frequent phrase extraction.
    """
    detected_trends = []
    for phrases in noun_phrases_list:
        post_trends = []
        for phrase in phrases:
            # 1. Check against seed terms
            matched = False
            for seed in SEED_TRENDS:
                if seed in phrase:
                    post_trends.append(seed)
                    matched = True
                    break
            
            # 2. If not a seed, but it's a frequent fashion term (simplified discovery)
            # In a real scenario, we might use TF-IDF or LDA here to find new topics.
            # For this pipeline, we'll keep the phrases that contain fashion keywords
            if not matched:
                fashion_keywords = ['jacket', 'pants', 'skirt', 'dress', 'shoes', 'style', 'look', 'vibe', 'aesthetic']
                if any(kw in phrase for kw in fashion_keywords):
                    post_trends.append(phrase)
                    
        # Remove duplicates
        detected_trends.append(list(set(post_trends)))
        
    return detected_trends

def get_top_trends(detected_trends_list, top_n=20):
    all_trends = list(itertools.chain.from_iterable(detected_trends_list))
    return Counter(all_trends).most_common(top_n)

if __name__ == "__main__":
    test_phrases = [["y2k aesthetic", "cool jacket"], ["vintage look", "oversized denim jacket"]]
    print(extract_trends_from_phrases(test_phrases))
