import re
import numpy as np
import pandas as pd
import torch

def clean_text(text):
    """
    Clean text data - same as provided code
    """
    if not text or pd.isna(text):
        return ""
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_extract_metadata_features(item):
    try:
        return extract_metadata_features(item)
    except Exception as e:
        print(f"Error extracting advanced features: {e}")
        return np.zeros(6, dtype=np.float32)


def preprocess_text(text):
    """Basic text preprocessing placeholder."""
    import re
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'[^a-z0-9\s]', '', text)              # remove special chars
    return text.strip()


def extract_metadata_features(item):
    """
    Extract robust metadata features from a data item (dict or pandas Series).
    Always returns 6 normalized features.
    """
    # Ensure we can access keys safely
    if isinstance(item, pd.Series):
        item = item.to_dict()

    features = []

    # Tweet count (normalized)
    tweet_count = float(item.get('tweet_count', 0) or 0)
    features.append(min(tweet_count / 100, 1.0))

    # Text length (normalized)
    text = str(item.get('text', '') or '')
    text_length = len(text)
    features.append(min(text_length / 500, 1.0))

    # Word count
    word_count = len(text.split())
    features.append(min(word_count / 100, 1.0))

    # Question marks count
    features.append(min(text.count('?') / 5, 1.0))

    # Exclamation marks count
    features.append(min(text.count('!') / 5, 1.0))

    # URL count
    url_count = len(re.findall(r'http\S+', text))
    features.append(min(url_count / 5, 1.0))

    return np.array(features, dtype=np.float32)

# ADD THESE MISSING FUNCTIONS:
def extract_advanced_metadata_features(row):
    """
    Safe extraction of advanced metadata features from a row.
    Handles both Series and dict types properly.
    """
    try:
        # Convert to dict if it's a Series
        if hasattr(row, 'to_dict'):
            item = row.to_dict()
        else:
            item = row
            
        # Extract features safely without boolean operations on Series
        features = []
        
        # Tweet count (normalized)
        tweet_count = item.get('tweet_count', 0)
        if pd.isna(tweet_count) or tweet_count is None:
            tweet_count = 0
        features.append(min(float(tweet_count) / 100, 1.0))

        # Text length (normalized)
        text = item.get('text', '')
        if pd.isna(text) or text is None:
            text = ''
        text = str(text)
        text_length = len(text)
        features.append(min(text_length / 500, 1.0))

        # Word count
        word_count = len(text.split())
        features.append(min(word_count / 100, 1.0))

        # Question marks count
        features.append(min(text.count('?') / 5, 1.0))

        # Exclamation marks count
        features.append(min(text.count('!') / 5, 1.0))

        # URL count
        url_count = len(re.findall(r'http\S+', text))
        features.append(min(url_count / 5, 1.0))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting advanced features: {e}")
        return np.zeros(6, dtype=np.float32)

def extract_linguistic_patterns(text):
    """
    Extract linguistic patterns from text.
    Returns a fixed-size feature vector.
    """
    try:
        text = str(text)
        features = []
        
        # Text length normalized
        features.append(min(len(text) / 500, 1.0))
        
        # Word count normalized
        words = text.split()
        features.append(min(len(words) / 100, 1.0))
        
        # Sentence count (approximate)
        sentences = re.split(r'[.!?]+', text)
        features.append(min(len([s for s in sentences if len(s.strip()) > 0]) / 10, 1.0))
        
        # Capitalization ratio
        if len(text) > 0:
            cap_ratio = sum(1 for c in text if c.isupper()) / len(text)
            features.append(min(cap_ratio * 10, 1.0))
        else:
            features.append(0.0)
            
        # Question marks density
        features.append(min(text.count('?') / 5, 1.0))
        
        # Exclamation marks density
        features.append(min(text.count('!') / 5, 1.0))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting linguistic patterns: {e}")
        return np.zeros(6, dtype=np.float32)