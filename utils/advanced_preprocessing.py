import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import torch
from textstat import flesch_reading_ease, smog_index
from textblob import TextBlob

def extract_advanced_metadata_features(item, n_topics=5):
    """
    Extract more sophisticated metadata features with enhanced fake news indicators
    """
    features = []
    
    # Basic features with better normalization
    tweet_count = item.get('tweet_count', 0)
    features.append(min(tweet_count / 50, 2.0))  # Allow values > 1 for viral content
    
    text_length = item.get('text_length', 0)
    features.append(min(text_length / 800, 1.5))  # Longer texts might be more detailed
    
    # Enhanced sentiment features
    text = item.get('clean_text', '')
    
    # Advanced sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'worst', 'disgusting']
    sensational_words = ['shocking', 'breaking', 'urgent', 'secret', 'exposed', 'unbelievable']
    
    positive_count = sum(1 for word in positive_words if word in text.lower())
    negative_count = sum(1 for word in negative_words if word in text.lower())
    sensational_count = sum(1 for word in sensational_words if word in text.lower())
    
    features.append(min(positive_count / 3, 1.0))
    features.append(min(negative_count / 3, 1.0))
    features.append(min(sensational_count / 2, 1.5))  # Sensationalism indicator
    
    # Enhanced punctuation analysis
    features.append(min(text.count('?') / 3, 1.0))
    features.append(min(text.count('!') / 3, 1.5))  # Multiple exclamations suspicious
    
    # URL and social media indicators
    features.append(min(len(re.findall(r'http\S+', text)) / 3, 1.0))
    features.append(min(len(re.findall(r'@\w+', text)) / 5, 1.0))
    features.append(min(len(re.findall(r'#\w+', text)) / 5, 1.0))
    
    # Readability features
    try:
        readability = flesch_reading_ease(text)
        features.append(max(0, min(readability / 100, 1.0)))
    except:
        features.append(0.5)
    
    # Text complexity
    try:
        smog = smog_index(text)
        features.append(min(smog / 20, 1.0))
    except:
        features.append(0.3)
    
    # Emotional intensity (simple version)
    emotional_indicators = text.count('!') + text.count('?') + sensational_count
    features.append(min(emotional_indicators / 10, 1.0))
    
    return np.array(features, dtype=np.float32)

def extract_topic_features(texts, n_topics=5):
    """
    Extract topic features using LDA with enhanced preprocessing
    """
    # Enhanced text cleaning
    cleaned_texts = []
    for text in texts:
        if not text or len(text.strip()) == 0:
            cleaned_texts.append("")
            continue
        
        # Remove URLs, mentions, and special characters
        clean_text = re.sub(r'http\S+', '', text)
        clean_text = re.sub(r'@\w+', '', clean_text)
        clean_text = re.sub(r'#\w+', '', clean_text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()
        
        cleaned_texts.append(clean_text)
    
    if len(cleaned_texts) < 2:
        # Return default features if not enough texts
        return np.zeros((len(texts), n_topics)), None
    
    try:
        tfidf = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = tfidf.fit_transform(cleaned_texts)
        
        lda = LatentDirichletAllocation(
            n_components=n_topics, 
            random_state=42,
            learning_method='online'
        )
        topic_features = lda.fit_transform(tfidf_matrix)
        
        return topic_features, lda
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        return np.zeros((len(texts), n_topics)), None

def extract_linguistic_patterns(text):
    """
    Extract linguistic patterns indicative of fake news
    """
    patterns = {}
    
    text_lower = text.lower()
    
    # Clickbait patterns
    clickbait_phrases = [
        'you won\'t believe', 'shocking', 'what happened next', 'going viral',
        'this will blow your mind', 'secret they don\'t want you to know'
    ]
    patterns['clickbait_score'] = sum(1 for phrase in clickbait_phrases if phrase in text_lower) / len(clickbait_phrases)
    
    # Authority appeals
    authority_phrases = [
        'experts say', 'studies show', 'scientists prove', 'doctors recommend',
        'government reports', 'official sources'
    ]
    patterns['authority_appeals'] = sum(1 for phrase in authority_phrases if phrase in text_lower) / len(authority_phrases)
    
    # Emotional manipulation
    emotional_words = [
        'outrageous', 'disgusting', 'horrible', 'terrible', 'devastating',
        'heartbreaking', 'disgraceful', 'appalling'
    ]
    patterns['emotional_intensity'] = sum(1 for word in emotional_words if word in text_lower) / len(emotional_words)
    
    # Certainty indicators
    certainty_words = [
        'certainly', 'definitely', 'undoubtedly', 'clearly', 'obviously',
        'absolutely', 'unquestionably', 'proven'
    ]
    patterns['certainty_level'] = sum(1 for word in certainty_words if word in text_lower) / len(certainty_words)
    
    # Conspiracy indicators
    conspiracy_terms = [
        'cover-up', 'deep state', 'mainstream media', 'they are lying',
        'the truth they\'re hiding', 'wake up'
    ]
    patterns['conspiracy_indicators'] = sum(1 for term in conspiracy_terms if term in text_lower) / len(conspiracy_terms)
    
    return patterns

def create_cross_modal_features(text_features, image_features, metadata_features):
    """
    Create cross-modal interaction features
    """
    if text_features is None or image_features is None:
        return np.zeros(10)  # Default features
    
    # Convert to numpy if they are tensors
    if torch.is_tensor(text_features):
        text_features = text_features.cpu().numpy()
    if torch.is_tensor(image_features):
        image_features = image_features.cpu().numpy()
    if torch.is_tensor(metadata_features):
        metadata_features = metadata_features.cpu().numpy()
    
    # Basic statistics for cross-modal alignment
    cross_modal_features = []
    
    # Feature magnitude ratios
    text_norm = np.linalg.norm(text_features) if len(text_features) > 0 else 0.1
    image_norm = np.linalg.norm(image_features) if len(image_features) > 0 else 0.1
    
    cross_modal_features.append(text_norm / (image_norm + 1e-8))
    cross_modal_features.append(image_norm / (text_norm + 1e-8))
    
    # Feature correlation (simplified)
    if len(text_features) == len(image_features):
        correlation = np.corrcoef(text_features[:min(10, len(text_features))], 
                                 image_features[:min(10, len(image_features))])[0,1]
        cross_modal_features.append(correlation if not np.isnan(correlation) else 0)
    else:
        cross_modal_features.append(0)
    
    # Metadata influence
    if metadata_features is not None and len(metadata_features) > 0:
        cross_modal_features.extend(metadata_features[:5])  # Use first 5 metadata features
    else:
        cross_modal_features.extend([0] * 5)
    
    # Pad or truncate to 10 features
    if len(cross_modal_features) < 10:
        cross_modal_features.extend([0] * (10 - len(cross_modal_features)))
    else:
        cross_modal_features = cross_modal_features[:10]
    
    return np.array(cross_modal_features, dtype=np.float32)