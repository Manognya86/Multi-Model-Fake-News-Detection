from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer
import numpy as np
import time
from datetime import datetime, timedelta

import logging
import os
import re
import requests
import json
import sys
import threading
from collections import deque

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import config
    logger.info("✅ Configuration loaded successfully")
except ImportError as e:
    logger.warning(f"❌ Could not import config: {e}")
    # Fallback config
    class FallbackConfig:
        NEWS_API_KEY = '0742802fb6b843bc87c2b13b9e1f1be0'
        GNEWS_API_KEY = '510006514cda39e2c0b0644bec34fe85'
    config = FallbackConfig()

# Global variables for components
model = None
tokenizer = None
current_model_name = "Enhanced Fake News Detector"
analysis_history = deque(maxlen=1000)  # Use deque for efficient rolling window
real_time_news = []
system_metrics = {
    'total_requests': 0,
    'fake_count': 0,
    'real_count': 0,
    'avg_confidence': 0.0,
    'last_update': datetime.now()
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🖥️ Using device: {device}")

# Import your model architecture
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.unified_multimodal import UnifiedMultimodalModel
    from models.multimodal_bert_model import create_model_from_registry
    logger.info("✅ Successfully imported model architectures")
except ImportError as e:
    logger.warning(f"❌ Could not import model architectures: {e}")

class SimpleFakeNewsModel(nn.Module):
    """Simple model for fake news detection"""
    def __init__(self, vocab_size=50000, embedding_dim=100, hidden_dim=64, output_dim=2):
        super(SimpleFakeNewsModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use the last hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.classifier(hidden)
        return output

def load_trained_model():
    """Load your actual trained multimodal model"""
    global model, tokenizer, current_model_name
    
    try:
        # Try multiple possible model paths
        model_paths = [
            r"D:\UMich\2 Sem\Deep Learning\Term Project\multimodal-fake-news-detection\models\saved_models\multimodal_final.pth",
            r"D:\UMich\2 Sem\Deep Learning\Term Project\multimodal-fake-news-detection\models\saved\best_model.pth",
            r"D:\UMich\2 Sem\Deep Learning\Term Project\multimodal-fake-news-detection\models\saved_models\best_model.pth",
            r"./saved_models/best_unified_multimodal_model.pth",
            r"./saved_models/unified_multimodal_model.pth"
        ]
        
        successful_load = False
        loaded_path = ""
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    logger.info(f"🔄 Attempting to load model from: {model_path}")
                    
                    # Try to load UnifiedMultimodalModel first
                    model = UnifiedMultimodalModel(
                        text_model_name='distilroberta-base',
                        metadata_dim=6,
                        image_dim=768,
                        graph_dim=256,
                        n_classes=2,
                        dropout_prob=0.3
                    )
                    
                    # Load trained weights
                    state_dict = torch.load(model_path, map_location=device)
                    
                    # Handle state dict with 'module.' prefix (if trained with DataParallel)
                    if all(key.startswith('module.') for key in state_dict.keys()):
                        state_dict = {k[7:]: v for k, v in state_dict.items()}
                    
                    model.load_state_dict(state_dict)
                    model.to(device)
                    model.eval()
                    
                    # Load tokenizer
                    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
                    
                    successful_load = True
                    loaded_path = model_path
                    current_model_name = "Trained Multimodal Fake News Detector"
                    logger.info(f"✅ Trained multimodal model loaded successfully from: {model_path}")
                    break
                    
                except Exception as e:
                    logger.warning(f"❌ Failed to load from {model_path}: {e}")
                    continue
        
        if not successful_load:
            # Try to create model from registry as fallback
            try:
                logger.info("🔄 Trying to create model from registry...")
                model = create_model_from_registry(
                    model_type='enhanced',
                    text_model_name='distilroberta-base',
                    metadata_dim=6,
                    n_classes=2
                )
                model.to(device)
                model.eval()
                tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
                current_model_name = "Enhanced Model (Registry)"
                logger.info("✅ Model created from registry successfully")
                successful_load = True
            except Exception as e:
                logger.warning(f"❌ Failed to create model from registry: {e}")
        
        return successful_load
        
    except Exception as e:
        logger.error(f"❌ Failed to load trained model: {e}")
        return False

def predict_with_trained_model(text, metadata_features):
    """Make prediction using your trained multimodal model"""
    global model, tokenizer
    
    try:
        # Preprocess text
        cleaned_text = clean_text(text)
        
        # Ensure metadata features are the right length
        if len(metadata_features) < 6:
            metadata_features = metadata_features + [0.0] * (6 - len(metadata_features))
        elif len(metadata_features) > 6:
            metadata_features = metadata_features[:6]
        
        # Tokenize
        encoding = tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        metadata_tensor = torch.tensor([metadata_features], dtype=torch.float).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata=metadata_tensor,
                image_data=None,
                graph_data=None
            )
            
            # Get probabilities
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            elif isinstance(outputs, dict) and 'output' in outputs:
                logits = outputs['output']
            else:
                logits = outputs
                
            probabilities = torch.softmax(logits, dim=1)
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
        
        # Format results
        is_fake = fake_prob > real_prob
        confidence = max(fake_prob, real_prob)
        
        # Generate explanation based on probabilities
        if fake_prob > 0.7:
            explanation = "High probability of fake news based on multimodal analysis"
        elif real_prob > 0.7:
            explanation = "High probability of real news based on multimodal analysis"
        else:
            explanation = "Uncertain classification - requires further verification"
        
        return {
            'is_fake': is_fake,
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'confidence_level': 'HIGH' if confidence > 0.8 else 'MEDIUM' if confidence > 0.6 else 'LOW',
            'uncertainty': 1 - confidence,
            'explanation': explanation,
            'feature_importance': {
                'text_analysis': 0.6,
                'metadata_analysis': 0.3,
                'model_confidence': confidence,
                'model_used': 'Trained Multimodal'
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error in model prediction: {e}")
        raise e

def clean_text(text):
    """Enhanced text cleaning function"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    
    return text.strip()

def extract_metadata_features(data):
    """Extract comprehensive metadata features from input data"""
    text = data.get('text', '')
    tweet_count = data.get('tweet_count', 0)
    source = data.get('source', '')
    
    # Text-based features
    text_length = len(text) if text else 0
    word_count = len(text.split()) if text else 0
    avg_word_length = text_length / max(word_count, 1)
    
    # Sentiment indicators
    has_exclamation = 1.0 if '!' in text else 0.0
    has_question = 1.0 if '?' in text else 0.0
    has_uppercase = 1.0 if any(c.isupper() for c in text[:100]) else 0.0
    
    # Content indicators
    has_url = 1.0 if 'http' in text.lower() else 0.0
    has_numbers = 1.0 if any(c.isdigit() for c in text) else 0.0
    
    # Source credibility (simple heuristic)
    credible_sources = ['bbc', 'reuters', 'associated press', 'ap news', 'cnn', 'the new york times']
    source_credibility = 1.0 if any(source.lower() for source in credible_sources if source in source.lower()) else 0.0
    
    # Normalize features
    normalized_length = min(text_length / 1000, 1.0)
    normalized_tweets = min(tweet_count / 1000, 1.0)
    normalized_word_length = min(avg_word_length / 20, 1.0)
    
    return [
        normalized_length,
        normalized_tweets,
        normalized_word_length,
        has_exclamation,
        has_url,
        source_credibility,
        has_question,
        has_uppercase,
        has_numbers
    ]

def get_heuristic_prediction(text, metadata):
    """Advanced heuristic-based prediction"""
    text_lower = text.lower()
    
    # Fake news indicators (weighted)
    fake_indicators = {
        'breaking news': 2.0, 'shocking': 1.5, 'you won\'t believe': 2.5, 'viral': 1.2,
        'must watch': 1.8, 'miracle cure': 2.8, 'hidden truth': 2.2, 'they don\'t want you to know': 2.7,
        'mainstream media hiding': 2.5, 'secret': 1.8, 'conspiracy': 2.4, 'wake up': 1.9,
        'elite': 1.7, 'suppressed': 2.1
    }
    
    # Real news indicators (weighted)
    real_indicators = {
        'according to sources': 1.8, 'official statement': 2.2, 'research shows': 2.0,
        'study found': 1.9, 'experts say': 1.7, 'peer-reviewed': 2.5, 'clinical trial': 2.3,
        'scientific evidence': 2.4, 'university study': 1.8, 'journal publication': 2.1,
        'according to data': 1.6
    }
    
    # Calculate scores
    fake_score = sum(weight for indicator, weight in fake_indicators.items() if indicator in text_lower)
    real_score = sum(weight for indicator, weight in real_indicators.items() if indicator in text_lower)
    
    # Text characteristics
    text_length = len(text)
    word_count = len(text.split())
    
    # Length factor (longer texts are generally more credible)
    length_factor = min(text_length / 800, 1.2)
    
    # Complexity factor
    avg_sentence_length = word_count / max(text.count('.') + text.count('!') + text.count('?'), 1)
    complexity_factor = min(avg_sentence_length / 25, 1.2)
    
    # Source credibility from metadata
    source_credibility = metadata[5] if len(metadata) > 5 else 0.5
    
    # Calculate final probability
    total_score = max(fake_score + real_score, 0.1)
    base_fake_probability = fake_score / total_score
    
    # Apply factors
    adjusted_probability = base_fake_probability * (1 - length_factor * 0.1) * (1 - complexity_factor * 0.1) * (1 - source_credibility * 0.3)
    
    # Ensure probability is within bounds
    fake_probability = min(max(adjusted_probability, 0.05), 0.95)
    
    is_fake = fake_probability > 0.5
    confidence = max(fake_probability, 1 - fake_probability)
    
    # Determine confidence level
    if confidence > 0.8:
        confidence_level = 'HIGH'
    elif confidence > 0.6:
        confidence_level = 'MEDIUM'
    else:
        confidence_level = 'LOW'
    
    # Generate explanation
    if fake_score > real_score:
        explanation = "Content contains sensational language and common fake news patterns."
    else:
        explanation = "Content follows standard reporting patterns with credible language."
    
    return {
        'is_fake': is_fake,
        'confidence': confidence,
        'fake_probability': fake_probability,
        'real_probability': 1 - fake_probability,
        'confidence_level': confidence_level,
        'uncertainty': 1 - confidence,
        'explanation': explanation,
        'feature_importance': {
            'sensational_language': min(fake_score / 10, 1.0),
            'credible_indicators': min(real_score / 10, 1.0),
            'text_length': length_factor - 1.0,
            'source_credibility': source_credibility
        }
    }

def predict_with_model(text, metadata_features):
    """Predict using available model or fallback to heuristics"""
    global model
    
    if model is not None:
        try:
            return predict_with_trained_model(text, metadata_features)
        except Exception as e:
            logger.warning(f"Model prediction failed, using heuristics: {e}")
            return get_heuristic_prediction(text, metadata_features)
    else:
        return get_heuristic_prediction(text, metadata_features)

def analyze_social_context(metadata):
    """Analyze social media context from metadata"""
    tweet_count = metadata[1] if len(metadata) > 1 else 0
    
    virality_score = min(tweet_count * 2, 1.0)
    
    return {
        'virality_score': virality_score,
        'bot_likelihood': min(tweet_count * 3, 0.7),
        'coordination_score': 0.1,
        'user_credibility': 0.7 - (virality_score * 0.3)
    }

def get_real_news_from_apis():
    """Fetch real news from NewsAPI and GNews APIs"""
    news_items = []
    
    try:
        # Try NewsAPI first
        newsapi_url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=10&apiKey={config.NEWS_API_KEY}"
        logger.info(f"📡 Fetching news from NewsAPI...")
        response = requests.get(newsapi_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])[:5]
            logger.info(f"📰 Received {len(articles)} articles from NewsAPI")
            
            for article in articles:
                if article.get('title') and article.get('title') != '[Removed]':
                    content = article.get('description') or article.get('title') or 'No content available'
                    news_items.append({
                        'title': article.get('title', 'No title'),
                        'content': content,
                        'source': article.get('source', {}).get('name', 'Unknown Source'),
                        'published_at': article.get('publishedAt', datetime.now().isoformat()),
                        'url': article.get('url', '#'),
                        'category': 'General',
                        'author': article.get('author', 'Unknown Author'),
                        'is_real_news': True
                    })
            logger.info(f"✅ Successfully processed {len(news_items)} news from NewsAPI")
        else:
            logger.warning(f"❌ NewsAPI request failed with status: {response.status_code}")
    except Exception as e:
        logger.warning(f"❌ NewsAPI fetch failed: {e}")
    
    try:
        # Try GNews API as backup
        if len(news_items) < 3:
            gnews_url = f"https://gnews.io/api/v4/top-headlines?token={config.GNEWS_API_KEY}&lang=en&max=5"
            logger.info(f"📡 Fetching news from GNews API...")
            response = requests.get(gnews_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])[:5]
                logger.info(f"📰 Received {len(articles)} articles from GNews")
                
                for article in articles:
                    if article.get('title'):
                        content = article.get('description') or article.get('title') or 'No content available'
                        news_items.append({
                            'title': article.get('title', 'No title'),
                            'content': content,
                            'source': article.get('source', {}).get('name', 'Unknown Source'),
                            'published_at': article.get('publishedAt', datetime.now().isoformat()),
                            'url': article.get('url', '#'),
                            'category': 'General',
                            'author': article.get('source', {}).get('name', 'Unknown Author'),
                            'is_real_news': True
                        })
                logger.info(f"✅ Total news items from APIs: {len(news_items)}")
            else:
                logger.warning(f"❌ GNews API request failed with status: {response.status_code}")
    except Exception as e:
        logger.warning(f"❌ GNews API fetch failed: {e}")
    
    # Fallback to sample data if APIs fail
    if not news_items:
        logger.info("🔄 Using sample news data as fallback")
        return get_sample_news_fallback()
    
    return news_items

def get_sample_news_fallback():
    """Original sample news function as fallback"""
    return [
        {
            'title': 'Breaking: Major Scientific Discovery Announced',
            'content': 'Researchers have announced a groundbreaking discovery in renewable energy technology that could transform the industry.',
            'source': 'Science Daily',
            'published_at': datetime.now().isoformat(),
            'url': 'https://example.com/news/1',
            'category': 'Science',
            'author': 'Dr. Sarah Chen',
            'is_real_news': True
        },
        {
            'title': 'Economic Growth Exceeds Expectations',
            'content': 'Recent economic data shows growth exceeding expert predictions across multiple sectors.',
            'source': 'Financial Times',
            'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
            'url': 'https://example.com/news/2',
            'category': 'Economy',
            'author': 'Economics Correspondent',
            'is_real_news': True
        }
    ]

def generate_real_time_news():
    """Generate real-time news using APIs when available"""
    global real_time_news
    
    try:
        # Try to get real news from APIs
        news_items = get_real_news_from_apis()
        
        # Add some fake news samples for demonstration
        fake_samples = [
            {
                'title': 'Secret Government Alien Program Revealed',
                'content': 'Whistleblower claims government has been hiding alien technology for decades!',
                'source': 'Conspiracy News Network',
                'published_at': datetime.now().isoformat(),
                'url': 'https://example.com/fake/1',
                'category': 'Conspiracy',
                'author': 'Anonymous Insider',
                'is_sample_fake': True
            },
            {
                'title': 'Miracle Weight Loss Pill Discovered',
                'content': 'Doctors hate this one simple trick! Lose 30 pounds in one week without diet!',
                'source': 'Health Secrets Daily',
                'published_at': (datetime.now() - timedelta(minutes=30)).isoformat(),
                'url': 'https://example.com/fake/2',
                'category': 'Health',
                'author': 'Wellness Expert',
                'is_sample_fake': True
            }
        ]
        
        all_news = news_items + fake_samples
        logger.info(f"🔄 Analyzing {len(all_news)} news items for fake news detection...")
        
        # Analyze each news item
        for i, news in enumerate(all_news):
            try:
                result = predict_with_model(news['content'], [])
                news['prediction'] = {
                    'label': 'FAKE' if result.get('is_fake', False) else 'REAL',
                    'confidence': round(result.get('confidence', 0.5), 3),
                    'fake_probability': round(result.get('fake_probability', 0.5), 3),
                    'real_probability': round(result.get('real_probability', 0.5), 3),
                    'confidence_level': result.get('confidence_level', 'MEDIUM')
                }
                
                # Calculate time ago
                try:
                    published_str = news['published_at'].replace('Z', '+00:00')
                    published_time = datetime.fromisoformat(published_str)
                    time_diff = datetime.now().astimezone() - published_time
                    
                    if time_diff.total_seconds() < 60:
                        news['time_ago'] = 'Just now'
                    elif time_diff.total_seconds() < 3600:
                        minutes = int(time_diff.total_seconds() / 60)
                        news['time_ago'] = f'{minutes} min ago'
                    else:
                        hours = int(time_diff.total_seconds() / 3600)
                        news['time_ago'] = f'{hours} hours ago'
                except:
                    news['time_ago'] = 'Recently'
                    
                logger.info(f"📊 News {i+1}: '{news['title'][:50]}...' - Prediction: {news['prediction']['label']} (Confidence: {news['prediction']['confidence']})")
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing news item {i+1}: {e}")
                news['prediction'] = {
                    'label': 'UNKNOWN',
                    'confidence': 0.0,
                    'fake_probability': 0.5,
                    'real_probability': 0.5,
                    'confidence_level': 'LOW'
                }
                news['time_ago'] = 'Unknown'
        
        real_time_news = all_news
        logger.info(f"✅ Real-time news updated with {len(all_news)} items at {datetime.now().strftime('%H:%M:%S')}")
        return all_news
        
    except Exception as e:
        logger.error(f"❌ Error generating real-time news: {e}")
        return get_sample_news_fallback()

def update_system_metrics():
    """Update system metrics based on analysis history"""
    global system_metrics
    
    if analysis_history:
        system_metrics['total_requests'] = len(analysis_history)
        system_metrics['fake_count'] = sum(1 for analysis in analysis_history if analysis['prediction'] == 'FAKE')
        system_metrics['real_count'] = sum(1 for analysis in analysis_history if analysis['prediction'] == 'REAL')
        system_metrics['avg_confidence'] = round(sum(analysis['confidence'] for analysis in analysis_history) / len(analysis_history), 3)
        system_metrics['last_update'] = datetime.now()
        
        logger.info(f"📈 Metrics Updated - Total: {system_metrics['total_requests']}, "
                   f"Fake: {system_metrics['fake_count']}, Real: {system_metrics['real_count']}, "
                   f"Avg Confidence: {system_metrics['avg_confidence']}")

def initialize_components():
    """Initialize all components including trained model"""
    global tokenizer, model, current_model_name
    
    logger.info("🔄 Initializing application components...")
    
    # Try to load trained model first
    if load_trained_model():
        logger.info("🎯 Using trained multimodal model for predictions")
    else:
        # Fallback to tokenizer only
        try:
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            logger.info("✅ Tokenizer initialized (using heuristic fallback)")
            current_model_name = "Enhanced Fake News Detector (Heuristics)"
        except Exception as e:
            logger.warning(f"Tokenizer initialization failed: {e}")
            tokenizer = None

# Background news refresh function
def background_news_refresh():
    """Background task to refresh news periodically"""
    while True:
        try:
            time.sleep(300)  # 5 minutes
            logger.info("🔄 Background news refresh triggered")
            generate_real_time_news()
        except Exception as e:
            logger.error(f"❌ Background news refresh failed: {e}")

# Start background refresh thread
news_refresh_thread = threading.Thread(target=background_news_refresh, daemon=True)
news_refresh_thread.start()

# Initialize components when module loads
initialize_components()
generate_real_time_news()  # Initialize real-time news

@app.route('/')
def home():
    """Main application page"""
    logger.info("🏠 Home page accessed")
    return render_template('index.html', 
                         real_time_news=real_time_news,
                         model_name=current_model_name,
                         model_loaded=model is not None,
                         system_metrics=system_metrics)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'components': {
            'model_loaded': model is not None,
            'model_name': current_model_name,
            'tokenizer_ready': tokenizer is not None,
            'analysis_count': len(analysis_history),
            'using_trained_model': model is not None,
            'real_time_news_count': len(real_time_news),
            'system_metrics': system_metrics
        },
        'system': {
            'device': str(device),
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'uptime_seconds': time.time()
        }
    }
    logger.info("❤️ Health check performed - System healthy")
    return jsonify(health_status)

@app.route('/test/model')
def test_model():
    """Test if trained model is working"""
    logger.info("🧪 Model test endpoint accessed")
    
    if model is None:
        logger.warning("❌ Model test failed - No model loaded")
        return jsonify({'status': 'no_model', 'message': 'Trained model not loaded'})
    
    try:
        # Test prediction
        test_text = "This is a test news article about scientific research and verified information."
        metadata = [0.5, 0.3, 0.2, 0.1, 0.4, 0.6]
        result = predict_with_trained_model(test_text, metadata)
        
        logger.info(f"✅ Model test successful - Prediction: {result}")
        
        return jsonify({
            'status': 'success',
            'model_loaded': True,
            'test_prediction': result,
            'model_name': current_model_name,
            'device': str(device)
        })
    
    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/analyze', methods=['POST'])
def enhanced_analyze():
    """Enhanced analysis endpoint with comprehensive features"""
    start_time = time.time()
    logger.info("🔍 Analysis request received")
    
    try:
        data = request.get_json()
        if not data:
            logger.warning("❌ Analysis failed - No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '')
        metadata = data.get('metadata', {})
        
        if not text or not isinstance(text, str):
            logger.warning("❌ Analysis failed - Invalid text content")
            return jsonify({'error': 'Valid text content is required'}), 400
        
        if len(text.strip()) < 10:
            logger.warning("❌ Analysis failed - Text too short")
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400
        
        if len(text) > 10000:
            logger.warning("❌ Analysis failed - Text too long")
            return jsonify({'error': 'Text too long (maximum 10,000 characters)'}), 400
        
        logger.info(f"📝 Analyzing text: {text[:100]}...")
        
        # Enhanced text preprocessing
        cleaned_text = clean_text(text)
        
        # Extract metadata features
        try:
            metadata_features = extract_metadata_features({
                'text': cleaned_text,
                'tweet_count': metadata.get('tweet_count', 0),
                'source': metadata.get('source', '')
            })
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            metadata_features = [0.0] * 6
        
        # Social context analysis
        social_context = analyze_social_context(metadata_features)
        
        # Model prediction
        model_result = predict_with_model(cleaned_text, metadata_features)
        
        # Advanced features analysis
        advanced_features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'avg_sentence_length': len(text.split()) / max(text.count('.') + text.count('!') + text.count('?'), 1),
            'has_exclamation': 1.0 if '!' in text else 0.0,
            'has_question': 1.0 if '?' in text else 0.0,
            'has_quotes': 1.0 if '"' in text or "'" in text else 0.0,
            'readability_score': min(len(text) / 100, 1.0)
        }
        
        processing_time = time.time() - start_time
        
        # Generate comprehensive response
        response = {
            'analysis_id': f"ana_{np.random.randint(10000, 99999)}_{int(time.time())}",
            'prediction': {
                'label': 'FAKE' if model_result.get('is_fake', False) else 'REAL',
                'confidence': float(model_result.get('confidence', 0.5)),
                'fake_probability': float(model_result.get('fake_probability', 0.5)),
                'real_probability': float(model_result.get('real_probability', 0.5)),
                'confidence_level': model_result.get('confidence_level', 'MEDIUM'),
                'uncertainty': float(model_result.get('uncertainty', 0.5))
            },
            'explanations': {
                'feature_importance': model_result.get('feature_importance', {}),
                'key_factors': model_result.get('explanation', 'Basic text analysis'),
                'detection_reasoning': generate_detection_reasoning(model_result, text),
                'model_used': 'Trained Multimodal Model' if model is not None else 'Heuristic Analysis'
            },
            'additional_analysis': {
                'linguistic_analysis': advanced_features,
                'social_context': social_context,
                'metadata_analysis': {
                    'source_credibility': metadata_features[5] if len(metadata_features) > 5 else 0.0,
                    'content_quality': min(len(text) / 500, 1.0)
                }
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': current_model_name,
                'processing_time': f"{processing_time:.3f}s",
                'text_length': len(text),
                'language_detected': 'en',
                'word_count': len(text.split()),
                'using_trained_model': model is not None
            }
        }
        
        # Add to analysis history
        analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'prediction': response['prediction']['label'],
            'confidence': response['prediction']['confidence'],
            'model_used': 'Trained Model' if model is not None else 'Heuristics'
        })
        
        # Update system metrics
        update_system_metrics()
        
        logger.info(f"✅ Analysis completed in {processing_time:.3f}s - "
                   f"Result: {response['prediction']['label']} - "
                   f"Confidence: {response['prediction']['confidence']} - "
                   f"Model: {'Trained' if model is not None else 'Heuristic'}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def generate_detection_reasoning(result, text):
    """Generate detailed reasoning for the detection result"""
    is_fake = result.get('is_fake', False)
    confidence = result.get('confidence', 0.5)
    
    if is_fake:
        reasons = [
            "Content contains sensational or emotional language",
            "Unverified claims without credible sources",
            "Patterns commonly found in misinformation"
        ]
    else:
        reasons = [
            "Content follows standard reporting patterns",
            "Reasonable language without excessive sensationalism",
            "Appears to be from credible sources"
        ]
    
    if confidence > 0.8:
        confidence_note = "High confidence in this analysis"
    elif confidence > 0.6:
        confidence_note = "Moderate confidence in this analysis"
    else:
        confidence_note = "Low confidence - analysis may be uncertain"
    
    return {
        'reasons': reasons,
        'confidence_note': confidence_note,
        'key_indicators': list(result.get('feature_importance', {}).keys())[:3]
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Legacy prediction endpoint for backward compatibility"""
    logger.info("🔮 Legacy predict endpoint accessed")
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        tweet_count = data.get('tweet_count', 0)
        
        if not text:
            logger.warning("❌ Predict failed - No text provided")
            return jsonify({'error': 'Text content is required'}), 400
        
        # Basic preprocessing
        cleaned_text = clean_text(text)
        
        # Create metadata
        metadata_features = extract_metadata_features({
            'text': cleaned_text,
            'tweet_count': tweet_count
        })
        
        # Get prediction
        result = predict_with_model(cleaned_text, metadata_features)
        
        # Prepare response
        response_result = {
            'prediction': 'Fake' if result['is_fake'] else 'Real',
            'confidence': result['confidence'],
            'fake_probability': result['fake_probability'],
            'real_probability': result['real_probability'],
            'model_used': current_model_name,
            'explanation': result['explanation'],
            'using_trained_model': model is not None
        }
        
        logger.info(f"✅ Legacy prediction completed - Result: {response_result['prediction']}")
        
        return jsonify(response_result)
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def performance_dashboard():
    """Performance monitoring dashboard"""
    logger.info("📊 Dashboard accessed")
    
    try:
        update_system_metrics()  # Ensure metrics are up to date
        
        # Calculate additional metrics for dashboard
        if analysis_history:
            recent_analyses = list(analysis_history)[-10:]  # Last 10 analyses
            confidence_trend = [analysis['confidence'] for analysis in recent_analyses]
            prediction_trend = [1 if analysis['prediction'] == 'FAKE' else 0 for analysis in recent_analyses]
        else:
            recent_analyses = []
            confidence_trend = []
            prediction_trend = []
        
        dashboard_data = {
            'performance_summary': {
                'total_requests': system_metrics['total_requests'],
                'fake_count': system_metrics['fake_count'],
                'real_count': system_metrics['real_count'],
                'fake_ratio': round(system_metrics['fake_count'] / max(system_metrics['total_requests'], 1), 3),
                'real_ratio': round(system_metrics['real_count'] / max(system_metrics['total_requests'], 1), 3),
                'average_confidence': system_metrics['avg_confidence'],
                'system_status': 'operational',
                'model_in_use': current_model_name,
                'uptime_hours': round(time.time() / 3600, 1),
                'model_loaded': model is not None,
                'last_update': system_metrics['last_update'].isoformat()
            },
            'trends': {
                'confidence_trend': confidence_trend,
                'prediction_trend': prediction_trend,
                'timestamps': [analysis['timestamp'] for analysis in recent_analyses]
            },
            'recent_alerts': [
                {
                    'type': 'System Healthy',
                    'message': 'All systems operational',
                    'severity': 'LOW',
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'recent_analyses': recent_analyses,
            'system_status': 'operational',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add alerts based on system state
        if system_metrics['fake_ratio'] > 0.5:
            dashboard_data['recent_alerts'].append({
                'type': 'High Fake Ratio',
                'message': f'Unusually high fake news ratio detected: {system_metrics["fake_ratio"]:.1%}',
                'severity': 'MEDIUM',
                'timestamp': datetime.now().isoformat()
            })
        
        if system_metrics['avg_confidence'] < 0.6:
            dashboard_data['recent_alerts'].append({
                'type': 'Low Confidence',
                'message': f'Average prediction confidence is low: {system_metrics["avg_confidence"]:.1%}',
                'severity': 'MEDIUM',
                'timestamp': datetime.now().isoformat()
            })
        
        logger.info(f"📈 Dashboard data generated - "
                   f"Total: {system_metrics['total_requests']}, "
                   f"Fake Ratio: {system_metrics['fake_count']/max(system_metrics['total_requests'], 1):.1%}")
        
        return jsonify(dashboard_data)
    
    except Exception as e:
        logger.error(f"❌ Dashboard error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Batch analysis endpoint for multiple texts"""
    start_time = time.time()
    logger.info("📦 Batch analysis request received")
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            logger.warning("❌ Batch analysis failed - Invalid texts array")
            return jsonify({'error': 'Texts array is required'}), 400
        
        if len(texts) > 100:
            logger.warning("❌ Batch analysis failed - Too many texts")
            return jsonify({'error': 'Batch size too large. Maximum 100 texts.'}), 400
        
        if len(texts) == 0:
            logger.warning("❌ Batch analysis failed - Empty texts array")
            return jsonify({'error': 'Texts array is empty'}), 400
        
        logger.info(f"📦 Processing batch of {len(texts)} texts...")
        
        results = []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text.strip()) < 10:
                results.append({
                    'text': text[:200] + '...' if text and len(text) > 200 else text,
                    'prediction': {
                        'label': 'ERROR',
                        'confidence': 0.0
                    },
                    'error': 'Invalid text input',
                    'analysis_id': f"ana_{np.random.randint(10000, 99999)}"
                })
                continue
            
            try:
                cleaned_text = clean_text(text)
                metadata_features = extract_metadata_features({'text': cleaned_text})
                result = predict_with_model(cleaned_text, metadata_features)
                
                results.append({
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'prediction': {
                        'label': 'FAKE' if result.get('is_fake', False) else 'REAL',
                        'confidence': float(result.get('confidence', 0.5))
                    },
                    'analysis_id': f"ana_{np.random.randint(10000, 99999)}",
                    'model_used': current_model_name,
                    'fake_probability': result.get('fake_probability', 0.5),
                    'real_probability': result.get('real_probability', 0.5),
                    'using_trained_model': model is not None
                })
                
                logger.info(f"📦 Batch item {i+1}: {result.get('is_fake', False)}")
                
            except Exception as e:
                results.append({
                    'text': text[:200] + '...' if text and len(text) > 200 else text,
                    'prediction': {
                        'label': 'ERROR',
                        'confidence': 0.0
                    },
                    'error': str(e),
                    'analysis_id': f"ana_{np.random.randint(10000, 99999)}"
                })
        
        processing_time = time.time() - start_time
        
        successful_analyses = sum(1 for r in results if r['prediction']['label'] != 'ERROR')
        logger.info(f"✅ Batch analysis completed - "
                   f"Successful: {successful_analyses}/{len(results)} - "
                   f"Time: {processing_time:.3f}s")
        
        return jsonify({
            'batch_id': f"batch_{np.random.randint(10000, 99999)}_{int(time.time())}",
            'results': results,
            'total_analyzed': len(results),
            'successful_analyses': successful_analyses,
            'failed_analyses': sum(1 for r in results if r['prediction']['label'] == 'ERROR'),
            'processing_time': f"{processing_time:.3f}s",
            'timestamp': datetime.now().isoformat(),
            'model_used': current_model_name,
            'trained_model_loaded': model is not None
        })
    
    except Exception as e:
        logger.error(f"❌ Batch analysis failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/news/trending')
def get_trending_news():
    """Fetch trending news from real APIs"""
    logger.info("📰 Trending news endpoint accessed")
    
    try:
        news_articles = get_real_news_from_apis()
        
        # Analyze each news item
        analyzed_news = []
        for news in news_articles:
            result = predict_with_model(news['content'], [])
            news['prediction'] = {
                'label': 'FAKE' if result.get('is_fake', False) else 'REAL',
                'confidence': round(result.get('confidence', 0.5), 3),
                'fake_probability': round(result.get('fake_probability', 0.5), 3),
                'real_probability': round(result.get('real_probability', 0.5), 3),
                'confidence_level': result.get('confidence_level', 'MEDIUM')
            }
            analyzed_news.append(news)
        
        logger.info(f"📰 Trending news generated - {len(analyzed_news)} items")
        
        return jsonify({
            'news': analyzed_news,
            'timestamp': datetime.now().isoformat(),
            'total_count': len(analyzed_news),
            'source': 'real_api' if analyzed_news and not analyzed_news[0].get('is_sample_fake') else 'sample',
            'model_used': current_model_name,
            'trained_model_loaded': model is not None,
            'apis_used': 'NewsAPI & GNews' if analyzed_news and not analyzed_news[0].get('is_sample_fake') else 'Sample Data'
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching trending news: {e}")
        return jsonify({'error': 'Failed to fetch news'}), 500

@app.route('/api/analyze/url', methods=['POST'])
def analyze_url():
    """Analyze news from a URL"""
    logger.info("🔗 URL analysis endpoint accessed")
    
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            logger.warning("❌ URL analysis failed - No URL provided")
            return jsonify({'error': 'URL is required'}), 400
        
        logger.info(f"🔗 Analyzing URL: {url}")
        
        # Simulate content extraction based on URL
        if 'miracle' in url.lower() or 'secret' in url.lower():
            sample_content = "Breaking news about a miraculous discovery that promises incredible results! This secret method will change everything you know!"
            logger.info("🔗 URL classified as potential fake news based on keywords")
        else:
            sample_content = "This article presents factual information based on research and verified sources. The content follows standard journalistic practices."
            logger.info("🔗 URL classified as potential real news")
        
        # Analyze the content
        result = predict_with_model(sample_content, [])
        
        logger.info(f"🔗 URL analysis completed - Prediction: {result.get('is_fake', False)}")
        
        return jsonify({
            'url': url,
            'content_preview': sample_content[:200] + '...',
            'prediction': {
                'label': 'FAKE' if result.get('is_fake', False) else 'REAL',
                'confidence': result.get('confidence', 0.5),
                'fake_probability': result.get('fake_probability', 0.5),
                'real_probability': result.get('real_probability', 0.5)
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'content_quality': 'high' if len(sample_content) > 100 else 'medium',
            'model_used': current_model_name,
            'trained_model_loaded': model is not None
        })
        
    except Exception as e:
        logger.error(f"❌ Error analyzing URL: {e}")
        return jsonify({'error': 'Failed to analyze URL'}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get system statistics"""
    logger.info("📈 Statistics endpoint accessed")
    
    try:
        update_system_metrics()  # Ensure metrics are up to date
        
        statistics = {
            'performance': {
                'total_requests': system_metrics['total_requests'],
                'fake_count': system_metrics['fake_count'],
                'real_count': system_metrics['real_count'],
                'fake_ratio': round(system_metrics['fake_count'] / max(system_metrics['total_requests'], 1), 3),
                'real_ratio': round(system_metrics['real_count'] / max(system_metrics['total_requests'], 1), 3),
                'average_confidence': system_metrics['avg_confidence'],
                'system_status': 'operational',
                'model_in_use': current_model_name,
                'trained_model_loaded': model is not None,
                'trained_model_usage_percentage': round(
                    sum(1 for a in analysis_history if a.get('model_used') == 'Trained Model') / max(len(analysis_history), 1) * 100, 1
                )
            },
            'system_health': {
                'model_loaded': model is not None,
                'model_name': current_model_name,
                'tokenizer_ready': tokenizer is not None,
                'gpu_available': torch.cuda.is_available(),
                'analysis_history_size': len(analysis_history),
                'real_time_news_count': len(real_time_news),
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time()
            }
        }
        
        logger.info(f"📈 Statistics generated - "
                   f"Total: {system_metrics['total_requests']}, "
                   f"Fake Ratio: {statistics['performance']['fake_ratio']:.1%}")
        
        return jsonify(statistics)
        
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.route('/system/info')
def system_info():
    """System information endpoint"""
    logger.info("🖥️ System info endpoint accessed")
    
    try:
        # Model information
        model_info = {
            'type': 'Multimodal Fake News Detector',
            'name': current_model_name,
            'loaded': model is not None,
            'device': str(device),
            'parameters': f'{sum(p.numel() for p in model.parameters()):,}' if model else 'Heuristic-based',
            'version': '2.0.0',
            'using_trained_model': model is not None
        }
        
        # System status
        system_status = {
            'components_initialized': {
                'model': model is not None,
                'tokenizer': tokenizer is not None,
                'analysis_engine': True,
                'news_feed': len(real_time_news) > 0,
                'api_connections': True
            },
            'performance': {
                'total_analyses': len(analysis_history),
                'average_confidence': system_metrics['avg_confidence'],
                'system_uptime': f"{time.time() / 3600:.1f} hours",
                'trained_model_usage': f"{sum(1 for a in analysis_history if a.get('model_used') == 'Trained Model') / max(len(analysis_history), 1) * 100:.1f}%",
                'news_feed_size': len(real_time_news)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("🖥️ System info generated successfully")
        
        return jsonify({
            'system': system_status,
            'model': model_info,
            'version': '2.0.0'
        })
        
    except Exception as e:
        logger.error(f"❌ System info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback about analysis results"""
    logger.info("💬 Feedback endpoint accessed")
    
    try:
        data = request.get_json()
        analysis_id = data.get('analysis_id', '')
        correct = data.get('correct', True)
        feedback_text = data.get('feedback', '')
        
        logger.info(f"💬 Feedback received - Analysis: {analysis_id}, Correct: {correct}, Feedback: {feedback_text[:50]}...")
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback received successfully',
            'feedback_id': f"fb_{np.random.randint(10000, 99999)}",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {e}")
        return jsonify({'error': 'Failed to submit feedback'}), 500

@app.route('/api/realtime/news')
def get_realtime_news():
    """Get real-time news feed"""
    logger.info("🔄 Real-time news endpoint accessed")
    
    try:
        # Update time_ago for all news items
        for news in real_time_news:
            try:
                published_time = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
                time_diff = datetime.now().astimezone() - published_time
                if time_diff.total_seconds() < 60:
                    news['time_ago'] = 'Just now'
                elif time_diff.total_seconds() < 3600:
                    minutes = int(time_diff.total_seconds() / 60)
                    news['time_ago'] = f'{minutes} min ago'
                else:
                    hours = int(time_diff.total_seconds() / 3600)
                    news['time_ago'] = f'{hours} hours ago'
            except:
                news['time_ago'] = 'Recently'
        
        logger.info(f"📰 Real-time news served - {len(real_time_news)} items")
        
        return jsonify({
            'news': real_time_news,
            'timestamp': datetime.now().isoformat(),
            'total_count': len(real_time_news),
            'model_used': current_model_name,
            'trained_model_loaded': model is not None
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching real-time news: {e}")
        return jsonify({'error': 'Failed to fetch real-time news'}), 500

@app.route('/api/history')
def get_analysis_history():
    """Get analysis history"""
    logger.info("📚 History endpoint accessed")
    
    try:
        history_list = list(analysis_history)[-20:]  # Last 20 analyses
        
        logger.info(f"📚 History served - {len(history_list)} items")
        
        return jsonify({
            'history': history_list,
            'total_count': len(analysis_history),
            'timestamp': datetime.now().isoformat(),
            'model_used': current_model_name,
            'trained_model_loaded': model is not None
        })
    except Exception as e:
        logger.error(f"❌ Error fetching history: {e}")
        return jsonify({'error': 'Failed to fetch history'}), 500

@app.route('/api/sources')
def get_sources():
    """Get source database"""
    logger.info("📋 Sources endpoint accessed")
    
    try:
        sources = [
            {
                'name': 'BBC News', 'type': 'mainstream', 'credibility': 95, 'country': 'UK', 'verified': True,
                'description': 'Public service broadcaster known for high journalistic standards'
            },
            {
                'name': 'Reuters', 'type': 'mainstream', 'credibility': 96, 'country': 'International', 'verified': True,
                'description': 'International news organization with strong fact-checking'
            },
            {
                'name': 'Associated Press', 'type': 'mainstream', 'credibility': 94, 'country': 'USA', 'verified': True,
                'description': 'Non-profit news cooperative with global coverage'
            },
            {
                'name': 'CNN', 'type': 'mainstream', 'credibility': 88, 'country': 'USA', 'verified': True,
                'description': 'Major cable news network with comprehensive coverage'
            },
            {
                'name': 'Social Media', 'type': 'social', 'credibility': 45, 'country': 'Global', 'verified': False,
                'description': 'User-generated content with variable reliability'
            }
        ]
        
        logger.info(f"📋 Sources served - {len(sources)} sources")
        
        return jsonify({
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error fetching sources: {e}")
        return jsonify({'error': 'Failed to fetch sources'}), 500

@app.route('/api/status')
def api_status():
    """Check status of external APIs"""
    logger.info("🌐 API status check requested")
    
    try:
        # Test NewsAPI
        newsapi_status = "unknown"
        try:
            newsapi_url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey={config.NEWS_API_KEY}"
            response = requests.get(newsapi_url, timeout=5)
            newsapi_status = "active" if response.status_code == 200 else f"inactive ({response.status_code})"
            logger.info(f"🌐 NewsAPI status: {newsapi_status}")
        except Exception as e:
            newsapi_status = f"error: {str(e)}"
            logger.warning(f"🌐 NewsAPI check failed: {e}")
        
        # Test GNews API
        gnews_status = "unknown"
        try:
            gnews_url = f"https://gnews.io/api/v4/top-headlines?token={config.GNEWS_API_KEY}&lang=en&max=1"
            response = requests.get(gnews_url, timeout=5)
            gnews_status = "active" if response.status_code == 200 else f"inactive ({response.status_code})"
            logger.info(f"🌐 GNews API status: {gnews_status}")
        except Exception as e:
            gnews_status = f"error: {str(e)}"
            logger.warning(f"🌐 GNews API check failed: {e}")
        
        return jsonify({
            'apis': {
                'newsapi': newsapi_status,
                'gnews': gnews_status
            },
            'timestamp': datetime.now().isoformat(),
            'using_real_apis': newsapi_status == 'active' or gnews_status == 'active'
        })
        
    except Exception as e:
        logger.error(f"❌ API status check failed: {e}")
        return jsonify({'error': 'API status check failed'}), 500

@app.route('/refresh_news')
def refresh_news():
    """Manually refresh the news feed"""
    logger.info("🔄 Manual news refresh requested")
    
    try:
        global real_time_news
        real_time_news = generate_real_time_news()
        
        logger.info(f"✅ News feed manually refreshed with {len(real_time_news)} items")
        
        return jsonify({
            'status': 'success',
            'message': f'News feed refreshed with {len(real_time_news)} items',
            'timestamp': datetime.now().isoformat(),
            'news_count': len(real_time_news)
        })
    except Exception as e:
        logger.error(f"❌ Error refreshing news: {e}")
        return jsonify({'error': 'Failed to refresh news'}), 500

@app.route('/clear_history')
def clear_history():
    """Clear analysis history"""
    logger.info("🗑️ Clear history requested")
    
    try:
        global analysis_history, system_metrics
        analysis_history.clear()
        system_metrics = {
            'total_requests': 0,
            'fake_count': 0,
            'real_count': 0,
            'avg_confidence': 0.0,
            'last_update': datetime.now()
        }
        
        logger.info("✅ Analysis history cleared")
        
        return jsonify({
            'status': 'success',
            'message': 'Analysis history cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Error clearing history: {e}")
        return jsonify({'error': 'Failed to clear history'}), 500

@app.errorhandler(404)
def not_found(error):
    logger.warning(f"❌ 404 Error: {request.url}")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"❌ 500 Internal Server Error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Fake News Detection API")
    print("=" * 60)
    print(f"📱 Device: {device}")
    print(f"🤖 Model: {current_model_name}")
    print(f"🔤 Tokenizer: {'✅ Ready' if tokenizer is not None else '❌ Not available'}")
    print(f"🧠 Trained Model: {'✅ Loaded' if model is not None else '❌ Not available'}")
    print(f"📊 Analysis History: {len(analysis_history)} records")
    print(f"📰 Real-time News: {len(real_time_news)} items loaded")
    print(f"🌐 API Keys: NewsAPI {'✅ Loaded' if config.NEWS_API_KEY else '❌ Missing'}, GNews {'✅ Loaded' if config.GNEWS_API_KEY else '❌ Missing'}")
    print("=" * 60)
    print("🌐 ACCESS YOUR APPLICATION AT:")
    print("   🔗 http://localhost:5000")
    print("   🔗 http://127.0.0.1:5000")
    print("")
    print("🌐 Available Endpoints:")
    print("   GET  /health          - Health check")
    print("   GET  /test/model      - Test model loading")
    print("   POST /analyze         - Single text analysis") 
    print("   POST /predict         - Legacy prediction")
    print("   POST /batch_analyze   - Batch analysis")
    print("   GET  /dashboard       - Performance dashboard")
    print("   GET  /system/info     - System information")
    print("   GET  /api/news/trending - Trending news analysis")
    print("   POST /api/analyze/url - URL analysis")
    print("   POST /api/feedback    - Submit feedback")
    print("   GET  /api/realtime/news - Real-time news feed")
    print("   GET  /api/history     - Analysis history")
    print("   GET  /api/sources     - Source database")
    print("   GET  /api/status      - API status check")
    print("   GET  /refresh_news    - Refresh news feed")
    print("   GET  /clear_history   - Clear analysis history")
    print("=" * 60)
    print("📝 Logs are being written to app.log")
    print("🔄 Background news refresh is active (every 5 minutes)")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./evaluation_results', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
