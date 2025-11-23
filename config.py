import torch
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models")
LOG_PATH = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# List of dataset files expected
DATASET_FILES = [
    "data1.csv",
    "data2.csv", 
    "data3.csv",
    "gossipcop_fake.csv",
    "gossipcop_real.csv",
    "politifact_fake.csv",
    "politifact_real.csv",
]

# Model configuration
TEXT_MODEL_NAME = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 8
NUM_EPOCHS = 3
NUM_CLASSES = 2
METADATA_DIM = 9  # Updated to match our feature extraction

# Training configuration
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
GRADIENT_ACCUMULATION_STEPS = 2

# Feature configuration
USE_FOCAL_LOSS = False
USE_AMP = True

# Available trained models
AVAILABLE_MODELS = [
    "enhanced_model.pth",
    "fallback_model.pth"
]

# API configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '0742802fb6b843bc87c2b13b9e1f1be0')
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY', '510006514cda39e2c0b0644bec34fe85')


# Real-time Settings
NEWS_REFRESH_INTERVAL = 300  # 5 minutes
MAX_NEWS_ITEMS = 20
ANALYSIS_HISTORY_SIZE = 1000

# UI Settings
DASHBOARD_REFRESH_RATE = 60  # 1 minute
MAX_TEXT_LENGTH = 10000
MIN_TEXT_LENGTH = 10

# Analysis thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.6
CONFIDENCE_LOW = 0.4

# Fake news detection parameters
SENSATIONAL_WORDS = [
    'breaking', 'shocking', 'unbelievable', 'miracle', 'secret', 
    'hidden', 'they don\'t want you to know', 'mainstream media',
    'conspiracy', 'wake up', 'elite', 'suppressed'
]

CREDIBLE_INDICATORS = [
    'according to sources', 'official statement', 'research shows',
    'study found', 'experts say', 'peer-reviewed', 'clinical trial',
    'scientific evidence', 'university study', 'journal publication'
]

class Config:
    # Text model
    TEXT_MODEL_NAME = TEXT_MODEL_NAME
    MAX_LEN = MAX_LEN
    
    # Training
    BATCH_SIZE = BATCH_SIZE
    NUM_EPOCHS = NUM_EPOCHS
    LEARNING_RATE = LEARNING_RATE
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS
    EARLY_STOPPING_PATIENCE = EARLY_STOPPING_PATIENCE
    
    # Model dimensions
    METADATA_DIM = METADATA_DIM
    NUM_CLASSES = NUM_CLASSES
    
    # Paths
    DATA_PATH = DATA_PATH
    MODEL_SAVE_PATH = MODEL_SAVE_PATH
    LOG_PATH = LOG_PATH
    DATASET_FILES = DATASET_FILES
    
    # Device
    device = device
    
    # Features
    USE_AMP = USE_AMP
    USE_FOCAL_LOSS = USE_FOCAL_LOSS
    
    # APIs
    NEWS_API_KEY = NEWS_API_KEY
    GNEWS_API_KEY = GNEWS_API_KEY

    
    # Analysis settings
    MAX_TEXT_LENGTH = MAX_TEXT_LENGTH
    MIN_TEXT_LENGTH = MIN_TEXT_LENGTH
    CONFIDENCE_HIGH = CONFIDENCE_HIGH
    CONFIDENCE_MEDIUM = CONFIDENCE_MEDIUM
    CONFIDENCE_LOW = CONFIDENCE_LOW
    
    # Content analysis
    SENSATIONAL_WORDS = SENSATIONAL_WORDS
    CREDIBLE_INDICATORS = CREDIBLE_INDICATORS

config = Config()