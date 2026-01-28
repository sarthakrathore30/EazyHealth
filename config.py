import os

# Try to load dotenv, but don't fail if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Model settings
    MODEL_PATH = 'models/trained_model.joblib'
    VECTORIZER_PATH = 'models/symptom_vectorizer.joblib'
    LABEL_ENCODER_PATH = 'models/label_encoder.joblib'
    
    # Data paths
    DISEASES_DATA_PATH = 'data/diseases.json'
    SYMPTOMS_DATA_PATH = 'data/symptoms.json'
    
    # ML Model parameters
    MIN_CONFIDENCE_THRESHOLD = 0.15
    MAX_PREDICTIONS = 5
    
    # Emergency thresholds
    EMERGENCY_URGENCY_THRESHOLD = 4
    HIGH_RISK_SYMPTOM_COUNT = 3
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60