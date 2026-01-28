# HealthCare AI - Intelligent Disease Prediction System

A machine learning-powered health assistant that provides disease predictions based on symptoms, BMI calculations, and health information.

## 🚨 IMPORTANT DISCLAIMER

**This application is for EDUCATIONAL PURPOSES ONLY.**

- This tool does NOT provide medical diagnosis or treatment advice
- Always consult qualified healthcare professionals for medical concerns
- In case of emergency, call your local emergency number immediately
- AI predictions have limitations and may be inaccurate

## Features

### 🧠 ML-Powered Disease Prediction
- **Real Machine Learning**: Uses scikit-learn with Random Forest and Gradient Boosting ensemble
- **70+ Diseases**: Comprehensive database with genuine medical data covering 15+ categories
- **Symptom Vectorization**: TF-IDF-like encoding with synonym resolution
- **Risk-Adjusted Predictions**: Accounts for age, symptom duration, and severity
- **Categories Include**: Respiratory, Cardiovascular, Gastrointestinal, Neurological, Endocrine, Mental Health, Musculoskeletal, Skin, Renal, Eye, Infectious, Environmental, Sleep disorders

### 🚨 Emergency Detection
- **Critical Symptom Combinations**: Detects life-threatening conditions
- **Multi-factor Assessment**: Considers symptom combinations, duration, and age
- **Immediate Alerts**: Clear warnings with emergency contact information

### ⚖️ BMI Calculator
- BMI calculation with category classification
- Basal Metabolic Rate (BMR) calculation
- Total Daily Energy Expenditure (TDEE)
- Personalized health advice

### 💬 Health Chatbot
- NLP-powered responses on health topics
- Emergency keyword detection
- Disease information lookup

### 🔒 Security Features
- Input sanitization (XSS prevention)
- Request validation
- Rate limiting
- Secure data handling

## Project Structure

```
healthcare_ai/
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md
├── models/
│   ├── __init__.py
│   ├── disease_predictor.py  # ML model implementation
│   └── trained_model.joblib  # Saved model (auto-generated)
├── data/
│   ├── diseases.json         # Disease database
│   └── symptoms.json         # Symptom mappings & synonyms
├── utils/
│   ├── __init__.py
│   ├── sanitizer.py          # Input sanitization
│   └── emergency.py          # Emergency detection
├── templates/
│   └── index.html            # Main HTML template
└── static/
    ├── css/
    │   └── styles.css        # Custom styles
    └── js/
        └── main.js           # Frontend JavaScript
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd healthcare_ai
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **(Optional) Install scikit-learn for enhanced ML**

If you have Visual C++ Build Tools installed:
```bash
pip install scikit-learn
```

Without scikit-learn, the app will use a pure Python prediction engine that still works well.

5. **Run the application**
```bash
python app.py
```

6. **Open in browser**
```
http://localhost:5000
```

### Troubleshooting

**Python 3.13+ users:** Some packages may not have pre-built wheels yet. The application will work without scikit-learn using the pure Python fallback predictor.

**Windows users without Visual C++ Build Tools:** 
- The app works without scikit-learn
- Or install Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## API Endpoints

### Disease Prediction
```http
POST /api/predict
Content-Type: application/json

{
    "symptoms": ["fever", "cough", "fatigue"],
    "duration": "1_to_3_days",
    "age": 35
}
```

### BMI Calculation
```http
POST /api/bmi
Content-Type: application/json

{
    "weight": 70,
    "height": 175,
    "age": 30,
    "gender": "male",
    "activity": 1.55
}
```

### Health Chat
```http
POST /api/chat
Content-Type: application/json

{
    "message": "What should I do for a headache?"
}
```

### Get Model Metrics
```http
GET /api/model/metrics
```

### Get Symptoms
```http
GET /api/symptoms
```

### Get Diseases
```http
GET /api/diseases
```

## ML Model Details

### Architecture
- **Primary Model**: Random Forest Classifier (100 estimators)
- **Secondary Model**: Gradient Boosting Classifier (50 estimators)
- **Ensemble**: Weighted average (70% RF, 30% GB)

### Training
- Auto-trains on first run using disease-symptom data
- Generates augmented samples with symptom variations
- Cross-validation for accuracy measurement
- Model persisted to disk for future use

### Features
- Symptom vectorization with weight support
- Synonym resolution for natural input
- Duration and age-based risk adjustment
- Confidence scoring with symptom matching

## Configuration

Edit `config.py` to customize:

```python
class Config:
    MODEL_PATH = 'models/trained_model.joblib'
    MIN_CONFIDENCE_THRESHOLD = 0.15
    MAX_PREDICTIONS = 5
    EMERGENCY_URGENCY_THRESHOLD = 4
```

## Environment Variables

```bash
# Optional
export SECRET_KEY=your-secret-key
export DEBUG=False
export PORT=5000
```

## Development

### Running in Debug Mode
```bash
export DEBUG=True
python app.py
```

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Diseases
Edit `data/diseases.json`:
```json
{
    "New Disease": {
        "category": "Category",
        "symptoms": ["symptom1", "symptom2"],
        "weights": {"symptom1": 0.9, "symptom2": 0.8},
        "severity": "moderate",
        "urgency": 2,
        "description": "Description...",
        "recommendations": "Recommendations...",
        "precautions": ["Precaution 1", "Precaution 2"],
        "when_to_seek_help": "When to seek help...",
        "risk_factors": ["Risk 1", "Risk 2"]
    }
}
```

After adding, delete `models/trained_model.joblib` to retrain.

## Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Known Limitations

1. **Not a Medical Device**: This is an educational tool, not a certified medical device
2. **Disease Database**: 70+ diseases currently supported
3. **English Only**: Currently only supports English input
4. **No User Accounts**: No persistent user data storage
5. **Simplified NLP**: Chatbot uses keyword matching, not advanced NLP

## Future Improvements

- [ ] Add more diseases to database
- [ ] Implement deep learning models (TensorFlow/PyTorch)
- [ ] Add multi-language support
- [ ] Integrate with medical APIs
- [ ] Add user authentication
- [ ] Mobile app version

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

This project is for educational purposes only. Not licensed for commercial medical use.

## Acknowledgments

- Disease information compiled from public health resources
- ML implementation using scikit-learn
- UI built with Tailwind CSS

---

**⚠️ Remember: Always consult healthcare professionals for medical advice. This tool is for educational purposes only.**