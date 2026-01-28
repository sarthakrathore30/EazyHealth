"""
HealthCare AI - Flask Application
Machine Learning powered disease prediction system
"""

import os
import sys
import json
from datetime import datetime
from functools import wraps
import time

# Check Python version
print(f"Python version: {sys.version}")

from flask import Flask, render_template, request, jsonify

# Import custom modules
from config import Config

# Import with error handling
try:
    from models.disease_predictor import DiseasePredictionModel
except ImportError as e:
    print(f"Error importing disease_predictor: {e}")
    sys.exit(1)

try:
    from utils.sanitizer import InputSanitizer, RequestValidator
except ImportError as e:
    print(f"Warning: sanitizer not available: {e}")
    # Provide fallback
    class InputSanitizer:
        @classmethod
        def sanitize_chat_message(cls, msg):
            return msg[:1000] if msg else None
    class RequestValidator:
        @staticmethod
        def validate_prediction_request(data, valid_symptoms):
            symptoms = data.get('symptoms', [])
            if not symptoms:
                return False, "No symptoms"
            valid = [s for s in symptoms if s in valid_symptoms]
            return True, {'symptoms': valid, 'duration': data.get('duration', '1_to_3_days'), 'age': data.get('age')}
        @staticmethod
        def validate_bmi_request(data):
            try:
                return True, {
                    'weight': float(data.get('weight', 0)),
                    'height': float(data.get('height', 0)),
                    'age': int(data.get('age', 0)),
                    'gender': data.get('gender', 'male'),
                    'activity': float(data.get('activity', 1.2))
                }
            except:
                return False, "Invalid data"
        @staticmethod
        def validate_chat_request(data):
            msg = data.get('message', '')
            return (True, {'message': msg}) if msg else (False, "No message")

try:
    from utils.emergency import EmergencyDetector
except ImportError as e:
    print(f"Warning: emergency detector not available: {e}")
    # Provide fallback
    class EmergencyDetector:
        def __init__(self, path=None):
            pass
        def assess_emergency(self, symptoms, duration=None, age=None):
            class Assessment:
                level = type('Level', (), {'name': 'NONE'})()
                is_emergency = False
                message = ""
                matched_symptoms = []
                recommendations = []
                call_emergency = False
            return Assessment()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize ML model
print("Initializing ML Disease Prediction Model...")
disease_model = DiseasePredictionModel(
    diseases_path=Config.DISEASES_DATA_PATH,
    symptoms_path=Config.SYMPTOMS_DATA_PATH,
    model_path=Config.MODEL_PATH
)
print("Model initialization complete!")

# Initialize emergency detector
emergency_detector = EmergencyDetector(Config.SYMPTOMS_DATA_PATH)

# In-memory storage for health records (in production, use a database)
health_records = []
bmi_records = []

# Rate limiting decorator
request_counts = {}

def rate_limit(max_requests=60, window=60):
    """Simple rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Clean old entries
            request_counts[client_ip] = [
                t for t in request_counts.get(client_ip, [])
                if current_time - t < window
            ]
            
            # Check rate limit
            if len(request_counts.get(client_ip, [])) >= max_requests:
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
            
            # Record request
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            request_counts[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return wrapped
    return decorator


# Chatbot knowledge base
CHAT_KNOWLEDGE = {
    'greetings': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening'],
    'thanks': ['thank', 'thanks', 'appreciate', 'helpful'],
    'topics': {
        'fever': {
            'keywords': ['fever', 'temperature', 'hot'],
            'response': "Fever is typically a sign your body is fighting an infection. For adults, a fever is generally considered 100.4°F (38°C) or higher. Stay hydrated, rest, and take fever-reducing medications like acetaminophen or ibuprofen. <br><br><strong>Seek medical attention if:</strong> fever is above 103°F (39.4°C), lasts more than 3 days, or is accompanied by severe symptoms."
        },
        'headache': {
            'keywords': ['headache', 'head pain', 'migraine'],
            'response': "Headaches can have many causes including tension, dehydration, stress, or underlying conditions. Try resting in a quiet, dark room, staying hydrated, and taking OTC pain relievers. <br><br><strong>Seek immediate care for:</strong> sudden severe headaches, headaches with fever/stiff neck, or headaches after head injury."
        },
        'diet': {
            'keywords': ['diet', 'nutrition', 'eating', 'food', 'healthy eating'],
            'response': "A balanced diet should include:<ul><li>Plenty of fruits and vegetables (5+ servings daily)</li><li>Whole grains</li><li>Lean proteins</li><li>Healthy fats</li><li>Adequate water (8+ glasses daily)</li></ul>Limit processed foods, added sugars, and excessive salt."
        },
        'exercise': {
            'keywords': ['exercise', 'workout', 'fitness', 'physical activity'],
            'response': "Adults should aim for:<ul><li>At least 150 minutes of moderate-intensity OR 75 minutes of vigorous-intensity aerobic activity weekly</li><li>Muscle-strengthening activities 2+ days per week</li></ul>Start slowly if new to exercise. Consult a doctor before starting if you have health conditions."
        },
        'sleep': {
            'keywords': ['sleep', 'insomnia', 'tired', 'fatigue', 'rest'],
            'response': "Adults need 7-9 hours of quality sleep per night. Tips for better sleep:<ul><li>Maintain consistent sleep schedule</li><li>Create dark, quiet environment</li><li>Avoid screens before bed</li><li>Limit caffeine after noon</li><li>Avoid large meals before sleeping</li></ul>"
        },
        'stress': {
            'keywords': ['stress', 'anxiety', 'worried', 'nervous', 'mental health'],
            'response': "Chronic stress impacts both mental and physical health. Helpful strategies:<ul><li>Regular exercise</li><li>Mindfulness meditation</li><li>Deep breathing exercises</li><li>Adequate sleep</li><li>Limiting caffeine/alcohol</li><li>Social connections</li></ul>If stress significantly impacts your life, consider speaking with a mental health professional."
        },
        'cold': {
            'keywords': ['cold', 'runny nose', 'sneezing', 'congestion'],
            'response': "Common colds are viral and usually resolve within 7-10 days. Treatment focuses on symptom relief: rest, stay hydrated, use saline nasal spray, take OTC cold medications. <br><br>Seek medical attention if symptoms worsen after a week, you have high fever, or develop severe symptoms."
        },
        'covid': {
            'keywords': ['covid', 'coronavirus', 'covid-19'],
            'response': "COVID-19 symptoms include fever, cough, fatigue, loss of taste/smell, and shortness of breath. <br><br><strong>If you suspect COVID-19:</strong><ul><li>Get tested</li><li>Isolate from others</li><li>Monitor symptoms</li><li>Seek emergency care if you have difficulty breathing, persistent chest pressure, confusion, or bluish lips</li></ul>"
        },
        'heart': {
            'keywords': ['heart', 'cardiac', 'cardiovascular', 'blood pressure'],
            'response': "Heart health requires:<ul><li>Regular exercise</li><li>Balanced diet low in saturated fats</li><li>Not smoking</li><li>Managing stress</li><li>Healthy weight</li><li>Regular check-ups</li></ul><br><strong>Know warning signs of heart attack:</strong> chest pain/pressure, pain radiating to arm/jaw, shortness of breath, cold sweat."
        },
        'diabetes': {
            'keywords': ['diabetes', 'blood sugar', 'glucose', 'insulin'],
            'response': "Diabetes management involves:<ul><li>Regular blood sugar monitoring</li><li>Taking medications as prescribed</li><li>Balanced diet with controlled carbohydrates</li><li>Regular exercise</li><li>Maintaining healthy weight</li></ul>Work closely with your healthcare team and attend regular check-ups."
        }
    }
}


# ============== ROUTES ==============

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/model/metrics', methods=['GET'])
@rate_limit()
def get_model_metrics():
    """Get ML model performance metrics"""
    metrics = disease_model.get_model_metrics()
    return jsonify(metrics)


@app.route('/api/symptoms', methods=['GET'])
@rate_limit()
def get_symptoms():
    """Get all valid symptoms and categories"""
    all_symptoms = sorted(list(disease_model.get_valid_symptoms()))
    categories = disease_model.get_symptoms_by_category()
    
    return jsonify({
        'symptoms': all_symptoms,
        'categories': categories
    })


@app.route('/api/diseases', methods=['GET'])
@rate_limit()
def get_diseases():
    """Get all diseases from database"""
    diseases = disease_model.get_all_diseases()
    return jsonify({'diseases': diseases})


@app.route('/api/predict', methods=['POST'])
@rate_limit(max_requests=30)
def predict_disease():
    """
    Predict diseases based on symptoms using ML model
    
    Expected JSON:
    {
        "symptoms": ["symptom1", "symptom2", ...],
        "duration": "1_to_3_days",
        "age": 35  // optional
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate and sanitize input
    valid_symptoms = disease_model.get_valid_symptoms()
    is_valid, result = RequestValidator.validate_prediction_request(data, valid_symptoms)
    
    if not is_valid:
        return jsonify({'error': result}), 400
    
    symptoms = result['symptoms']
    duration = result['duration']
    age = result['age']
    
    # Check for emergency
    emergency_assessment = emergency_detector.assess_emergency(symptoms, duration, age)
    
    # Get ML predictions
    predictions = disease_model.predict(
        symptoms=symptoms,
        duration=duration,
        age=age,
        top_k=Config.MAX_PREDICTIONS
    )
    
    # Format predictions for response
    predictions_data = []
    for pred in predictions:
        predictions_data.append({
            'disease': pred.disease,
            'confidence': pred.confidence,
            'matched_symptoms': pred.matched_symptoms,
            'severity': pred.severity,
            'urgency': pred.urgency,
            'category': pred.category,
            'recommendations': pred.recommendations,
            'precautions': pred.precautions,
            'when_to_seek_help': pred.when_to_seek_help,
            'risk_factors': pred.risk_factors
        })
    
    # Store in health records
    if predictions_data:
        health_records.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symptoms': symptoms,
            'disease': predictions_data[0]['disease'],
            'confidence': predictions_data[0]['confidence']
        })
        # Keep only last 100 records
        while len(health_records) > 100:
            health_records.pop(0)
    
    return jsonify({
        'predictions': predictions_data,
        'emergency': {
            'level': emergency_assessment.level.name,
            'is_emergency': emergency_assessment.is_emergency,
            'message': emergency_assessment.message,
            'matched_symptoms': emergency_assessment.matched_symptoms,
            'recommendations': emergency_assessment.recommendations,
            'call_emergency': emergency_assessment.call_emergency
        }
    })


@app.route('/api/bmi', methods=['POST'])
@rate_limit()
def calculate_bmi():
    """
    Calculate BMI and provide health insights
    
    Expected JSON:
    {
        "weight": 70,
        "height": 175,
        "age": 30,
        "gender": "male",
        "activity": 1.55
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate input
    is_valid, result = RequestValidator.validate_bmi_request(data)
    
    if not is_valid:
        return jsonify({'error': result}), 400
    
    weight = result['weight']
    height = result['height'] / 100  # Convert cm to m
    age = result['age']
    gender = result['gender']
    activity = result['activity']
    
    # Calculate BMI
    bmi = round(weight / (height ** 2), 1)
    
    # Determine category
    if bmi < 18.5:
        category = "Underweight"
        advice = "Consider consulting a nutritionist to develop a healthy weight gain plan with balanced nutrition and possibly increased caloric intake."
    elif bmi < 25:
        category = "Normal weight"
        advice = "Great job! Maintain your healthy lifestyle with regular exercise and balanced nutrition. Continue with regular health check-ups."
    elif bmi < 30:
        category = "Overweight"
        advice = "Consider increasing physical activity and reviewing dietary habits. Focus on whole foods, reduce processed foods, and aim for gradual weight loss if desired."
    else:
        category = "Obese"
        advice = "Please consult a healthcare provider for a comprehensive weight management plan. They can help create a safe, sustainable approach to improving your health."
    
    # Calculate ideal weight range
    ideal_low = round(18.5 * (height ** 2), 1)
    ideal_high = round(24.9 * (height ** 2), 1)
    
    # Calculate BMR using Mifflin-St Jeor Equation
    if gender == 'male':
        bmr = round(10 * weight + 6.25 * (height * 100) - 5 * age + 5)
    else:
        bmr = round(10 * weight + 6.25 * (height * 100) - 5 * age - 161)
    
    # Calculate TDEE
    tdee = round(bmr * activity)
    
    # Store BMI record
    bmi_records.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'bmi': bmi,
        'category': category
    })
    # Keep only last 100 records
    while len(bmi_records) > 100:
        bmi_records.pop(0)
    
    return jsonify({
        'bmi': bmi,
        'category': category,
        'advice': advice,
        'ideal_weight_range': f"{ideal_low} - {ideal_high} kg",
        'bmr': bmr,
        'tdee': tdee
    })


@app.route('/api/chat', methods=['POST'])
@rate_limit()
def chat():
    """
    Health chatbot endpoint
    
    Expected JSON:
    {
        "message": "user message here"
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate input
    is_valid, result = RequestValidator.validate_chat_request(data)
    
    if not is_valid:
        return jsonify({'error': result}), 400
    
    message = result['message'].lower()
    
    # Check for emergency keywords first
    emergency_words = ['emergency', 'dying', "can't breathe", 'heart attack', 'stroke', 
                       'severe pain', 'unconscious', 'suicide', 'kill myself']
    if any(word in message for word in emergency_words):
        return jsonify({
            'response': """<span class='text-red-400 font-bold'>🚨 If this is a medical emergency, please call emergency services immediately:</span>
            <ul class='mt-2'>
                <li><strong>US:</strong> 911</li>
                <li><strong>UK:</strong> 1066</li>
                <li><strong>EU:</strong> 112</li>
                <li><strong>Mental Health Crisis (US):</strong> 988</li>
            </ul>
            <p class='mt-2'>Do not wait for online advice in an emergency. Every minute can be critical.</p>"""
        })
    
    # Check for greetings
    if any(word in message for word in CHAT_KNOWLEDGE['greetings']):
        return jsonify({
            'response': "Hello! I'm your health assistant. How can I help you today? Ask me about symptoms, health tips, nutrition, exercise, or wellness advice!<br><br><span class='text-xs text-yellow-400'>⚠️ Remember: I provide general information only, not medical advice.</span>"
        })
    
    # Check for thanks
    if any(word in message for word in CHAT_KNOWLEDGE['thanks']):
        return jsonify({
            'response': "You're welcome! Remember, I provide general information only. Always consult healthcare professionals for medical advice. Stay healthy! 🏥"
        })
    
    # Find matching topic
    for topic, data in CHAT_KNOWLEDGE['topics'].items():
        if any(keyword in message for keyword in data['keywords']):
            return jsonify({
                'response': f"{data['response']}<br><br><span class='text-xs text-yellow-400'>⚠️ This is general information. Consult a healthcare provider for personalized advice.</span>"
            })
    
    # Check for disease-related questions
    diseases = disease_model.get_all_diseases()
    for disease_name, info in diseases.items():
        if disease_name.lower() in message:
            symptoms_text = ', '.join(info.get('symptoms', [])[:5])
            return jsonify({
                'response': f"""<strong>{disease_name}</strong><br><br>
                {info.get('description', 'No description available.')}<br><br>
                <strong>Common symptoms:</strong> {symptoms_text}<br><br>
                <strong>Recommendations:</strong> {info.get('recommendations', 'Consult a healthcare provider.')}<br><br>
                <span class='text-xs text-yellow-400'>⚠️ This is general information. Please consult a healthcare provider for diagnosis and treatment.</span>"""
            })
    
    # Default response
    return jsonify({
        'response': """I can help with general health information on topics like:
        <ul class='mt-2 space-y-1'>
            <li>• Common symptoms (fever, headache, cold)</li>
            <li>• Lifestyle (diet, exercise, sleep, stress)</li>
            <li>• Conditions (diabetes, heart health)</li>
            <li>• Prevention (vaccination, hydration)</li>
        </ul>
        <br>Try asking about a specific health topic, or use our <strong>Symptom Analyzer</strong> for symptom-based guidance.
        <br><br><span class='text-xs text-yellow-400'>⚠️ For medical advice, always consult a healthcare professional.</span>"""
    })


@app.route('/api/records', methods=['GET'])
@rate_limit()
def get_records():
    """Get health records history"""
    return jsonify({
        'health_records': health_records[-20:],  # Last 20 records
        'bmi_records': bmi_records[-20:]
    })


@app.route('/api/records/clear', methods=['POST'])
@rate_limit()
def clear_records():
    """Clear all health records"""
    health_records.clear()
    bmi_records.clear()
    return jsonify({'status': 'success', 'message': 'All records cleared'})


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400


# ============== MAIN ==============

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"\n{'='*60}")
    print("  HealthCare AI - Disease Prediction System")
    print(f"{'='*60}")
    print(f"  Server running on: http://localhost:{port}")
    print(f"  Debug mode: {debug}")
    metrics = disease_model.get_model_metrics()
    print(f"  Diseases loaded: {metrics.get('num_diseases', 0)}")
    print(f"  Symptoms tracked: {metrics.get('num_symptoms', 0)}")
    print(f"  Model accuracy: {metrics.get('cross_val_mean', 'N/A')}%")
    print(f"  Model type: {metrics.get('model_type', 'unknown')}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)