"""
Input sanitization and validation utilities
"""
import re
import html
from typing import Any, List, Optional, Union

# Try to import bleach, but have fallback
try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    BLEACH_AVAILABLE = False
    print("bleach not available - using basic HTML escaping")


class InputSanitizer:
    """Handles all input sanitization to prevent XSS and injection attacks"""
    
    # Allowed HTML tags (minimal for safety)
    ALLOWED_TAGS = []
    ALLOWED_ATTRIBUTES = {}
    
    # Maximum input lengths
    MAX_SYMPTOM_LENGTH = 100
    MAX_MESSAGE_LENGTH = 1000
    MAX_SYMPTOMS_COUNT = 20
    
    # Valid input patterns
    SYMPTOM_PATTERN = re.compile(r'^[a-zA-Z\s\-\']+$')
    NUMBER_PATTERN = re.compile(r'^\d+\.?\d*$')
    
    @classmethod
    def sanitize_string(cls, text: str) -> str:
        """
        Sanitize a string input by removing HTML and dangerous characters
        """
        if not isinstance(text, str):
            return ""
        
        # Strip whitespace
        text = text.strip()
        
        # Remove HTML tags using bleach if available
        if BLEACH_AVAILABLE:
            text = bleach.clean(text, tags=cls.ALLOWED_TAGS, attributes=cls.ALLOWED_ATTRIBUTES, strip=True)
        else:
            # Fallback: remove all HTML tags using regex
            text = re.sub(r'<[^>]+>', '', text)
        
        # Escape any remaining HTML entities
        text = html.escape(text)
        
        # Remove null bytes and other control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text
    
    @classmethod
    def sanitize_symptom(cls, symptom: str) -> Optional[str]:
        """
        Sanitize and validate a symptom input
        Returns None if invalid
        """
        if not isinstance(symptom, str):
            return None
        
        # Basic sanitization
        symptom = symptom.strip().lower()
        
        # Check length
        if len(symptom) > cls.MAX_SYMPTOM_LENGTH or len(symptom) < 2:
            return None
        
        # Remove dangerous characters but allow hyphens and apostrophes
        symptom = re.sub(r'[<>"\'\\/;`]', '', symptom)
        
        # Validate pattern (letters, spaces, hyphens, apostrophes only)
        if not cls.SYMPTOM_PATTERN.match(symptom):
            return None
        
        return symptom
    
    @classmethod
    def sanitize_symptoms_list(cls, symptoms: List[str], valid_symptoms: set) -> List[str]:
        """
        Sanitize and validate a list of symptoms
        Only returns symptoms that exist in the valid symptoms set
        """
        if not isinstance(symptoms, list):
            return []
        
        # Limit number of symptoms
        if len(symptoms) > cls.MAX_SYMPTOMS_COUNT:
            symptoms = symptoms[:cls.MAX_SYMPTOMS_COUNT]
        
        sanitized = []
        for symptom in symptoms:
            clean_symptom = cls.sanitize_symptom(symptom)
            if clean_symptom and clean_symptom in valid_symptoms:
                sanitized.append(clean_symptom)
        
        return list(set(sanitized))  # Remove duplicates
    
    @classmethod
    def sanitize_number(cls, value: Any, min_val: float = None, max_val: float = None) -> Optional[float]:
        """
        Sanitize and validate a numeric input
        """
        try:
            if isinstance(value, str):
                value = value.strip()
                if not cls.NUMBER_PATTERN.match(value):
                    return None
            
            num = float(value)
            
            # Check range
            if min_val is not None and num < min_val:
                return None
            if max_val is not None and num > max_val:
                return None
            
            return num
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def sanitize_integer(cls, value: Any, min_val: int = None, max_val: int = None) -> Optional[int]:
        """
        Sanitize and validate an integer input
        """
        num = cls.sanitize_number(value, min_val, max_val)
        if num is not None:
            return int(num)
        return None
    
    @classmethod
    def sanitize_chat_message(cls, message: str) -> Optional[str]:
        """
        Sanitize a chat message
        """
        if not isinstance(message, str):
            return None
        
        message = cls.sanitize_string(message)
        
        if len(message) > cls.MAX_MESSAGE_LENGTH or len(message) < 1:
            return None
        
        return message
    
    @classmethod
    def sanitize_gender(cls, gender: str) -> Optional[str]:
        """
        Validate gender input
        """
        if not isinstance(gender, str):
            return None
        
        gender = gender.strip().lower()
        
        if gender in ['male', 'female', 'other']:
            return gender
        return None
    
    @classmethod
    def sanitize_duration(cls, duration: str) -> Optional[str]:
        """
        Validate symptom duration input
        """
        valid_durations = [
            'less_than_24_hours',
            '1_to_3_days',
            '3_to_7_days',
            '1_to_2_weeks',
            'more_than_2_weeks'
        ]
        
        if not isinstance(duration, str):
            return None
        
        duration = duration.strip().lower()
        
        if duration in valid_durations:
            return duration
        return None


class RequestValidator:
    """Validates complete API requests"""
    
    @staticmethod
    def validate_prediction_request(data: dict, valid_symptoms: set) -> tuple:
        """
        Validate a disease prediction request
        Returns (is_valid, sanitized_data or error_message)
        """
        if not isinstance(data, dict):
            return False, "Invalid request format"
        
        # Validate symptoms
        symptoms = data.get('symptoms', [])
        if not symptoms:
            return False, "No symptoms provided"
        
        sanitized_symptoms = InputSanitizer.sanitize_symptoms_list(symptoms, valid_symptoms)
        if not sanitized_symptoms:
            return False, "No valid symptoms provided"
        
        # Validate optional fields
        duration = data.get('duration', '1_to_3_days')
        sanitized_duration = InputSanitizer.sanitize_duration(duration) or '1_to_3_days'
        
        age = data.get('age')
        sanitized_age = InputSanitizer.sanitize_integer(age, 0, 120) if age else None
        
        return True, {
            'symptoms': sanitized_symptoms,
            'duration': sanitized_duration,
            'age': sanitized_age
        }
    
    @staticmethod
    def validate_bmi_request(data: dict) -> tuple:
        """
        Validate a BMI calculation request
        Returns (is_valid, sanitized_data or error_message)
        """
        if not isinstance(data, dict):
            return False, "Invalid request format"
        
        weight = InputSanitizer.sanitize_number(data.get('weight'), 20, 500)
        height = InputSanitizer.sanitize_number(data.get('height'), 50, 300)
        age = InputSanitizer.sanitize_integer(data.get('age'), 1, 120)
        gender = InputSanitizer.sanitize_gender(data.get('gender'))
        
        if weight is None:
            return False, "Invalid weight value (must be 20-500 kg)"
        if height is None:
            return False, "Invalid height value (must be 50-300 cm)"
        if age is None:
            return False, "Invalid age value (must be 1-120)"
        if gender is None:
            return False, "Invalid gender value"
        
        activity = InputSanitizer.sanitize_number(data.get('activity', 1.2), 1.0, 2.5) or 1.2
        
        return True, {
            'weight': weight,
            'height': height,
            'age': age,
            'gender': gender,
            'activity': activity
        }
    
    @staticmethod
    def validate_chat_request(data: dict) -> tuple:
        """
        Validate a chat message request
        Returns (is_valid, sanitized_data or error_message)
        """
        if not isinstance(data, dict):
            return False, "Invalid request format"
        
        message = InputSanitizer.sanitize_chat_message(data.get('message', ''))
        
        if not message:
            return False, "Message is required and must be less than 1000 characters"
        
        return True, {'message': message}