"""
Utilities Package
"""
from .sanitizer import InputSanitizer, RequestValidator
from .emergency import EmergencyDetector, EmergencyAssessment, EmergencyLevel

# Import feedback manager
try:
    from .feedback import FeedbackManager
    _has_feedback = True
except ImportError:
    _has_feedback = False

# Import vitals manager
try:
    from .vitals import VitalsManager
    _has_vitals = True
except ImportError:
    _has_vitals = False

if _has_feedback and _has_vitals:
    __all__ = [
        'InputSanitizer',
        'RequestValidator',
        'EmergencyDetector',
        'EmergencyAssessment',
        'EmergencyLevel',
        'FeedbackManager',
        'VitalsManager'
    ]
elif _has_feedback:
    __all__ = [
        'InputSanitizer',
        'RequestValidator',
        'EmergencyDetector',
        'EmergencyAssessment',
        'EmergencyLevel',
        'FeedbackManager'
    ]
elif _has_vitals:
    __all__ = [
        'InputSanitizer',
        'RequestValidator',
        'EmergencyDetector',
        'EmergencyAssessment',
        'EmergencyLevel',
        'VitalsManager'
    ]
else:
    __all__ = [
        'InputSanitizer',
        'RequestValidator',
        'EmergencyDetector',
        'EmergencyAssessment',
        'EmergencyLevel'
    ]