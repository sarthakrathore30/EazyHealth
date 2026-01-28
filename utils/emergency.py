"""
Emergency detection and risk assessment utilities
"""
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EmergencyLevel(Enum):
    """Emergency severity levels"""
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    LIFE_THREATENING = 5


@dataclass
class EmergencyAssessment:
    """Container for emergency assessment results"""
    level: EmergencyLevel
    is_emergency: bool
    message: str
    matched_symptoms: List[str]
    recommendations: List[str]
    call_emergency: bool


class EmergencyDetector:
    """
    Sophisticated emergency detection based on:
    - Symptom combinations
    - Duration
    - Severity indicators
    - Risk factors
    """
    
    # Critical symptom combinations that indicate emergencies
    CRITICAL_COMBINATIONS = [
        {
            'symptoms': ['severe chest pain', 'pain radiating to arm'],
            'condition': 'Possible Heart Attack',
            'level': EmergencyLevel.LIFE_THREATENING,
            'action': 'Call 911 immediately. Chew aspirin if available and not allergic.'
        },
        {
            'symptoms': ['chest pain', 'shortness of breath', 'cold sweat'],
            'condition': 'Possible Heart Attack',
            'level': EmergencyLevel.LIFE_THREATENING,
            'action': 'Call 911 immediately.'
        },
        {
            'symptoms': ['sudden numbness face or limbs', 'trouble speaking'],
            'condition': 'Possible Stroke',
            'level': EmergencyLevel.LIFE_THREATENING,
            'action': 'Call 911 immediately. Note the time symptoms started.'
        },
        {
            'symptoms': ['facial drooping', 'arm weakness', 'speech difficulty'],
            'condition': 'Possible Stroke (FAST)',
            'level': EmergencyLevel.LIFE_THREATENING,
            'action': 'Call 911 immediately. Time is critical.'
        },
        {
            'symptoms': ['severe abdominal pain right side', 'fever', 'vomiting'],
            'condition': 'Possible Appendicitis',
            'level': EmergencyLevel.CRITICAL,
            'action': 'Go to emergency room immediately. Do not eat or drink.'
        },
        {
            'symptoms': ['stiff neck', 'high fever', 'severe headache'],
            'condition': 'Possible Meningitis',
            'level': EmergencyLevel.LIFE_THREATENING,
            'action': 'Seek emergency care immediately.'
        },
        {
            'symptoms': ['difficulty breathing', 'bluish lips'],
            'condition': 'Severe Respiratory Distress',
            'level': EmergencyLevel.LIFE_THREATENING,
            'action': 'Call 911 immediately.'
        },
        {
            'symptoms': ['severe allergic reaction', 'difficulty breathing', 'swelling'],
            'condition': 'Possible Anaphylaxis',
            'level': EmergencyLevel.LIFE_THREATENING,
            'action': 'Use EpiPen if available. Call 911 immediately.'
        },
        {
            'symptoms': ['coughing up blood', 'chest pain', 'shortness of breath'],
            'condition': 'Possible Pulmonary Emergency',
            'level': EmergencyLevel.CRITICAL,
            'action': 'Seek emergency care immediately.'
        },
        {
            'symptoms': ['thoughts of death', 'hopelessness'],
            'condition': 'Mental Health Crisis',
            'level': EmergencyLevel.CRITICAL,
            'action': 'Call 988 (Suicide & Crisis Lifeline) or go to emergency room.'
        },
        {
            'symptoms': ['seizures', 'loss of consciousness'],
            'condition': 'Neurological Emergency',
            'level': EmergencyLevel.CRITICAL,
            'action': 'Call 911. Keep person safe, do not restrain.'
        }
    ]
    
    # Individual high-risk symptoms
    HIGH_RISK_SYMPTOMS = {
        'severe chest pain': EmergencyLevel.CRITICAL,
        'chest pressure': EmergencyLevel.HIGH,
        'difficulty breathing': EmergencyLevel.CRITICAL,
        'sudden numbness face or limbs': EmergencyLevel.CRITICAL,
        'trouble speaking': EmergencyLevel.CRITICAL,
        'loss of consciousness': EmergencyLevel.LIFE_THREATENING,
        'coughing up blood': EmergencyLevel.CRITICAL,
        'sudden severe headache': EmergencyLevel.HIGH,
        'seizures': EmergencyLevel.CRITICAL,
        'facial drooping': EmergencyLevel.LIFE_THREATENING,
        'thoughts of death': EmergencyLevel.HIGH,
        'severe allergic reaction': EmergencyLevel.LIFE_THREATENING,
        'inability to breathe': EmergencyLevel.LIFE_THREATENING,
        'severe bleeding': EmergencyLevel.CRITICAL,
        'bluish lips': EmergencyLevel.LIFE_THREATENING,
        'pain radiating to arm': EmergencyLevel.HIGH
    }
    
    # Duration-based risk escalation
    DURATION_RISK_MULTIPLIER = {
        'less_than_24_hours': 1.0,
        '1_to_3_days': 1.1,
        '3_to_7_days': 1.2,
        '1_to_2_weeks': 1.3,
        'more_than_2_weeks': 1.4
    }
    
    def __init__(self, symptoms_data_path: str = 'data/symptoms.json'):
        """Initialize with symptoms data"""
        try:
            with open(symptoms_data_path, 'r') as f:
                self.symptoms_data = json.load(f)
            self.emergency_symptoms = set(self.symptoms_data.get('emergency_symptoms', []))
        except Exception:
            self.emergency_symptoms = set()
            self.symptoms_data = {}
    
    def assess_emergency(
        self,
        symptoms: List[str],
        duration: str = '1_to_3_days',
        age: Optional[int] = None
    ) -> EmergencyAssessment:
        """
        Perform comprehensive emergency assessment
        
        Args:
            symptoms: List of reported symptoms
            duration: How long symptoms have been present
            age: Patient age (optional, for risk adjustment)
        
        Returns:
            EmergencyAssessment with detailed evaluation
        """
        symptoms_lower = [s.lower() for s in symptoms]
        symptoms_set = set(symptoms_lower)
        
        # Check for critical combinations first
        for combo in self.CRITICAL_COMBINATIONS:
            combo_symptoms = set(combo['symptoms'])
            if combo_symptoms.issubset(symptoms_set):
                return EmergencyAssessment(
                    level=combo['level'],
                    is_emergency=True,
                    message=f"⚠️ EMERGENCY: {combo['condition']} suspected based on your symptoms.",
                    matched_symptoms=list(combo_symptoms),
                    recommendations=[combo['action']],
                    call_emergency=combo['level'].value >= EmergencyLevel.CRITICAL.value
                )
        
        # Check individual high-risk symptoms
        matched_high_risk = []
        highest_level = EmergencyLevel.NONE
        
        for symptom in symptoms_lower:
            if symptom in self.HIGH_RISK_SYMPTOMS:
                matched_high_risk.append(symptom)
                level = self.HIGH_RISK_SYMPTOMS[symptom]
                if level.value > highest_level.value:
                    highest_level = level
        
        # Check emergency symptoms from data
        matched_emergency = [s for s in symptoms_lower if s in self.emergency_symptoms]
        
        # Combine matches
        all_concerning = list(set(matched_high_risk + matched_emergency))
        
        # Adjust for duration
        duration_multiplier = self.DURATION_RISK_MULTIPLIER.get(duration, 1.0)
        
        # Adjust for age (higher risk for very young or elderly)
        age_multiplier = 1.0
        if age is not None:
            if age < 5 or age > 65:
                age_multiplier = 1.3
            elif age < 12 or age > 50:
                age_multiplier = 1.1
        
        # Calculate adjusted level
        if highest_level.value > 0:
            adjusted_value = min(5, int(highest_level.value * duration_multiplier * age_multiplier))
            highest_level = EmergencyLevel(adjusted_value)
        
        # Build response
        if highest_level.value >= EmergencyLevel.LIFE_THREATENING.value:
            return EmergencyAssessment(
                level=highest_level,
                is_emergency=True,
                message="🚨 LIFE-THREATENING EMERGENCY: Your symptoms indicate a potentially life-threatening condition.",
                matched_symptoms=all_concerning,
                recommendations=[
                    "Call 911 (US) or your local emergency number immediately",
                    "Do not drive yourself to the hospital",
                    "Stay calm and wait for emergency services"
                ],
                call_emergency=True
            )
        elif highest_level.value >= EmergencyLevel.CRITICAL.value:
            return EmergencyAssessment(
                level=highest_level,
                is_emergency=True,
                message="⚠️ URGENT: Your symptoms require immediate medical attention.",
                matched_symptoms=all_concerning,
                recommendations=[
                    "Go to the emergency room immediately",
                    "Have someone drive you if possible",
                    "Call 911 if symptoms worsen"
                ],
                call_emergency=True
            )
        elif highest_level.value >= EmergencyLevel.HIGH.value:
            return EmergencyAssessment(
                level=highest_level,
                is_emergency=True,
                message="⚠️ WARNING: Your symptoms may require urgent medical care.",
                matched_symptoms=all_concerning,
                recommendations=[
                    "Seek medical attention within the next few hours",
                    "Consider visiting urgent care or emergency room",
                    "Monitor symptoms closely for any worsening"
                ],
                call_emergency=False
            )
        elif highest_level.value >= EmergencyLevel.MODERATE.value:
            return EmergencyAssessment(
                level=highest_level,
                is_emergency=False,
                message="ℹ️ ATTENTION: Some of your symptoms warrant medical attention.",
                matched_symptoms=all_concerning,
                recommendations=[
                    "Schedule an appointment with your doctor soon",
                    "Consider urgent care if symptoms worsen",
                    "Keep track of your symptoms"
                ],
                call_emergency=False
            )
        else:
            return EmergencyAssessment(
                level=EmergencyLevel.NONE,
                is_emergency=False,
                message="",
                matched_symptoms=[],
                recommendations=[],
                call_emergency=False
            )
    
    def get_emergency_numbers(self) -> Dict[str, str]:
        """Get emergency phone numbers by region"""
        return {
            'US': '911',
            'UK': '999',
            'EU': '112',
            'India': '112',
            'Australia': '000',
            'Canada': '911',
            'Mental Health (US)': '988'
        }


def check_symptom_severity(symptoms: List[str], symptom_weights: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Calculate overall symptom severity score
    
    Returns:
        Tuple of (severity_score, critical_symptoms)
    """
    total_weight = 0.0
    critical_symptoms = []
    
    for symptom in symptoms:
        weight = symptom_weights.get(symptom.lower(), 0.5)
        total_weight += weight
        
        if weight >= 0.9:
            critical_symptoms.append(symptom)
    
    # Normalize score
    avg_severity = total_weight / len(symptoms) if symptoms else 0
    
    return avg_severity, critical_symptoms