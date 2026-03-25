"""
Vitals Manager - Phase 2 Feature #4
Handles vitals storage, validation, stats, and alerts
"""
from datetime import datetime, timedelta
import uuid


# Normal ranges for alert checks
NORMAL_RANGES = {
    'bp_systolic': {'min': 90, 'max': 120, 'unit': 'mmHg'},
    'bp_diastolic': {'min': 60, 'max': 80, 'unit': 'mmHg'},
    'heart_rate': {'min': 60, 'max': 100, 'unit': 'bpm'},
    'blood_glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
    'spo2': {'min': 95, 'max': 100, 'unit': '%'},
    'temperature': {'min': 97.0, 'max': 99.0, 'unit': '°F'},
    'weight': {'min': 0, 'max': 999, 'unit': 'kg'}  # no alert for weight
}


class VitalsManager:
    """Manages vitals readings - in-memory storage"""

    def __init__(self):
        self.readings = []

    def add_reading(self, vitals_dict):
        """
        Validate and store a vitals reading.
        Returns (success, reading_or_error)
        """
        if not isinstance(vitals_dict, dict):
            return False, "Invalid data format"

        # Build a clean record
        record = {
            'id': str(uuid.uuid4()),
            'timestamp': vitals_dict.get(
                'timestamp',
                datetime.now().isoformat()
            ),
            'recorded_at': datetime.now().isoformat()
        }

        # Accept whichever vitals are provided
        numeric_fields = [
            'bp_systolic', 'bp_diastolic', 'heart_rate',
            'blood_glucose', 'spo2', 'temperature', 'weight'
        ]

        has_at_least_one = False
        for field in numeric_fields:
            val = vitals_dict.get(field)
            if val is not None:
                try:
                    record[field] = float(val)
                    has_at_least_one = True
                except (TypeError, ValueError):
                    pass

        if not has_at_least_one:
            return False, "At least one vital sign is required"

        # Store temp_unit if provided
        record['temp_unit'] = vitals_dict.get('temp_unit', 'F')
        record['weight_unit'] = vitals_dict.get('weight_unit', 'kg')

        # Convert temperature to °F for storage if °C provided
        if record.get('temperature') and record['temp_unit'] == 'C':
            record['temperature'] = round(
                record['temperature'] * 9 / 5 + 32, 1
            )
            record['temp_unit'] = 'F'

        # Convert weight to kg if lbs provided
        if record.get('weight') and record['weight_unit'] == 'lbs':
            record['weight'] = round(record['weight'] * 0.453592, 2)
            record['weight_unit'] = 'kg'

        # Get alerts
        record['alerts'] = self.check_alerts(record)

        self.readings.append(record)

        # Keep only last 500 readings
        while len(self.readings) > 500:
            self.readings.pop(0)

        return True, record

    def get_readings(self, days=30):
        """Return readings from last N days, sorted newest first."""
        cutoff = datetime.now() - timedelta(days=days)
        result = []
        for r in self.readings:
            try:
                ts = datetime.fromisoformat(r['timestamp'])
                if ts >= cutoff:
                    result.append(r)
            except Exception:
                result.append(r)

        return sorted(result, key=lambda x: x.get('timestamp', ''), reverse=True)

    def get_reading_by_id(self, reading_id):
        """Return a single reading by ID."""
        for r in self.readings:
            if r.get('id') == reading_id:
                return r
        return None

    def delete_reading(self, reading_id):
        """Remove a reading by ID. Returns True if found and deleted."""
        original_len = len(self.readings)
        self.readings = [r for r in self.readings if r.get('id') != reading_id]
        return len(self.readings) < original_len

    def get_stats(self, metric, days=30):
        """
        Returns min, max, avg for a given metric over last N days.
        metric: one of the numeric fields (e.g. 'heart_rate')
        """
        readings = self.get_readings(days)
        values = [
            r[metric] for r in readings
            if metric in r and r[metric] is not None
        ]

        if not values:
            return {'min': None, 'max': None, 'avg': None, 'count': 0}

        return {
            'min': round(min(values), 1),
            'max': round(max(values), 1),
            'avg': round(sum(values) / len(values), 1),
            'count': len(values)
        }

    def get_all_stats(self, days=30):
        """Returns stats for all metrics."""
        metrics = [
            'bp_systolic', 'bp_diastolic', 'heart_rate',
            'blood_glucose', 'spo2', 'temperature', 'weight'
        ]
        return {m: self.get_stats(m, days) for m in metrics}

    def check_alerts(self, vitals_dict):
        """
        Compare vitals against normal ranges.
        Returns list of alert dicts with status: 'normal', 'warning', 'danger'
        """
        alerts = []

        checks = {
            'bp_systolic': vitals_dict.get('bp_systolic'),
            'bp_diastolic': vitals_dict.get('bp_diastolic'),
            'heart_rate': vitals_dict.get('heart_rate'),
            'blood_glucose': vitals_dict.get('blood_glucose'),
            'spo2': vitals_dict.get('spo2'),
            'temperature': vitals_dict.get('temperature'),
        }

        label_map = {
            'bp_systolic': 'Blood Pressure (Systolic)',
            'bp_diastolic': 'Blood Pressure (Diastolic)',
            'heart_rate': 'Heart Rate',
            'blood_glucose': 'Blood Glucose',
            'spo2': 'SpO2',
            'temperature': 'Body Temperature',
        }

        for key, value in checks.items():
            if value is None:
                continue

            normal = NORMAL_RANGES[key]
            unit = normal['unit']
            low = normal['min']
            high = normal['max']

            # Calculate how far outside range
            if value < low:
                deviation = (low - value) / low * 100
                if deviation > 15:
                    status = 'danger'
                else:
                    status = 'warning'
                message = f"{label_map[key]} is low: {value} {unit} (normal: {low}–{high})"
            elif value > high:
                deviation = (value - high) / high * 100
                if deviation > 20:
                    status = 'danger'
                else:
                    status = 'warning'
                message = f"{label_map[key]} is high: {value} {unit} (normal: {low}–{high})"
            else:
                status = 'normal'
                message = f"{label_map[key]} is normal: {value} {unit}"

            alerts.append({
                'metric': key,
                'label': label_map[key],
                'value': value,
                'unit': unit,
                'status': status,
                'message': message,
                'normal_range': f"{low}–{high} {unit}"
            })

        return alerts