"""
Feedback Manager (Feature #18)
Handles prediction feedback storage and analytics
"""

from datetime import datetime


class FeedbackManager:
    """Manages prediction feedback in-memory"""
    
    def __init__(self):
        self.feedbacks = []
    
    def add_feedback(self, prediction_id, rating, was_accurate, comment=""):
        """
        Add feedback for a prediction
        
        Args:
            prediction_id: Unique identifier for the prediction
            rating: Integer 1-5 star rating
            was_accurate: Boolean or None indicating if prediction was accurate
            comment: Optional text comment (max 200 chars)
        
        Returns:
            bool: True if feedback was added successfully
        """
        feedback = {
            'prediction_id': prediction_id,
            'rating': int(rating),
            'was_accurate': was_accurate,
            'comment': str(comment)[:200] if comment else "",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.feedbacks.append(feedback)
        
        # Keep only last 500 feedbacks in memory
        while len(self.feedbacks) > 500:
            self.feedbacks.pop(0)
        
        return True
    
    def get_summary(self):
        """
        Calculate feedback summary statistics
        
        Returns:
            dict: Contains total count, average rating, accuracy rate
        """
        if not self.feedbacks:
            return {
                'total': 0,
                'average_rating': 0,
                'accuracy_rate': 0
            }
        
        total = len(self.feedbacks)
        avg_rating = sum(f['rating'] for f in self.feedbacks) / total
        
        # Calculate accuracy rate (only from feedbacks that have was_accurate set)
        accuracy_feedbacks = [f for f in self.feedbacks if f.get('was_accurate') is not None]
        if accuracy_feedbacks:
            accurate_count = sum(1 for f in accuracy_feedbacks if f['was_accurate'] is True)
            accuracy_rate = round((accurate_count / len(accuracy_feedbacks)) * 100, 1)
        else:
            accuracy_rate = 0
        
        return {
            'total': total,
            'average_rating': round(avg_rating, 1),
            'accuracy_rate': accuracy_rate
        }
    
    def get_recent(self, n=20):
        """
        Get the most recent n feedbacks
        
        Args:
            n: Number of recent feedbacks to return
            
        Returns:
            list: Recent feedback entries
        """
        return self.feedbacks[-n:]