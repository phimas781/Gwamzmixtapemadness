import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PerformanceMonitor:
    def __init__(self, model_path, log_file="performance_log.csv"):
        self.model = joblib.load(model_path)
        self.log_file = log_file
        
        if os.path.exists(log_file):
            self.performance_log = pd.read_csv(log_file)
        else:
            self.performance_log = pd.DataFrame(columns=[
                'timestamp', 'actual', 'predicted', 'error', 'error_pct', 'features'
            ])
    
    def log_performance(self, input_features, actual_streams):
        """Log actual vs predicted performance for new releases"""
        # Make prediction
        input_df = pd.DataFrame([input_features])
        prediction = self.model.predict(input_df)[0]
        
        # Calculate error
        error = abs(actual_streams - prediction)
        error_pct = (error / actual_streams) * 100
        
        # Create new entry
        new_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'actual': actual_streams,
            'predicted': prediction,
            'error': error,
            'error_pct': error_pct,
            'features': str(input_features)
        }
        
        # Append to log
        self.performance_log = pd.concat(
            [self.performance_log, pd.DataFrame([new_entry])], 
            ignore_index=True
        )
        
        # Save log
        self.save_log()
        
        return {
            'prediction': prediction,
            'error': error,
            'error_pct': error_pct
        }
    
    def generate_report(self):
        """Generate model performance report"""
        if self.performance_log.empty:
            return None
            
        report = {
            'total_predictions': len(self.performance_log),
            'mean_error': self.performance_log['error'].mean(),
            'median_error': self.performance_log['error'].median(),
            'mean_error_pct': self.performance_log['error_pct'].mean(),
            'accuracy_90_pct': (self.performance_log['error_pct'] < 10).mean(),
            'recent_performance': self.performance_log.tail(5)['error_pct'].mean()
        }
        return report
    
    def save_log(self):
        """Save performance log to CSV"""
        self.performance_log.to_csv(self.log_file, index=False)
    
    def plot_performance_trend(self):
        """Plot performance trend over time"""
        if self.performance_log.empty:
            return None
            
        plt.figure(figsize=(10, 6))
        self.performance_log['timestamp'] = pd.to_datetime(self.performance_log['timestamp'])
        self.performance_log.set_index('timestamp', inplace=True)
        self.performance_log['error_pct'].plot()
        plt.title('Prediction Error Over Time')
        plt.ylabel('Error Percentage')
        plt.xlabel('Date')
        plt.grid(True)
        return plt

if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor('gwamz_model.pkl')
    
    # Sample input features
    sample_features = {
        'release_year': 2025,
        'release_month': 6,
        'release_quarter': 2,
        'release_dayofweek': 3,
        'total_tracks_in_album': 3,
        'explicit': 1,
        'is_single': 1,
        'is_remix': 0,
        'is_sped_up': 1,
        'is_jersey': 0,
        'title_length': 15,
        'days_since_last_release': 90,
        'release_sequence': 15,
        'artist_popularity': 45,
        'version_original': 0,
        'version_sped_up': 1,
        'version_remix': 0,
        'version_edit': 0,
        'version_jersey_club': 0,
        'version_instrumental': 0
    }
    
    # Log a sample prediction
    result = monitor.log_performance(sample_features, 1500000)
    print(f"Logged prediction: {result}")
    
    # Generate report
    report = monitor.generate_report()
    print("\nPerformance Report:")
    for k, v in report.items():
        print(f"{k.replace('_', ' ').title()}: {v}")