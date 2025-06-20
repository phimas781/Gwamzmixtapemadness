# utils/data_loader.py
import joblib
import pandas as pd

# Global cached model
_MODEL = None

def load_model(model_path):
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
    return _MODEL

def load_data(file_path):
    """Load only essential columns to reduce memory"""
    cols = [
        'release_date', 'release_year', 'total_tracks_in_album',
        'explicit', 'track_name', 'Number Of Strems'
    ]
    return pd.read_csv(file_path, usecols=cols)