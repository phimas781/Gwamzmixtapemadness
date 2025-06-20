import pandas as pd
import joblib

def load_data(file_path):
    """Load historical song data from CSV"""
    return pd.read_csv(file_path)

def load_model(model_path):
    """Load trained model from pickle file"""
    return joblib.load(model_path)