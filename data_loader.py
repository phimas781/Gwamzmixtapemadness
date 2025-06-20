import joblib
import numpy as np

# Lightweight model loader
def load_model(model_path):
    try:
        # Try direct load first
        return joblib.load(model_path)
    except Exception:
        try:
            # Fallback to numpy-based load
            import numpy as np
            with open(model_path, 'rb') as f:
                return np.load(f, allow_pickle=True).item()
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")