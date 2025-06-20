# compatibility_check.py
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn
import joblib
import plotly
import pyarrow

st.title("Dependency Compatibility Report")

versions = {
    "Streamlit": st.__version__,
    "Pandas": pd.__version__,
    "NumPy": np.__version__,
    "Scikit-learn": sklearn.__version__,
    "Matplotlib": matplotlib.__version__,
    "Seaborn": seaborn.__version__,
    "Joblib": joblib.__version__,
    "Plotly": plotly.__version__,
    "PyArrow": pyarrow.__version__
}

st.table(pd.DataFrame(list(versions.items()), columns=["Package", "Version"])

# Check critical compatibilities
if pd.__version__.startswith('2.0.') and np.__version__.startswith('1.24.'):
    st.success("✅ Pandas and NumPy versions are compatible")
else:
    st.error("❌ Pandas and NumPy versions are incompatible")

# Test model loading
try:
    import joblib
    joblib.load('gwamz_model.pkl')
    st.success("✅ Model file loads successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {str(e)}")