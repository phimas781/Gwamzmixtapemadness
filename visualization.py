import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_feature_impact(feature_impacts):
    """Create a feature impact visualization"""
    df = pd.DataFrame({
        'Feature': list(feature_impacts.keys()),
        'Impact': list(feature_impacts.values())
    }).sort_values('Impact', ascending=True)
    
    fig = px.bar(
        df, 
        x='Impact', 
        y='Feature', 
        orientation='h',
        color='Impact',
        color_continuous_scale='Blues',
        title='Feature Impact on Stream Prediction'
    )
    
    fig.update_layout(
        xaxis_title='Relative Impact Score',
        yaxis_title='',
        coloraxis_showscale=False,
        height=400
    )
    
    return fig

def plot_historical_trends(df):
    """Create historical trends visualization"""
    # Prepare data
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d/%m/%Y')
    df['month'] = df['release_date'].dt.month_name()
    df['year'] = df['release_date'].dt.year
    
    # Group by month and year
    monthly = df.groupby(['year', 'month'])['Number Of Strems'].mean().reset_index()
    
    # Create plot
    fig = px.line(
        monthly, 
        x='month', 
        y='Number Of Strems', 
        color='year',
        markers=True,
        title='Average Streams by Release Month',
        labels={'Number Of Strems': 'Average Streams', 'month': 'Release Month'}
    )
    
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]},
        height=500
    )
    
    return fig