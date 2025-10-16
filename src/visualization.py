import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

def create_risk_heatmap(predictions_df: pd.DataFrame):
    """Create risk level heatmap"""
    color_map = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
    
    fig = go.Figure(data=go.Table(
        header=dict(values=['Region', 'Risk Level', 'Probability', 'Status'],
                   fill_color='lightblue',
                   align='left'),
        cells=dict(values=[
            predictions_df['region'],
            predictions_df['risk_level'],
            [f"{p:.1%}" for p in predictions_df['poverty_risk_probability']],
            ['🔴' if r == 'HIGH' else '🟡' if r == 'MEDIUM' else '🟢' 
             for r in predictions_df['risk_level']]
        ],
        fill_color=[['white' if r == 'LOW' else 'lightyellow' if r == 'MEDIUM' else 'lightcoral' 
                    for r in predictions_df['risk_level']] * 4],
        align='left')
    ))
    
    fig.update_layout(title="Regional Poverty Risk Assessment", height=600)
    return fig

def create_time_series_chart(df: pd.DataFrame, region: str, indicator: str):
    """Create time series chart for specific indicator"""
    region_data = df[df['region'] == region].sort_values('date')
    
    fig = px.line(region_data, x='date', y=indicator, 
                  title=f"{indicator.replace('_', ' ').title()} - {region}")
    fig.update_layout(xaxis_title="Date", yaxis_title=indicator.replace('_', ' ').title())
    return fig

def create_confusion_matrix(cm: list):
    """Create confusion matrix visualization"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Spike', 'Predicted Spike'],
        y=['Actual No Spike', 'Actual Spike'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def create_metrics_gauge(accuracy: float, title: str = "Model Accuracy"):
    """Create gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = accuracy * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_indicator_dashboard(df: pd.DataFrame):
    """Create multi-indicator dashboard"""
    indicators = ['mobile_money_volume', 'food_price_index', 'unemployment_estimate', 'rainfall']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[ind.replace('_', ' ').title() for ind in indicators]
    )
    
    for i, indicator in enumerate(indicators):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Aggregate by date across all regions
        agg_data = df.groupby('date')[indicator].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(x=agg_data['date'], y=agg_data[indicator], 
                      name=indicator.replace('_', ' ').title()),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title="Key Economic Indicators Over Time")
    return fig

def create_risk_summary_cards(predictions_df: pd.DataFrame):
    """Create summary statistics"""
    high_risk = len(predictions_df[predictions_df['risk_level'] == 'HIGH'])
    medium_risk = len(predictions_df[predictions_df['risk_level'] == 'MEDIUM'])
    low_risk = len(predictions_df[predictions_df['risk_level'] == 'LOW'])
    
    return {
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'total_regions': len(predictions_df)
    }
