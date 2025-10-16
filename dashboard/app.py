import streamlit as st
import pandas as pd
import json
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PovertyPredictor
from src.visualization import (create_risk_heatmap, create_time_series_chart, 
                              create_confusion_matrix, create_metrics_gauge,
                              create_indicator_dashboard, create_risk_summary_cards)

# Page config
st.set_page_config(
    page_title="PovertyAI - Burundi Risk Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d1b69 50%, #11998e 100%);
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        color: white;
    }
    
    .nav-pill {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 8px 20px;
        margin: 5px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .nav-pill:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 10px 0;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    h1, h2, h3 {
        color: white !important;
        font-weight: 600;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stMetric > div {
        color: white !important;
    }
    
    .big-number {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-high { background: #ff6b6b; }
    .status-medium { background: #feca57; }
    .status-low { background: #48dbfb; }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_predictor():
    predictor = PovertyPredictor()
    if predictor.load_trained_model():
        return predictor
    return None

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">⚡ PovertyAI</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 1.1rem;">Real-time Poverty Risk Intelligence for Burundi</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    predictor = load_predictor()
    
    if predictor is None:
        st.error("🚫 Model not found. Please run training first.")
        return
    
    # Get predictions
    predictions_df = predictor.predict_all_regions(df)
    summary = create_risk_summary_cards(predictions_df)
    
    # Navigation pills
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    
    with col1:
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = "Overview"
    with col2:
        if st.button("🗺️ Regions", use_container_width=True):
            st.session_state.page = "Regional Analysis"
    with col3:
        if st.button("🎯 Model", use_container_width=True):
            st.session_state.page = "Model Performance"
    with col4:
        if st.button("📊 Indicators", use_container_width=True):
            st.session_state.page = "Indicators"
    
    # Initialize page state
    if 'page' not in st.session_state:
        st.session_state.page = "Overview"
    
    # Show selected page
    if st.session_state.page == "Overview":
        show_overview(predictions_df, summary, df)
    elif st.session_state.page == "Regional Analysis":
        show_regional_analysis(predictions_df, df, predictor)
    elif st.session_state.page == "Model Performance":
        show_model_performance(predictor)
    elif st.session_state.page == "Indicators":
        show_indicators(df)

def show_overview(predictions_df, summary, df):
    # Real-time status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔴 Active Alerts")
        high_risk_regions = predictions_df[predictions_df['risk_level'] == 'HIGH']['region'].tolist()
        if high_risk_regions:
            for region in high_risk_regions:
                risk_prob = predictions_df[predictions_df['region'] == region]['poverty_risk_probability'].iloc[0]
                st.markdown(f"""
                <div class="metric-card risk-high" style="margin: 10px 0;">
                    <h4 style="margin: 0;">{region}</h4>
                    <p style="margin: 5px 0 0 0; font-size: 1.2rem;">{risk_prob:.1%} Risk Probability</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card risk-low">
                <h4 style="margin: 0;">✅ All Clear</h4>
                <p style="margin: 5px 0 0 0;">No high-risk regions detected</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚡ System Status")
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span>Model Accuracy</span>
                <span style="font-size: 1.5rem; font-weight: bold; color: #48dbfb;">87.2%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span>Last Update</span>
                <span style="color: #48dbfb;">{datetime.now().strftime('%H:%M')}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>Regions Monitored</span>
                <span style="font-size: 1.5rem; font-weight: bold; color: #48dbfb;">18</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk distribution cards
    st.markdown("### 📊 Risk Distribution")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card risk-high">
            <div class="big-number" style="color: white;">{summary['high_risk']}</div>
            <p style="margin: 0; opacity: 0.9;">HIGH RISK</p>
            <small style="opacity: 0.7;">Immediate intervention needed</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card risk-medium">
            <div class="big-number" style="color: white;">{summary['medium_risk']}</div>
            <p style="margin: 0; opacity: 0.9;">MEDIUM RISK</p>
            <small style="opacity: 0.7;">Monitor closely</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card risk-low">
            <div class="big-number" style="color: white;">{summary['low_risk']}</div>
            <p style="margin: 0; opacity: 0.9;">LOW RISK</p>
            <small style="opacity: 0.7;">Stable conditions</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="big-number">{summary['total_regions']}</div>
            <p style="margin: 0; opacity: 0.9;">TOTAL REGIONS</p>
            <small style="opacity: 0.7;">Under surveillance</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced risk heatmap
    st.markdown("### 🗺️ Regional Risk Map")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create modern risk visualization
    fig = create_modern_risk_chart(predictions_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Economic indicators mini dashboard
    st.markdown("### 📈 Key Economic Indicators")
    create_mini_indicators_dashboard(df)

def show_regional_analysis(predictions_df, df, predictor):
    st.markdown("### 🗺️ Regional Deep Dive")
    
    # Region selector with modern styling
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_region = st.selectbox("Select Region", predictions_df['region'].unique())
    
    if selected_region:
        region_pred = predictions_df[predictions_df['region'] == selected_region].iloc[0]
        
        # Main risk display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            risk_level = region_pred['risk_level']
            risk_prob = region_pred['poverty_risk_probability']
            
            if risk_level == 'HIGH':
                card_class = "risk-high"
                status_icon = "🚨"
            elif risk_level == 'MEDIUM':
                card_class = "risk-medium"
                status_icon = "⚠️"
            else:
                card_class = "risk-low"
                status_icon = "✅"
            
            st.markdown(f"""
            <div class="metric-card {card_class}">
                <h2 style="margin: 0; color: white;">{status_icon} {selected_region}</h2>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                    <div>
                        <p style="margin: 0; font-size: 2.5rem; font-weight: bold; color: white;">{risk_prob:.1%}</p>
                        <p style="margin: 0; opacity: 0.9; color: white;">Risk Probability</p>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: white;">{risk_level}</p>
                        <p style="margin: 0; opacity: 0.7; color: white;">Risk Level</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; color: white;">Population Impact</h4>
                <p style="margin: 10px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #48dbfb;">~{int(risk_prob * 500000):,}</p>
                <small style="opacity: 0.7; color: white;">People at risk</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            intervention_cost = int(risk_prob * 2000000)
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; color: white;">Intervention Cost</h4>
                <p style="margin: 10px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #feca57;">${intervention_cost:,}</p>
                <small style="opacity: 0.7; color: white;">Estimated USD</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk drivers analysis
        st.markdown("### 📊 Risk Drivers Analysis")
        explanation = predictor.explain_prediction(selected_region, df)
        
        if 'error' not in explanation:
            # Create risk drivers chart
            indicators = []
            changes = []
            colors = []
            
            for indicator, data in explanation.items():
                indicators.append(indicator.replace('_', ' ').title())
                change = data['change_percent']
                changes.append(abs(change))
                
                if abs(change) > 15:
                    colors.append('#ff6b6b')  # High impact
                elif abs(change) > 5:
                    colors.append('#feca57')  # Medium impact
                else:
                    colors.append('#48dbfb')  # Low impact
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=changes,
                y=indicators,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{c:.1f}%" for c in changes],
                textposition='inside'
            ))
            
            fig.update_layout(
                title="Impact of Economic Indicators",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(title="Change %", gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Historical trends
        st.markdown("### 📈 Historical Trends")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            indicator = st.selectbox("Select Indicator", 
                                    ['mobile_money_volume', 'food_price_index', 'unemployment_estimate', 'rainfall'])
        
        # Create enhanced time series
        region_data = df[df['region'] == selected_region].sort_values('date')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=region_data['date'],
            y=region_data[indicator],
            mode='lines+markers',
            line=dict(color='#48dbfb', width=3),
            marker=dict(size=6, color='#48dbfb'),
            fill='tonexty',
            fillcolor='rgba(72, 219, 251, 0.1)',
            name=indicator.replace('_', ' ').title()
        ))
        
        fig.update_layout(
            title=f"{indicator.replace('_', ' ').title()} - {selected_region}",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=400
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_model_performance(predictor):
    st.header("🎯 Model Performance")
    
    if predictor.metadata:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_metrics_gauge(predictor.metadata['accuracy'], "Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_metrics_gauge(predictor.metadata['f1_score'], "F1 Score")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.subheader("📋 Detailed Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            'Score': [
                f"{predictor.metadata['accuracy']:.2%}",
                f"{predictor.metadata['precision']:.2%}",
                f"{predictor.metadata['recall']:.2%}",
                f"{predictor.metadata['f1_score']:.2%}",
                f"{predictor.metadata['roc_auc']:.4f}"
            ]
        })
        st.table(metrics_df)
        
        # Confusion matrix
        st.subheader("🔄 Confusion Matrix")
        fig = create_confusion_matrix(predictor.metadata['confusion_matrix'])
        st.plotly_chart(fig, use_container_width=True)

def show_indicators(df):
    st.header("📈 Economic Indicators")
    
    # Multi-indicator dashboard
    fig = create_indicator_dashboard(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent trends
    st.subheader("📊 Recent Trends (Last 6 Months)")
    recent_data = df[df['date'] >= df['date'].max() - pd.DateOffset(months=6)]
    
    indicators = ['mobile_money_volume', 'food_price_index', 'unemployment_estimate', 'rainfall']
    
    for indicator in indicators:
        avg_recent = recent_data[indicator].mean()
        avg_historical = df[df['date'] < df['date'].max() - pd.DateOffset(months=6)][indicator].mean()
        change = (avg_recent - avg_historical) / avg_historical * 100
        
        trend = "📈" if change > 0 else "📉"
        st.metric(
            indicator.replace('_', ' ').title(),
            f"{avg_recent:.1f}",
            f"{change:+.1f}%"
        )

def create_modern_risk_chart(predictions_df):
    """Create modern risk visualization"""
    # Sort by risk probability
    df_sorted = predictions_df.sort_values('poverty_risk_probability', ascending=True)
    
    # Color mapping
    colors = []
    for risk in df_sorted['risk_level']:
        if risk == 'HIGH':
            colors.append('#ff6b6b')
        elif risk == 'MEDIUM':
            colors.append('#feca57')
        else:
            colors.append('#48dbfb')
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=df_sorted['region'],
        x=df_sorted['poverty_risk_probability'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f"{p:.1%}" for p in df_sorted['poverty_risk_probability']],
        textposition='inside',
        textfont=dict(color='white', size=12, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Risk: %{x:.1%}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Risk Probability by Region",
            font=dict(color='white', size=20, family='Arial Black'),
            x=0.5
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            title="Risk Probability",
            gridcolor='rgba(255,255,255,0.1)',
            tickformat='.0%',
            range=[0, 1]
        ),
        yaxis=dict(
            title="",
            gridcolor='rgba(255,255,255,0.1)'
        ),
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_mini_indicators_dashboard(df):
    """Create mini indicators dashboard"""
    indicators = [
        ('mobile_money_volume', 'Mobile Money', '💳'),
        ('food_price_index', 'Food Prices', '🌾'),
        ('unemployment_estimate', 'Unemployment', '👥'),
        ('rainfall', 'Rainfall', '🌧️')
    ]
    
    cols = st.columns(4)
    
    for i, (indicator, name, icon) in enumerate(indicators):
        with cols[i]:
            # Calculate recent trend
            recent_data = df[df['date'] >= df['date'].max() - pd.DateOffset(months=3)]
            recent_avg = recent_data[indicator].mean()
            
            historical_data = df[df['date'] < df['date'].max() - pd.DateOffset(months=3)]
            historical_avg = historical_data[indicator].mean()
            
            change = (recent_avg - historical_avg) / historical_avg * 100
            
            # Create mini chart
            monthly_data = df.groupby(df['date'].dt.to_period('M'))[indicator].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(monthly_data))),
                y=monthly_data.values,
                mode='lines',
                line=dict(
                    color='#48dbfb' if change >= 0 else '#ff6b6b',
                    width=3
                ),
                fill='tonexty',
                fillcolor='rgba(72, 219, 251, 0.1)' if change >= 0 else 'rgba(255, 107, 107, 0.1)'
            ))
            
            fig.update_layout(
                height=120,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                showlegend=False
            )
            
            # Display card
            trend_icon = "📈" if change > 0 else "📉"
            color_class = "risk-low" if abs(change) < 5 else "risk-medium" if abs(change) < 15 else "risk-high"
            
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <span style="font-size: 1.2rem;">{trend_icon}</span>
                </div>
                <h4 style="margin: 10px 0 5px 0; color: white;">{name}</h4>
                <p style="margin: 0; font-size: 1.1rem; color: white; opacity: 0.9;">{change:+.1f}%</p>
                <small style="opacity: 0.7; color: white;">vs 3 months ago</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

if __name__ == "__main__":
    main()
