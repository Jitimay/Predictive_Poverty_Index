import streamlit as st
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PovertyPredictor
from src.visualization import (create_risk_heatmap, create_time_series_chart, 
                              create_confusion_matrix, create_metrics_gauge,
                              create_indicator_dashboard, create_risk_summary_cards)

# Page config
st.set_page_config(
    page_title="Predictive Poverty Index - Burundi",
    page_icon="🚨",
    layout="wide"
)

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
    st.title("🚨 Predictive Poverty Index - Burundi")
    st.markdown("**AI-Powered Early Warning System for Poverty Prevention**")
    
    # Load data
    df = load_data()
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Model not found. Please run training first.")
        return
    
    # Get predictions
    predictions_df = predictor.predict_all_regions(df)
    summary = create_risk_summary_cards(predictions_df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select Page", 
                               ["Overview", "Regional Analysis", "Model Performance", "Indicators"])
    
    if page == "Overview":
        show_overview(predictions_df, summary, df)
    elif page == "Regional Analysis":
        show_regional_analysis(predictions_df, df, predictor)
    elif page == "Model Performance":
        show_model_performance(predictor)
    elif page == "Indicators":
        show_indicators(df)

def show_overview(predictions_df, summary, df):
    st.header("📊 Overview Dashboard")
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 HIGH RISK", summary['high_risk'], "Regions")
    with col2:
        st.metric("🟡 MEDIUM RISK", summary['medium_risk'], "Regions")
    with col3:
        st.metric("🟢 LOW RISK", summary['low_risk'], "Regions")
    with col4:
        st.metric("📍 TOTAL", summary['total_regions'], "Regions")
    
    # Risk heatmap
    st.subheader("Regional Risk Assessment")
    fig = create_risk_heatmap(predictions_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🎯 Key Insights")
    high_risk_regions = predictions_df[predictions_df['risk_level'] == 'HIGH']['region'].tolist()
    if high_risk_regions:
        st.warning(f"**Immediate Attention Required:** {', '.join(high_risk_regions)}")
    else:
        st.success("No regions currently at high risk!")

def show_regional_analysis(predictions_df, df, predictor):
    st.header("🗺️ Regional Analysis")
    
    # Region selector
    selected_region = st.selectbox("Select Region", predictions_df['region'].unique())
    
    if selected_region:
        region_pred = predictions_df[predictions_df['region'] == selected_region].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Risk Probability", f"{region_pred['poverty_risk_probability']:.1%}")
        with col2:
            risk_color = "🔴" if region_pred['risk_level'] == 'HIGH' else "🟡" if region_pred['risk_level'] == 'MEDIUM' else "🟢"
            st.metric("Risk Level", f"{risk_color} {region_pred['risk_level']}")
        
        # Explanation
        st.subheader("📈 What's Driving the Risk?")
        explanation = predictor.explain_prediction(selected_region, df)
        
        if 'error' not in explanation:
            for indicator, data in explanation.items():
                change = data['change_percent']
                trend = "📈" if change > 5 else "📉" if change < -5 else "➡️"
                st.write(f"{trend} **{indicator.replace('_', ' ').title()}**: {change:+.1f}% change")
        
        # Time series
        st.subheader("📊 Historical Trends")
        indicator = st.selectbox("Select Indicator", 
                                ['mobile_money_volume', 'food_price_index', 'unemployment_estimate'])
        fig = create_time_series_chart(df, selected_region, indicator)
        st.plotly_chart(fig, use_container_width=True)

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

if __name__ == "__main__":
    main()
