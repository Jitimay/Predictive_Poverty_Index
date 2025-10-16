import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

class PovertyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        
    def load_trained_model(self):
        """Load saved model and metadata"""
        try:
            self.model = load_model('models/lstm_model.h5')
            with open('models/model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            return True
        except:
            return False
    
    def prepare_scaler(self, df: pd.DataFrame):
        """Prepare scaler from data"""
        feature_cols = ['mobile_money_volume', 'electricity_consumption', 'health_clinic_visits',
                       'school_attendance_rate', 'food_price_index', 'inflation_rate',
                       'exchange_rate', 'rainfall', 'unemployment_estimate']
        
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[feature_cols])
    
    def predict_poverty_risk(self, region_data: np.ndarray) -> tuple:
        """Predict poverty risk for region sequence"""
        if self.model is None:
            return 0.0, "Model not loaded"
        
        # Ensure correct shape
        if len(region_data.shape) == 2:
            region_data = region_data.reshape(1, region_data.shape[0], region_data.shape[1])
        
        prediction = self.model.predict(region_data, verbose=0)[0][0]
        
        if prediction > 0.7:
            risk_level = "HIGH"
        elif prediction > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        return float(prediction), risk_level
    
    def predict_all_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch predictions for all regions"""
        if self.scaler is None:
            self.prepare_scaler(df)
        
        feature_cols = ['mobile_money_volume', 'electricity_consumption', 'health_clinic_visits',
                       'school_attendance_rate', 'food_price_index', 'inflation_rate',
                       'exchange_rate', 'rainfall', 'unemployment_estimate']
        
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.transform(df[feature_cols])
        
        results = []
        
        for region in df['region'].unique():
            region_data = df_scaled[df_scaled['region'] == region].sort_values('date')
            
            if len(region_data) >= 12:
                # Use last 12 months
                sequence = region_data[feature_cols].tail(12).values
                prediction, risk_level = self.predict_poverty_risk(sequence)
                
                results.append({
                    'region': region,
                    'poverty_risk_probability': prediction,
                    'risk_level': risk_level,
                    'last_updated': region_data['date'].max()
                })
        
        return pd.DataFrame(results)
    
    def explain_prediction(self, region: str, df: pd.DataFrame) -> dict:
        """Explain what drives the prediction"""
        feature_cols = ['mobile_money_volume', 'electricity_consumption', 'health_clinic_visits',
                       'school_attendance_rate', 'food_price_index', 'inflation_rate',
                       'exchange_rate', 'rainfall', 'unemployment_estimate']
        
        region_data = df[df['region'] == region].sort_values('date').tail(12)
        
        if len(region_data) < 12:
            return {"error": "Insufficient data"}
        
        # Calculate recent trends
        trends = {}
        for col in feature_cols:
            recent_avg = region_data[col].tail(3).mean()
            historical_avg = region_data[col].head(9).mean()
            change = (recent_avg - historical_avg) / historical_avg * 100
            trends[col] = {
                'recent_avg': recent_avg,
                'historical_avg': historical_avg,
                'change_percent': change
            }
        
        return trends
