import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with forward fill"""
    return df.fillna(method='ffill').fillna(method='bfill')

def normalize_features(df: pd.DataFrame) -> tuple:
    """Normalize features to 0-1 range"""
    feature_cols = ['mobile_money_volume', 'electricity_consumption', 'health_clinic_visits',
                   'school_attendance_rate', 'food_price_index', 'inflation_rate',
                   'exchange_rate', 'rainfall', 'unemployment_estimate']
    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler

def create_sequences(df: pd.DataFrame, lookback: int = 12) -> tuple:
    """Create LSTM sequences"""
    feature_cols = ['mobile_money_volume', 'electricity_consumption', 'health_clinic_visits',
                   'school_attendance_rate', 'food_price_index', 'inflation_rate',
                   'exchange_rate', 'rainfall', 'unemployment_estimate']
    
    X, y, regions, dates = [], [], [], []
    
    for region in df['region'].unique():
        region_data = df[df['region'] == region].sort_values('date')
        
        for i in range(lookback, len(region_data)):
            X.append(region_data[feature_cols].iloc[i-lookback:i].values)
            y.append(region_data['poverty_spike_next_quarter'].iloc[i])
            regions.append(region)
            dates.append(region_data['date'].iloc[i])
    
    return np.array(X), np.array(y), regions, dates

def train_test_split_timeseries(X: np.ndarray, y: np.ndarray, test_size: float = 0.3) -> tuple:
    """Split preserving time order"""
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
