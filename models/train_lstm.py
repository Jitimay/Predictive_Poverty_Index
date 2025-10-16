import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_data, clean_missing_values, normalize_features, create_sequences, train_test_split_timeseries

def build_lstm_model(input_shape: tuple) -> Sequential:
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape, activation='relu'),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    """Complete training pipeline"""
    print("[INFO] Loading data...")
    df = load_data('data/raw_data.csv')
    df = clean_missing_values(df)
    
    print("[INFO] Preprocessing data...")
    df_scaled, scaler = normalize_features(df)
    
    print("[INFO] Creating sequences...")
    X, y, regions, dates = create_sequences(df_scaled, lookback=12)
    print(f"[INFO] Created {len(X)} sequences")
    
    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split_timeseries(X, y, test_size=0.3)
    
    print("[INFO] Building model...")
    model = build_lstm_model((X.shape[1], X.shape[2]))
    
    print("[INFO] Training model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("[INFO] Evaluating model...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nFINAL RESULTS:")
    print(f"├─ Test Accuracy: {accuracy:.2%}")
    print(f"├─ Precision: {precision:.2%}")
    print(f"├─ Recall: {recall:.2%}")
    print(f"├─ F1 Score: {f1:.2%}")
    print(f"└─ ROC-AUC: {auc:.4f}")
    
    # Save model and metadata
    model.save('models/lstm_model.h5')
    
    metadata = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'input_shape': X.shape[1:],
        'features': ['mobile_money_volume', 'electricity_consumption', 'health_clinic_visits',
                    'school_attendance_rate', 'food_price_index', 'inflation_rate',
                    'exchange_rate', 'rainfall', 'unemployment_estimate']
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("[SUCCESS] Model training complete!")
    return model, metadata

if __name__ == "__main__":
    train_model()
