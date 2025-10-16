#!/usr/bin/env python3
"""
Quick Demo Launcher for Predictive Poverty Index System
Run this script to set up and launch the complete demo
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import pandas, numpy, sklearn, tensorflow, streamlit, plotly
        print("✅ All dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def generate_data_if_needed():
    """Generate synthetic data if not exists"""
    if not os.path.exists('data/raw_data.csv'):
        print("📊 Generating synthetic data...")
        os.makedirs('data', exist_ok=True)
        from data.generate_data import generate_synthetic_data
        generate_synthetic_data()
        print("✅ Data generated")
    else:
        print("✅ Data already exists")

def train_model_if_needed():
    """Train model if not exists"""
    if not os.path.exists('models/lstm_model.h5'):
        print("🤖 Training LSTM model...")
        os.makedirs('models', exist_ok=True)
        from models.train_lstm import train_model
        train_model()
        print("✅ Model trained")
    else:
        print("✅ Model already exists")

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("🚀 Launching dashboard...")
    print("📱 Dashboard will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard/app.py", 
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")

def main():
    """Main demo launcher"""
    print("🚨 PREDICTIVE POVERTY INDEX - DEMO LAUNCHER")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('src', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('dashboard', exist_ok=True)
    
    # Create empty __init__.py
    with open('src/__init__.py', 'w') as f:
        pass
    
    print("1️⃣ Checking dependencies...")
    check_dependencies()
    
    print("\n2️⃣ Preparing data...")
    generate_data_if_needed()
    
    print("\n3️⃣ Preparing model...")
    train_model_if_needed()
    
    print("\n4️⃣ Launching dashboard...")
    time.sleep(2)
    launch_dashboard()

if __name__ == "__main__":
    main()
