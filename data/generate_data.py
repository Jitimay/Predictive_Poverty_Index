import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data():
    """Generate realistic economic data for Burundi regions"""
    
    regions = ['Bujumbura', 'Gitega', 'Ngozi', 'Kayanza', 'Muyinga', 'Ruyigi', 
               'Cankuzo', 'Rutana', 'Bururi', 'Makamba', 'Rumonge', 'Cibitoke', 
               'Bubanza', 'Muramvya', 'Mwaro', 'Karusi', 'Kirundo', 'Bujumbura Rural']
    
    start_date = datetime(2019, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(60)]  # 5 years monthly
    
    data = []
    np.random.seed(42)
    
    for region in regions:
        base_values = {
            'mobile_money_volume': np.random.uniform(50, 200),
            'electricity_consumption': np.random.uniform(20, 100),
            'health_clinic_visits': np.random.uniform(80, 150),
            'school_attendance_rate': np.random.uniform(70, 95),
            'food_price_index': 100,
            'inflation_rate': np.random.uniform(2, 8),
            'exchange_rate': 1900,
            'rainfall': np.random.uniform(50, 200),
            'unemployment_estimate': np.random.uniform(5, 25)
        }
        
        for i, date in enumerate(dates):
            # Add trends and seasonality
            trend_factor = 1 + (i * 0.002)  # Slight upward trend
            seasonal = 1 + 0.1 * np.sin(2 * np.pi * i / 12)  # Annual cycle
            
            # Crisis events (2020, 2022, 2024)
            crisis_factor = 1.0
            if i in [12, 13, 14, 36, 37, 38, 56, 57, 58]:  # Crisis months
                crisis_factor = 0.7 if np.random.random() > 0.3 else 1.0
            
            # Generate correlated indicators
            mobile_money = base_values['mobile_money_volume'] * trend_factor * seasonal * crisis_factor * np.random.uniform(0.8, 1.2)
            electricity = base_values['electricity_consumption'] * trend_factor * crisis_factor * np.random.uniform(0.9, 1.1)
            clinic_visits = base_values['health_clinic_visits'] * (2 - crisis_factor) * np.random.uniform(0.9, 1.1)
            school_attendance = base_values['school_attendance_rate'] * crisis_factor * np.random.uniform(0.95, 1.05)
            food_price = base_values['food_price_index'] * (1 + i * 0.01) * (2 - crisis_factor) * np.random.uniform(0.95, 1.05)
            inflation = base_values['inflation_rate'] * (2 - crisis_factor) * np.random.uniform(0.8, 1.2)
            exchange = base_values['exchange_rate'] * (1 + i * 0.005) * (2 - crisis_factor) * np.random.uniform(0.98, 1.02)
            rainfall = base_values['rainfall'] * seasonal * crisis_factor * np.random.uniform(0.7, 1.3)
            unemployment = base_values['unemployment_estimate'] * (2 - crisis_factor) * np.random.uniform(0.9, 1.1)
            
            # Determine poverty spike (3-6 months ahead)
            future_crisis = 0
            if i + 3 < len(dates) and (i + 3) in [12, 13, 14, 36, 37, 38, 56, 57, 58]:
                future_crisis = 1
            elif i + 4 < len(dates) and (i + 4) in [12, 13, 14, 36, 37, 38, 56, 57, 58]:
                future_crisis = 1
            elif i + 5 < len(dates) and (i + 5) in [12, 13, 14, 36, 37, 38, 56, 57, 58]:
                future_crisis = 1
            elif i + 6 < len(dates) and (i + 6) in [12, 13, 14, 36, 37, 38, 56, 57, 58]:
                future_crisis = 1
            
            # Add some noise to target
            if np.random.random() < 0.1:  # 10% noise
                future_crisis = 1 - future_crisis
            
            data.append({
                'date': date,
                'region': region,
                'mobile_money_volume': round(mobile_money, 2),
                'electricity_consumption': round(electricity, 2),
                'health_clinic_visits': round(clinic_visits, 2),
                'school_attendance_rate': round(min(100, school_attendance), 2),
                'food_price_index': round(food_price, 2),
                'inflation_rate': round(inflation, 2),
                'exchange_rate': round(exchange, 2),
                'rainfall': round(rainfall, 2),
                'unemployment_estimate': round(min(50, unemployment), 2),
                'poverty_spike_next_quarter': future_crisis
            })
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw_data.csv', index=False)
    print(f"Generated {len(df)} records for {len(regions)} regions")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
