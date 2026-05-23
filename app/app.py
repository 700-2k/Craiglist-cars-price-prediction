import sys
import os
import joblib
import pandas as pd
sys.path.insert(0, os.path.abspath('.'))

# Simple CLI app to demonstrate prediction
def main():
    print("=== Used Car Price Predictor ===")
    print("Loading model...")
    try:
        model = joblib.load("models/final_model.pkl")
    except FileNotFoundError:
        print("Model not found. Please run 'python -m src.train' first.")
        return

    print("Please enter the car details (or press Enter to use default value):")
    
    # We collect inputs here
    year_str = input("Year (e.g. 2015): ").strip()
    year = int(year_str) if year_str else 2015
    
    manufacturer = input("Manufacturer (e.g. toyota): ").strip() or "toyota"
    model_name = input("Model (e.g. camry): ").strip() or "camry"
    odometer_str = input("Odometer (miles, e.g. 80000): ").strip()
    odometer = float(odometer_str) if odometer_str else 80000.0
    
    region = input("Region (e.g. los angeles): ").strip() or "los angeles"
    state = input("State (e.g. ca): ").strip() or "ca"
    
    car_data = {
        "year": year,
        "manufacturer": manufacturer,
        "model": model_name,
        "condition": "excellent",
        "cylinders": "4 cylinders",
        "fuel": "gas",
        "odometer": odometer,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "silver",
        "description": "",
        "lat": 34.0,
        "long": -118.0,
        "state": state,
        "region": region
    }
    
    df = pd.DataFrame([car_data])
    price = model.predict(df)[0]
    
    print("\n--- Result ---")
    print(f"Predicted Price: ${price:,.2f}")

if __name__ == "__main__":
    main()