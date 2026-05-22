import joblib
import pandas as pd
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from src.train import categorical_without_description, numeric_columns
from src.features import CraigslistFeatureEngineer

def predict_single(model_path: str, input_data: dict) -> float:
    pipe = joblib.load(model_path)
    df_in = pd.DataFrame([input_data])
    return pipe.predict(df_in)[0]

def predict_batch(model_path: str, data_path: str) -> pd.Series:
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path)
    preds = pipe.predict(df)
    return preds

if __name__ == "__main__":
    sample_car = {
        "year": 2015,
        "manufacturer": "toyota",
        "model": "camry",
        "condition": "excellent",
        "cylinders": "4 cylinders",
        "fuel": "gas",
        "odometer": 80000,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "silver",
        "description": "very nice car, well maintained",
        "lat": 34.05,
        "long": -118.24,
        "state": "ca",
        "region": "los angeles"
    }
    
    price = predict_single("models/final_model.pkl", sample_car)
    print(f"Predicted price: ${price:.2f}")