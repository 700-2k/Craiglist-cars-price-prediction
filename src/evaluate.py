import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.train import categorical_without_description, numeric_columns
from src.features import CraigslistFeatureEngineer


def evaluate_model(model_path: str, test_data_path: str):
    print("Loading test data...")
    df = pd.read_csv(test_data_path, index_col=0)
    
    if "price" not in df.columns:
        print("Error: Test data must contain 'price' column.")
        return
        
    df = df.dropna(subset=["price"])
    df = df[(df["price"] >= 500) & (df["price"] <= 100000)]
    df = df[df["year"] >= 1980]
    
    y = df["price"].copy()
    X = df.drop(columns=["price"]).copy()
    
    print("Loading model...")
    pipe = joblib.load(model_path)
    
    print("Evaluating...")
    preds = pipe.predict(X)
    
    mae = mean_absolute_error(y, preds)
    mape = mean_absolute_percentage_error(y, preds)
    
    print(f"Test MAE:  {mae:.2f} USD")
    print(f"Test MAPE: {mape:.2%}")
    
    return mae, mape


if __name__ == "__main__":
    evaluate_model("models/final_model.pkl", "data/processed/train.csv")