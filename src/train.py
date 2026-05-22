import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from src.features import CraigslistFeatureEngineer

def categorical_without_description(X_df: pd.DataFrame) -> list[str]:
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [col for col in cat_cols if col != "description"]

def numeric_columns(X_df: pd.DataFrame) -> list[str]:
    return X_df.select_dtypes(exclude=["object", "category"]).columns.tolist()

def make_pipeline():
    feature_encoder = ColumnTransformer(
        transformers=[
            ("description_tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=5), "description"),
            ("categorical_ohe", OneHotEncoder(handle_unknown="ignore"), categorical_without_description),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    
    cb_base = CatBoostRegressor(random_state=42, verbose=0, iterations=100, depth=6)
    model = TransformedTargetRegressor(regressor=cb_base, func=np.log1p, inverse_func=np.expm1)

    return Pipeline(steps=[
        ("preprocess", CraigslistFeatureEngineer()), 
        ("encode", feature_encoder),
        ("model", model)
    ])

def train_model(data_path: str, model_save_path: str):
    print("Loading data...")
    df = pd.read_csv(data_path, index_col=0)
    
    # Subsample for speed in academic context
    df = df.sample(min(10000, len(df)), random_state=42)
    
    y = df["price"].copy()
    X = df.drop(columns=["price"]).copy()
    
    print("Building pipeline...")
    pipe = make_pipeline()
    
    print("Training model...")
    pipe.fit(X, y)
    
    print(f"Saving model to {model_save_path}...")
    joblib.dump(pipe, model_save_path)
    
    train_preds = pipe.predict(X)
    mae = mean_absolute_error(y, train_preds)
    print(f"Training completed. Train MAE: {mae:.2f}")

if __name__ == "__main__":
    train_model("data/interim/train_filtered.csv", "models/final_model.pkl")