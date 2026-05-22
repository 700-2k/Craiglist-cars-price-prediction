import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge

from src.features import CraigslistFeatureEngineer

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def categorical_without_description(X_df: pd.DataFrame) -> list[str]:
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [col for col in cat_cols if col != "description"]


def numeric_columns(X_df: pd.DataFrame) -> list[str]:
    return X_df.select_dtypes(exclude=["object", "category"]).columns.tolist()


def make_pipeline():
    feature_encoder = ColumnTransformer(
        transformers=[
            ("description_tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 1), min_df=10), "description"),
            ("categorical_ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), categorical_without_description),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    
    base_models = []
    base_models.append(("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
    
    if XGB_AVAILABLE:
        base_models.append(("xgb", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)))
    
    if CATBOOST_AVAILABLE:
        base_models.append(("cb", CatBoostRegressor(iterations=100, verbose=0, random_state=42, thread_count=-1)))
    
    stack_base = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge()
    )
    
    model = TransformedTargetRegressor(regressor=stack_base, func=np.log1p, inverse_func=np.expm1)

    return Pipeline(steps=[
        ("preprocess", CraigslistFeatureEngineer()), 
        ("encode", feature_encoder),
        ("model", model)
    ])


def train_model(data_path: str, model_save_path: str):
    print("Loading data...")
    df = pd.read_csv(data_path, index_col=0)
    
    df = df.sample(min(10000, len(df)), random_state=42)
    
    y = df["price"].copy()
    X = df.drop(columns=["price"]).copy()
    
    print("Building pipeline with Stacking...")
    pipe = make_pipeline()
    
    print("Training model...")
    pipe.fit(X, y)
    
    print(f"Saving model to {model_save_path}...")
    joblib.dump(pipe, model_save_path)
    
    train_preds = pipe.predict(X)
    mae = mean_absolute_error(y, train_preds)
    print(f"Training completed. Train MAE: {mae:.2f}")


if __name__ == "__main__":
    train_model("data/processed/train.csv", "models/final_model.pkl")
