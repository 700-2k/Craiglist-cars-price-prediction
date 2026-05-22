import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.base import clone

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


def get_default_models():
    models = {
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostRegressor(iterations=100, verbose=0, random_state=42, thread_count=-1)

    return models


def train_model(model, X, y):
    model.fit(X, y)
    return model


def run_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(
        clone(model), X, y,
        cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    mae_mean = -scores.mean()
    mae_std = scores.std()
    return mae_mean, mae_std


def tune_random_search(model, param_dist, X, y, n_iter=20):
    search = RandomizedSearchCV(
        model, param_dist,
        n_iter=n_iter, cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=42
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, -search.best_score_


def tune_grid_search(model, param_grid, X, y):
    grid_search = GridSearchCV(
        model, param_grid,
        cv=5, scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def get_param_distributions():
    return {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
        "CatBoost": {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
        },
        "DecisionTree": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    }
