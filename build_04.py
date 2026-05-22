import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell('# 4. Модели: Обучение и Подбор Гиперпараметров\nВ этом ноутбуке мы обучим Decision Tree, Random Forest и CatBoost, используя логарифм цены (через `TransformedTargetRegressor`) и подберем гиперпараметры с помощью `GridSearchCV`.'),
    
    nbf.v4.new_code_cell('''import sys\nsys.path.append("..")\nfrom pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from catboost import CatBoostRegressor

from src.data import CraigslistPreprocessor'''),

    nbf.v4.new_code_cell('''# Загрузка данных
data_path = Path("../data/interim/train_filtered.csv")
df = pd.read_csv(data_path, index_col=0)

# Небольшая выборка для ускорения обучения в академических целях
df = df.sample(2000, random_state=42)

y = df["price"].copy()
X = df.drop(columns=["price"]).copy()

print(f"X shape: {X.shape}, y shape: {y.shape}")'''),

    nbf.v4.new_code_cell('''# Вспомогательные функции для пайплайна
def categorical_without_description(X_df: pd.DataFrame) -> list[str]:
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [col for col in cat_cols if col != "description"]

def numeric_columns(X_df: pd.DataFrame) -> list[str]:
    return X_df.select_dtypes(exclude=["object", "category"]).columns.tolist()

def make_base_pipeline():
    feature_encoder = ColumnTransformer(
        transformers=[
            ("description_tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=5), "description"),
            ("categorical_ohe", OneHotEncoder(handle_unknown="ignore"), categorical_without_description),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", CraigslistPreprocessor()), ("encode", feature_encoder)])'''),

    nbf.v4.new_code_cell('''# Подготовка CV и метрики
cv = KFold(n_splits=3, shuffle=True, random_state=42) # 3 фолда для скорости
scoring = {"mae": make_scorer(mean_absolute_error, greater_is_better=False)}'''),

    nbf.v4.new_code_cell('''# 1. Decision Tree (через TransformedTargetRegressor для логарифма цены)
dt_base = DecisionTreeRegressor(random_state=42)
dt_model = TransformedTargetRegressor(regressor=dt_base, func=np.log1p, inverse_func=np.expm1)

pipe_dt = Pipeline([
    ("prep", make_base_pipeline()),
    ("model", dt_model)
])

# Сетка гиперпараметров для DT
param_grid_dt = {
    "model__regressor__max_depth": [10, 20],
    "model__regressor__min_samples_split": [10]
}

gs_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1, verbose=2)
gs_dt.fit(X, y)
print("Лучшие параметры Decision Tree:", gs_dt.best_params_)
print("Лучшая MAE:", -gs_dt.best_score_)'''),

    nbf.v4.new_code_cell('''# 2. Random Forest
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_model = TransformedTargetRegressor(regressor=rf_base, func=np.log1p, inverse_func=np.expm1)

pipe_rf = Pipeline([
    ("prep", make_base_pipeline()),
    ("model", rf_model)
])

# Небольшая сетка для RF для экономии времени
param_grid_rf = {
    "model__regressor__n_estimators": [10],
    "model__regressor__max_depth": [10]
}

gs_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1, verbose=2)
gs_rf.fit(X, y)
print("Лучшие параметры Random Forest:", gs_rf.best_params_)
print("Лучшая MAE:", -gs_rf.best_score_)'''),

    nbf.v4.new_code_cell('''# 3. CatBoost (с логарифмированием таргета)
cb_base = CatBoostRegressor(random_state=42, verbose=0, thread_count=-1)
cb_model = TransformedTargetRegressor(regressor=cb_base, func=np.log1p, inverse_func=np.expm1)

pipe_cb = Pipeline([
    ("prep", make_base_pipeline()),
    ("model", cb_model)
])

param_grid_cb = {
    "model__regressor__iterations": [100],
    "model__regressor__depth": [6]
}

gs_cb = GridSearchCV(pipe_cb, param_grid_cb, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1, verbose=2)
gs_cb.fit(X, y)
print("Лучшие параметры CatBoost:", gs_cb.best_params_)
print("Лучшая MAE:", -gs_cb.best_score_)'''),

    nbf.v4.new_code_cell('''# Сохранение результатов лучшей модели
results = {
    "Decision Tree": -gs_dt.best_score_,
    "Random Forest": -gs_rf.best_score_,
    "CatBoost": -gs_cb.best_score_
}

best_model_name = min(results, key=results.get)
print(f"Лучшая модель: {best_model_name} с MAE: {results[best_model_name]:.2f}")

# Для анализа ошибок позже сохраним лучшую модель (мы пересоздадим пайплайн для анализа)
import joblib
best_pipe = gs_cb.best_estimator_ if best_model_name == "CatBoost" else (gs_rf.best_estimator_ if best_model_name == "Random Forest" else gs_dt.best_estimator_)

# Обучаем финальную модель на всём train
best_pipe.fit(X, y)
joblib.dump(best_pipe, "../data/interim/best_model_pipeline.pkl")
print("Финальная модель сохранена.")''')
]

with open('notebooks/04_models.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook 04_models.ipynb created.")
