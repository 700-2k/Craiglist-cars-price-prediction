import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(name, model, X, y_log_true):
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_log_true)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def compare_models(results_list):
    df = pd.DataFrame(results_list)
    return df.sort_values("MAE").reset_index(drop=True)


def plot_model_comparison(results_df, save_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="MAE", y="Model", palette="viridis")
    plt.title("Сравнение моделей по MAE")
    plt.xlabel("MAE ($)")
    plt.ylabel("")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tuning_comparison(before, after, save_path):
    if len(before) == 0 or len(after) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(before))
    width = 0.35

    ax.bar(x - width/2, before, width, label="Default", color="#1f77b4")
    ax.bar(x + width/2, after, width, label="Tuned", color="#2ca02c")

    ax.set_ylabel("MAE ($)")
    ax.set_title("Сравнение MAE до и после подбора гиперпараметров")
    ax.set_xticks(x)
    ax.set_xticklabels(list(before.keys()))
    ax.legend()
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
