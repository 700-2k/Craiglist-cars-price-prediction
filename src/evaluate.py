import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true_log, y_pred_log):
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_true_log)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def plot_actual_vs_predicted(y_true, y_pred, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    
    ax.set_xlabel('Actual Price ($)', fontsize=12)
    ax.set_ylabel('Predicted Price ($)', fontsize=12)
    ax.set_title('Actual vs Predicted', fontsize=14)
    ax.legend()
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, save_path):
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    
    ax.set_xlabel('Residual (Actual - Predicted) ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14)
    ax.legend()
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return residuals.mean(), residuals.std()


def get_worst_predictions(X, y_true, y_pred, n=20):
    df = X.copy()
    df['price_true'] = y_true.values if hasattr(y_true, 'values') else y_true
    df['price_pred'] = y_pred
    df['error_abs'] = np.abs(df['price_true'] - df['price_pred'])
    df['error_pct'] = (df['error_abs'] / df['price_true']) * 100
    
    return df.nlargest(n, 'error_abs')


def plot_error_by_category(results_df, column, title, save_path, top_n=15):
    stats = results_df.groupby(column).agg({
        'price': 'count',
        'error_abs': 'mean',
        'error_pct': 'mean'
    }).rename(columns={'price': 'count', 'error_abs': 'MAE', 'error_pct': 'MAPE'}).reset_index()
    
    if top_n and len(stats) > top_n:
        top_categories = stats.nlargest(top_n, 'count')[column].tolist()
        stats = stats[stats[column].isin(top_categories)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=stats, x='MAE', y=column, ax=axes[0], palette='viridis')
    axes[0].set_title(f'{title} - MAE', fontsize=12)
    axes[0].set_xlabel('MAE ($)')
    axes[0].set_ylabel('')
    
    sns.barplot(data=stats, x='MAPE', y=column, ax=axes[1], palette='flare')
    axes[1].set_title(f'{title} - MAPE', fontsize=12)
    axes[1].set_xlabel('MAPE (%)')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return stats


def run_permutation_importance(model, X, y_log, n_repeats=5, sample_size=3000):
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
        y_sample = y_log.loc[X_sample.index] if hasattr(y_log, 'loc') else y_log[X_sample.index]
    else:
        X_sample = X
        y_sample = y_log
    
    r = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=n_repeats,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': r.importances_mean,
        'Std': r.importances_std
    }).sort_values('Importance', ascending=False)
    
    return importance_df


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
    plt.title("Model Comparison by MAE")
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
    ax.set_title("MAE Comparison: Default vs Tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(list(before.keys()))
    ax.legend()
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
