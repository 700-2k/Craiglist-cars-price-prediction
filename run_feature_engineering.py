"""
Generate data/processed/ files with proper memory management.
Processes train and test separately to minimize peak memory usage.
"""
import os, gc
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/notebooks")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

CURRENT_YEAR = 2022
SMOOTHING    = 20

out_path = Path('../data/processed')
out_path.mkdir(parents=True, exist_ok=True)
Path('../models').mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CYLINDERS_MAP = {
    '3 cylinders': 3, '4 cylinders': 4, '5 cylinders': 5,
    '6 cylinders': 6, '8 cylinders': 8, '10 cylinders': 10,
    '12 cylinders': 12, 'other': 0
}
LOW_CARD_COLS  = ['condition', 'transmission', 'drive', 'fuel', 'title_status', 'type']
HIGH_CARD_COLS = ['manufacturer_grouped', 'model_grouped', 'region']
CAT_COLS       = ['condition', 'drive', 'transmission', 'fuel',
                  'manufacturer', 'title_status', 'type', 'model']
COLS_TO_DROP   = [
    'year', 'odometer', 'lat', 'long', 'region', 'manufacturer',
    'model', 'cylinders', 'manufacturer_grouped', 'model_grouped',
    'mileage_per_year', 'description', 'state', 'paint_color', 'posting_date'
]
NUMERICAL_COLS = [
    'car_age', 'log_odometer', 'cylinders_num', 'cylinders_other_flag',
    'car_age_squared', 'log_odometer_squared',
    'age_x_odometer', 'age_x_cylinders', 'odometer_x_cylinders',
    'log_mileage_per_year',
    'region_mean_price', 'region_median_price', 'region_price_std',
    'manuf_mean_price', 'manuf_median_price', 'manuf_price_std',
    'manufacturer_grouped_target', 'model_grouped_target', 'region_target'
]


def basic_transforms(X, year_mode, odo_mean, cylinders_map, cat_cols):
    X = X.copy()
    X['year']     = X['year'].fillna(year_mode)
    X['car_age']  = CURRENT_YEAR - X['year']
    X['odometer'] = X['odometer'].fillna(odo_mean)
    X['log_odometer'] = np.log1p(X['odometer'])
    X['cylinders'] = X['cylinders'].fillna('other')
    X['cylinders_num']        = X['cylinders'].map(cylinders_map).fillna(0).astype(np.int8)
    X['cylinders_other_flag'] = (X['cylinders'] == 'other').astype(np.int8)
    for col in cat_cols:
        X[col] = X[col].fillna('unknown')
    return X


def group_rare(s_train, s_test, min_count):
    rare    = s_train.value_counts()[lambda x: x < min_count].index
    s_train = s_train.copy(); s_test = s_test.copy()
    s_train[s_train.isin(rare)] = 'other'
    s_test[s_test.isin(rare)]   = 'other'
    return s_train, s_test


def ohe_apply(X, low_card_cols, train_cols=None):
    X_ohe = pd.get_dummies(X, columns=low_card_cols,
                            prefix=low_card_cols, drop_first=False, dtype=np.int8)
    if train_cols is not None:
        X_ohe = X_ohe.reindex(columns=train_cols, fill_value=0)
    return X_ohe


def add_interactions(df):
    df['car_age_squared']      = (df['car_age'] ** 2).astype(np.float32)
    df['log_odometer_squared'] = (df['log_odometer'] ** 2).astype(np.float32)
    df['age_x_odometer']       = (df['car_age'] * df['log_odometer']).astype(np.float32)
    df['age_x_cylinders']      = (df['car_age'] * df['cylinders_num']).astype(np.float32)
    df['odometer_x_cylinders'] = (df['log_odometer'] * df['cylinders_num']).astype(np.float32)
    df['mileage_per_year']      = (df['odometer'] / (df['car_age'] + 1)).astype(np.float32)
    df['log_mileage_per_year']  = np.log1p(df['mileage_per_year']).astype(np.float32)
    return df


def save_numpy_csv(arr, index, columns, path):
    """Memory-efficient CSV save via numpy."""
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + list(columns))
        for i, row in zip(index, arr):
            writer.writerow([i] + row.tolist())
    print(f'  Saved {len(index)} rows to {path}')


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print("Loading data...")
df_train = pd.read_csv('../data/interim/train_filtered.csv', index_col=0)
df_test  = pd.read_csv('../data/interim/test.csv', index_col=0)
df_test  = df_test[
    (df_test['year'] >= 1980) &
    (df_test['odometer'] > 0) & (df_test['odometer'] <= 500_000) &
    (df_test['price'] >= 500)  & (df_test['price'] <= 100_000)
].copy()
print(f"Train: {df_train.shape}, Test: {df_test.shape}")

test_raw = df_test.copy()

X_train_raw = df_train.drop(columns=['price'])
y_train      = df_train['price']
y_train_log  = np.log1p(y_train)
X_test_raw  = df_test.drop(columns=['price'])
y_test       = df_test['price']
y_test_log   = np.log1p(y_test)
del df_train, df_test; gc.collect()

# Fit parameters from train
year_mode = X_train_raw['year'].mode()[0]
odo_mean  = X_train_raw['odometer'].mean()

# ---------------------------------------------------------------------------
# 2. Basic transforms
# ---------------------------------------------------------------------------
print("Basic transforms...")
X_tr = basic_transforms(X_train_raw, year_mode, odo_mean, CYLINDERS_MAP, CAT_COLS)
X_te = basic_transforms(X_test_raw,  year_mode, odo_mean, CYLINDERS_MAP, CAT_COLS)
del X_train_raw, X_test_raw; gc.collect()

# Group rare categories (fit on train)
X_tr['model_grouped'], X_te['model_grouped'] = group_rare(X_tr['model'], X_te['model'], 20)
if X_tr['manufacturer'].nunique() > 50:
    X_tr['manufacturer_grouped'], X_te['manufacturer_grouped'] = group_rare(
        X_tr['manufacturer'], X_te['manufacturer'], 100)
else:
    X_tr['manufacturer_grouped'] = X_tr['manufacturer']
    X_te['manufacturer_grouped'] = X_te['manufacturer']

# ---------------------------------------------------------------------------
# 3. OHE
# ---------------------------------------------------------------------------
print("OHE...")
X_tr_ohe = ohe_apply(X_tr, LOW_CARD_COLS)
train_ohe_cols = X_tr_ohe.columns.tolist()
X_te_ohe = ohe_apply(X_te, LOW_CARD_COLS, train_cols=train_ohe_cols)
del X_tr, X_te; gc.collect()
print(f"  After OHE: {X_tr_ohe.shape}")

# ---------------------------------------------------------------------------
# 4. Target Encoding (fit on train)
# ---------------------------------------------------------------------------
print("Target encoding...")
global_mean_log = float(y_train_log.mean())
target_encoders = {}

for col in HIGH_CARD_COLS:
    if col not in X_tr_ohe.columns:
        continue
    temp  = pd.Series(y_train_log.values, index=X_tr_ohe.index, name='target')
    cats  = X_tr_ohe[col]
    stats = pd.concat([cats, temp], axis=1).groupby(col)['target'].agg(['mean', 'count'])
    stats['smoothed'] = (
        stats['mean'] * stats['count'] + SMOOTHING * global_mean_log
    ) / (stats['count'] + SMOOTHING)
    mapping = stats['smoothed'].to_dict()
    target_encoders[col] = mapping
    X_tr_ohe[f'{col}_target'] = X_tr_ohe[col].map(mapping).fillna(global_mean_log).astype(np.float32)
    X_te_ohe[f'{col}_target'] = X_te_ohe[col].map(mapping).fillna(global_mean_log).astype(np.float32)
    print(f"  TE: {col} ({len(mapping)} cats)")

# ---------------------------------------------------------------------------
# 5. Region & manufacturer stats (fit on train)
# ---------------------------------------------------------------------------
print("Region/manufacturer stats...")
region_stats = (X_tr_ohe[['region']].join(y_train.rename('price'))
                .groupby('region')['price'].agg(['mean', 'median', 'std'])
                .rename(columns={'mean': 'region_mean_price',
                                 'median': 'region_median_price',
                                 'std': 'region_price_std'}))
global_r = region_stats.mean()
X_tr_ohe = X_tr_ohe.join(region_stats, on='region')
X_te_ohe = X_te_ohe.join(region_stats, on='region')
for col in ['region_mean_price', 'region_median_price', 'region_price_std']:
    X_tr_ohe[col] = X_tr_ohe[col].fillna(global_r[col]).astype(np.float32)
    X_te_ohe[col] = X_te_ohe[col].fillna(global_r[col]).astype(np.float32)

manuf_stats = (X_tr_ohe[['manufacturer']].join(y_train.rename('price'))
               .groupby('manufacturer')['price'].agg(['mean', 'median', 'std'])
               .rename(columns={'mean': 'manuf_mean_price',
                                'median': 'manuf_median_price',
                                'std': 'manuf_price_std'}))
global_m = manuf_stats.mean()
X_tr_ohe = X_tr_ohe.join(manuf_stats, on='manufacturer')
X_te_ohe = X_te_ohe.join(manuf_stats, on='manufacturer')
for col in ['manuf_mean_price', 'manuf_median_price', 'manuf_price_std']:
    X_tr_ohe[col] = X_tr_ohe[col].fillna(global_m[col]).astype(np.float32)
    X_te_ohe[col] = X_te_ohe[col].fillna(global_m[col]).astype(np.float32)

# ---------------------------------------------------------------------------
# 6. Interactions
# ---------------------------------------------------------------------------
print("Interaction features...")
X_tr_ohe = add_interactions(X_tr_ohe)
X_te_ohe = add_interactions(X_te_ohe)

# Drop raw columns
X_tr_fe = X_tr_ohe.drop(columns=[c for c in COLS_TO_DROP if c in X_tr_ohe.columns])
X_te_fe = X_te_ohe.drop(columns=[c for c in COLS_TO_DROP if c in X_te_ohe.columns])
X_te_fe = X_te_fe.reindex(columns=X_tr_fe.columns, fill_value=0)
del X_tr_ohe, X_te_ohe; gc.collect()
print(f"  After drop: {X_tr_fe.shape}")

# ---------------------------------------------------------------------------
# 7. Scaling (fit on train)
# ---------------------------------------------------------------------------
print("Scaling...")
cols_to_scale = [c for c in NUMERICAL_COLS if c in X_tr_fe.columns]
scaler = StandardScaler()

# Convert to float32 arrays for scaling
tr_vals = X_tr_fe[cols_to_scale].values.astype(np.float64)
scaler.fit(tr_vals)
X_tr_fe[cols_to_scale] = scaler.transform(tr_vals).astype(np.float32)
del tr_vals

te_vals = X_te_fe[cols_to_scale].values.astype(np.float64)
X_te_fe[cols_to_scale] = scaler.transform(te_vals).astype(np.float32)
del te_vals; gc.collect()

# ---------------------------------------------------------------------------
# 8. Feature selection (fit on train)
# ---------------------------------------------------------------------------
print("Feature selection...")
X_tr_clean = X_tr_fe.fillna(0)
X_te_clean = X_te_fe.fillna(0)
X_te_clean = X_te_clean.reindex(columns=X_tr_clean.columns, fill_value=0)
del X_tr_fe, X_te_fe; gc.collect()

K = min(50, X_tr_clean.shape[1])
# Use numpy array to save memory during SelectKBest
tr_arr = X_tr_clean.values.astype(np.float32)
selector = SelectKBest(score_func=f_regression, k=K)
selector.fit(tr_arr, y_train_log.values)
selected_mask     = selector.get_support()
selected_features = X_tr_clean.columns[selected_mask].tolist()
print(f"  Selected {len(selected_features)} features")

# Extract final arrays
X_train_arr = tr_arr[:, selected_mask]
del tr_arr; gc.collect()

te_arr = X_te_clean.values.astype(np.float32)
X_test_arr  = te_arr[:, selected_mask]
del te_arr, X_tr_clean, X_te_clean; gc.collect()

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
print("Saving processed data...")
train_index = y_train_log.index
test_index  = y_test_log.index

save_numpy_csv(X_train_arr, train_index, selected_features, out_path / 'X_train.csv')
save_numpy_csv(X_test_arr,  test_index,  selected_features, out_path / 'X_test.csv')

y_train_log.rename('price').to_csv(out_path / 'y_train.csv', header=True)
y_test_log.rename('price').to_csv(out_path / 'y_test.csv',  header=True)
print(f"  Saved y_train ({len(y_train_log)}) and y_test ({len(y_test_log)})")

# Save test_raw in chunks
chunk_size = 10000
test_raw_path = str(out_path / 'test_raw.csv')
for i, start in enumerate(range(0, len(test_raw), chunk_size)):
    chunk = test_raw.iloc[start:start + chunk_size]
    chunk.to_csv(test_raw_path, mode='w' if i == 0 else 'a', header=(i == 0))
print(f"  Saved test_raw ({len(test_raw)} rows)")

# Artifacts
joblib.dump(scaler,            '../models/scaler.pkl')
joblib.dump(selected_features, '../models/feature_names.pkl')
joblib.dump(target_encoders,   '../models/target_encoders.pkl')
print("  Saved scaler.pkl, feature_names.pkl, target_encoders.pkl")

print()
print("=" * 50)
print("DONE!")
print(f"  X_train: {X_train_arr.shape}")
print(f"  X_test:  {X_test_arr.shape}")
print(f"  Features: {selected_features[:5]} ...")
print("=" * 50)
