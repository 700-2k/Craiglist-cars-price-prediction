from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


CURRENT_YEAR = None
CYLINDERS_MAP = {
    "3 cylinders": 3,
    "4 cylinders": 4,
    "5 cylinders": 5,
    "6 cylinders": 6,
    "8 cylinders": 8,
    "10 cylinders": 10,
    "12 cylinders": 12,
    "other": 0
}

CATEGORICAL_COLS_TO_FILL = [
    'condition', 'drive', 'transmission', 'fuel', 
    'manufacturer', 'title_status', 'type', 'model'
]

LOW_CARDINALITY_COLS = [
    'condition', 'transmission', 'drive', 'fuel', 'title_status', 'type'
]

HIGH_CARDINALITY_COLS = ['manufacturer_grouped', 'model_grouped', 'region']

NUMERICAL_COLS_TO_SCALE = [
    'car_age', 'log_odometer', 'cylinders_num',
    'car_age_squared', 'log_odometer_squared',
    'age_x_odometer', 'age_x_cylinders', 'odometer_x_cylinders',
    'mileage_per_year', 'log_mileage_per_year',
    'region_mean_price', 'region_median_price', 'region_price_std',
    'manuf_mean_price', 'manuf_median_price', 'manuf_price_std',
    'dist_to_region_center',
    'manufacturer_target', 'model_target', 'region_target'
]

COLS_TO_DROP = [
    'year', 'odometer', 'lat', 'long', 'region', 'manufacturer', 
    'model', 'cylinders', 'manufacturer_grouped', 'model_grouped',
    'lat_region_mean', 'long_region_mean', 'mileage_per_year'
]


def add_car_age(df: pd.DataFrame, current_year: Optional[int] = None) -> pd.DataFrame:
    result = df.copy()
    if current_year is None:
        current_year = result['year'].max()
    result['year'] = result['year'].fillna(result['year'].mode()[0])
    result['car_age'] = current_year - result['year']
    return result


def log_transform_odometer(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result['log_odometer'] = np.log1p(result['odometer'])
    return result


def map_cylinders(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result['cylinders'] = result['cylinders'].fillna('other')
    result['cylinders_num'] = result['cylinders'].map(CYLINDERS_MAP).fillna(0).astype(int)
    result['cylinders_other_flag'] = (result['cylinders'] == 'other').astype(int)
    return result


def fill_missing_categoricals(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    result = df.copy()
    if cols is None:
        cols = CATEGORICAL_COLS_TO_FILL
    for col in cols:
        if col in result.columns:
            result[col] = result[col].fillna('unknown')
    return result


def group_rare_categories(series: pd.Series, min_count: int = 20) -> Tuple[pd.Series, int]:
    value_counts = series.value_counts()
    rare_categories = value_counts[value_counts < min_count].index
    result = series.copy()
    result[series.isin(rare_categories)] = 'other'
    return result, len(rare_categories)


def target_encode(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    column: str, 
    y_train: pd.Series, 
    smoothing: float = 10
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    global_mean = y_train.mean()
    agg = X_train.join(y_train).groupby(column)[y_train.name].agg(['mean', 'count'])
    agg['smoothed_mean'] = (
        agg['mean'] * agg['count'] + smoothing * global_mean
    ) / (agg['count'] + smoothing)
    mapping = agg['smoothed_mean'].to_dict()
    train_encoded = X_train[column].map(mapping).fillna(global_mean)
    test_encoded = X_test[column].map(mapping).fillna(global_mean)
    return train_encoded, test_encoded, mapping


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result['car_age_squared'] = result['car_age'] ** 2
    result['log_odometer_squared'] = result['log_odometer'] ** 2
    result['age_x_odometer'] = result['car_age'] * result['log_odometer']
    result['age_x_cylinders'] = result['car_age'] * result['cylinders_num']
    result['odometer_x_cylinders'] = result['log_odometer'] * result['cylinders_num']
    result['mileage_per_year'] = result['odometer'] / (result['car_age'] + 1)
    result['log_mileage_per_year'] = np.log1p(result['mileage_per_year'])
    return result


def add_region_features(
    df: pd.DataFrame, 
    region_stats: pd.DataFrame
) -> pd.DataFrame:
    result = df.merge(
        region_stats[['region', 'region_mean_price', 'region_median_price', 'region_price_std']], 
        on='region', 
        how='left'
    )
    result['region_mean_price'] = result['region_mean_price'].fillna(region_stats['region_mean_price'].mean())
    result['region_median_price'] = result['region_median_price'].fillna(region_stats['region_median_price'].mean())
    result['region_price_std'] = result['region_price_std'].fillna(region_stats['region_price_std'].mean())
    return result


def add_manufacturer_features(
    df: pd.DataFrame, 
    manuf_stats: pd.DataFrame
) -> pd.DataFrame:
    result = df.merge(
        manuf_stats[['manufacturer', 'manuf_mean_price', 'manuf_median_price', 'manuf_price_std']], 
        on='manufacturer', 
        how='left'
    )
    global_manuf_mean = manuf_stats['manuf_mean_price'].mean()
    global_manuf_median = manuf_stats['manuf_median_price'].mean()
    global_manuf_std = manuf_stats['manuf_price_std'].mean()
    result['manuf_mean_price'] = result['manuf_mean_price'].fillna(global_manuf_mean)
    result['manuf_median_price'] = result['manuf_median_price'].fillna(global_manuf_median)
    result['manuf_price_std'] = result['manuf_price_std'].fillna(global_manuf_std)
    return result


class FeatureEngineerPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = None
        self.selector = None
        self.target_encoders = {}
        self.region_stats = None
        self.manuf_stats = None
        self.current_year = None
        self.selected_features = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureEngineerPipeline":
        df = X.copy()
        self.current_year = df['year'].max()
        
        df = add_car_age(df, self.current_year)
        df = log_transform_odometer(df)
        df = map_cylinders(df)
        df = fill_missing_categoricals(df)
        
        X_train, X_temp, y_train, y_temp = self._split_data(df, y)
        
        X_train['model_grouped'], _ = group_rare_categories(X_train['model'], min_count=20)
        X_temp['model_grouped'], _ = group_rare_categories(X_temp['model'], min_count=20)
        
        if X_train['manufacturer'].nunique() > 50:
            X_train['manufacturer_grouped'], _ = group_rare_categories(X_train['manufacturer'], min_count=100)
            X_temp['manufacturer_grouped'], _ = group_rare_categories(X_temp['manufacturer'], min_count=100)
        else:
            X_train['manufacturer_grouped'] = X_train['manufacturer']
            X_temp['manufacturer_grouped'] = X_temp['manufacturer']
        
        X_train_ohe = pd.get_dummies(X_train, columns=LOW_CARDINALITY_COLS, prefix=LOW_CARDINALITY_COLS, drop_first=False, dtype=int)
        X_temp_ohe = pd.get_dummies(X_temp, columns=LOW_CARDINALITY_COLS, prefix=LOW_CARDINALITY_COLS, drop_first=False, dtype=int)
        X_temp_ohe = X_temp_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)
        
        train_with_price = X_train_ohe.join(y_train)
        self.region_stats = train_with_price.groupby('region')['price'].agg(['mean', 'median', 'std', 'count']).reset_index()
        self.region_stats.columns = ['region', 'region_mean_price', 'region_median_price', 'region_price_std', 'region_count']
        
        self.manuf_stats = train_with_price.groupby('manufacturer')['price'].agg(['mean', 'median', 'std', 'count']).reset_index()
        self.manuf_stats.columns = ['manufacturer', 'manuf_mean_price', 'manuf_median_price', 'manuf_price_std', 'manuf_count']
        
        for col in HIGH_CARDINALITY_COLS:
            if col in X_train_ohe.columns:
                train_enc, _, mapping = target_encode(X_train_ohe, X_temp_ohe, col, y_train, smoothing=20)
                X_train_ohe[f'{col}_target'] = train_enc
                self.target_encoders[col] = mapping
        
        X_train_fe = add_region_features(X_train_ohe, self.region_stats)
        X_train_fe = add_manufacturer_features(X_train_fe, self.manuf_stats)
        
        region_coords = train_with_price.groupby('region')[['lat', 'long']].mean().reset_index()
        X_train_fe = X_train_fe.merge(region_coords, on='region', how='left', suffixes=('', '_region_mean'))
        global_lat = region_coords['lat'].mean()
        global_long = region_coords['long'].mean()
        X_train_fe['lat_region_mean'] = X_train_fe['lat_region_mean'].fillna(global_lat)
        X_train_fe['long_region_mean'] = X_train_fe['long_region_mean'].fillna(global_long)
        X_train_fe['dist_to_region_center'] = np.sqrt(
            (X_train_fe['lat'] - X_train_fe['lat_region_mean'])**2 + 
            (X_train_fe['long'] - X_train_fe['long_region_mean'])**2
        )
        
        X_train_fe = add_interaction_features(X_train_fe)
        
        cols_to_scale = [col for col in NUMERICAL_COLS_TO_SCALE if col in X_train_fe.columns]
        self.scaler = StandardScaler()
        X_train_scaled = X_train_fe.copy()
        X_train_scaled.loc[:, cols_to_scale] = self.scaler.fit_transform(X_train_fe[cols_to_scale])
        
        X_train_final = X_train_scaled.drop(columns=[col for col in COLS_TO_DROP if col in X_train_scaled.columns])
        
        k = min(50, X_train_final.shape[1])
        self.selector = SelectKBest(score_func=f_regression, k=k)
        self.selector.fit(X_train_final, y_train)
        selected_mask = self.selector.get_support()
        self.selected_features = X_train_final.columns[selected_mask].tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        check_is_fitted(self, ['scaler', 'selector', 'selected_features', 'current_year'])
        
        df = X.copy()
        df = add_car_age(df, self.current_year)
        df = log_transform_odometer(df)
        df = map_cylinders(df)
        df = fill_missing_categoricals(df)
        
        df['model_grouped'], _ = group_rare_categories(df['model'], min_count=20)
        
        if df['manufacturer'].nunique() > 50:
            df['manufacturer_grouped'], _ = group_rare_categories(df['manufacturer'], min_count=100)
        else:
            df['manufacturer_grouped'] = df['manufacturer']
        
        df_ohe = pd.get_dummies(df, columns=LOW_CARDINALITY_COLS, prefix=LOW_CARDINALITY_COLS, drop_first=False, dtype=int)
        
        for col in HIGH_CARDINALITY_COLS:
            if col in df_ohe.columns and col in self.target_encoders:
                mapping = self.target_encoders[col]
                global_mean = y.mean() if y is not None else 0
                df_ohe[f'{col}_target'] = df_ohe[col].map(mapping).fillna(global_mean)
        
        if self.region_stats is not None:
            df_fe = add_region_features(df_ohe, self.region_stats)
        else:
            df_fe = df_ohe
            
        if self.manuf_stats is not None:
            df_fe = add_manufacturer_features(df_fe, self.manuf_stats)
        
        if self.region_stats is not None:
            region_coords = X.groupby('region')[['lat', 'long']].mean().reset_index()
            df_fe = df_fe.merge(region_coords, on='region', how='left', suffixes=('', '_region_mean'))
            global_lat = region_coords['lat'].mean()
            global_long = region_coords['long'].mean()
            df_fe['lat_region_mean'] = df_fe['lat_region_mean'].fillna(global_lat)
            df_fe['long_region_mean'] = df_fe['long_region_mean'].fillna(global_long)
            df_fe['dist_to_region_center'] = np.sqrt(
                (df_fe['lat'] - df_fe['lat_region_mean'])**2 + 
                (df_fe['long'] - df_fe['long_region_mean'])**2
            )
        
        df_fe = add_interaction_features(df_fe)
        
        cols_to_scale = [col for col in NUMERICAL_COLS_TO_SCALE if col in df_fe.columns]
        df_scaled = df_fe.copy()
        df_scaled.loc[:, cols_to_scale] = self.scaler.transform(df_fe[cols_to_scale])
        
        df_final = df_scaled.drop(columns=[col for col in COLS_TO_DROP if col in df_scaled.columns])
        
        X_selected = df_final[self.selected_features]
        
        return X_selected
    
    def _split_data(self, df: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(df, y, test_size=test_size, random_state=random_state)
        return X_train, X_temp, y_train, y_temp


def build_features(
    df: pd.DataFrame, 
    y: Optional[pd.Series] = None, 
    fit: bool = True, 
    artifacts: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    if fit:
        pipeline = FeatureEngineerPipeline()
        pipeline.fit(df, y)
        X_transformed = pipeline.transform(df, y)
        artifacts = {
            'scaler': pipeline.scaler,
            'selector': pipeline.selector,
            'target_encoders': pipeline.target_encoders,
            'region_stats': pipeline.region_stats,
            'manuf_stats': pipeline.manuf_stats,
            'current_year': pipeline.current_year,
            'selected_features': pipeline.selected_features
        }
        return X_transformed, artifacts
    else:
        if artifacts is None:
            raise ValueError("artifacts must be provided when fit=False")
        pipeline = FeatureEngineerPipeline()
        pipeline.scaler = artifacts['scaler']
        pipeline.selector = artifacts['selector']
        pipeline.target_encoders = artifacts['target_encoders']
        pipeline.region_stats = artifacts['region_stats']
        pipeline.manuf_stats = artifacts['manuf_stats']
        pipeline.current_year = artifacts['current_year']
        pipeline.selected_features = artifacts['selected_features']
        X_transformed = pipeline.transform(df, y)
        return X_transformed, artifacts
