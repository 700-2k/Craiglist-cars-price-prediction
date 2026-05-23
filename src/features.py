from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


CURRENT_YEAR = 2022
SMOOTHING = 20

CYLINDERS_MAP: Dict[str, int] = {
    "3 cylinders": 3, "4 cylinders": 4, "5 cylinders": 5,
    "6 cylinders": 6, "8 cylinders": 8, "10 cylinders": 10,
    "12 cylinders": 12, "other": 0,
}

CAT_COLS: List[str] = [
    "condition", "drive", "transmission", "fuel",
    "manufacturer", "title_status", "type", "model",
]

LOW_CARD_COLS: List[str] = [
    "condition", "transmission", "drive", "fuel", "title_status", "type",
]

HIGH_CARD_COLS: List[str] = ["manufacturer_grouped", "model_grouped", "region"]

NUMERICAL_COLS: List[str] = [
    "car_age", "log_odometer", "cylinders_num", "cylinders_other_flag",
    "car_age_squared", "log_odometer_squared",
    "age_x_odometer", "age_x_cylinders", "odometer_x_cylinders",
    "log_mileage_per_year",
    "region_mean_price", "region_median_price", "region_price_std",
    "manuf_mean_price", "manuf_median_price", "manuf_price_std",
    "manufacturer_grouped_target", "model_grouped_target", "region_target",
]

COLS_TO_DROP: List[str] = [
    "year", "odometer", "lat", "long", "region", "manufacturer",
    "model", "cylinders", "manufacturer_grouped", "model_grouped",
    "mileage_per_year", "description", "state", "paint_color", "posting_date",
]


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------

def basic_transforms(
    X: pd.DataFrame,
    year_mode: float,
    odo_mean: float,
    cylinders_map: Dict[str, int],
    cat_cols: List[str],
) -> pd.DataFrame:
    X = X.copy()
    X["year"] = X["year"].fillna(year_mode)
    X["car_age"] = CURRENT_YEAR - X["year"]
    X["odometer"] = X["odometer"].fillna(odo_mean)
    X["log_odometer"] = np.log1p(X["odometer"])
    X["cylinders"] = X["cylinders"].fillna("other")
    X["cylinders_num"] = X["cylinders"].map(cylinders_map).fillna(0).astype(np.int8)
    X["cylinders_other_flag"] = (X["cylinders"] == "other").astype(np.int8)
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna("unknown")
    return X


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df["car_age_squared"] = (df["car_age"] ** 2).astype(np.float32)
    df["log_odometer_squared"] = (df["log_odometer"] ** 2).astype(np.float32)
    df["age_x_odometer"] = (df["car_age"] * df["log_odometer"]).astype(np.float32)
    df["age_x_cylinders"] = (df["car_age"] * df["cylinders_num"]).astype(np.float32)
    df["odometer_x_cylinders"] = (df["log_odometer"] * df["cylinders_num"]).astype(np.float32)
    df["mileage_per_year"] = (df["odometer"] / (df["car_age"] + 1)).astype(np.float32)
    df["log_mileage_per_year"] = np.log1p(df["mileage_per_year"]).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Sklearn-compatible pipeline
# ---------------------------------------------------------------------------

class FeatureEngineerPipeline(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline that mirrors the notebook logic exactly.

    fit() — learns all statistics from X_train / y_train only (no leakage).
    transform() — applies fitted transforms to any split.
    """

    def __init__(self) -> None:
        self.year_mode_: Optional[float] = None
        self.odo_mean_: Optional[float] = None
        self.rare_models_: Optional[set] = None
        self.rare_manufacturers_: Optional[set] = None
        self.group_manufacturer_: bool = False
        self.train_ohe_cols_: Optional[List[str]] = None
        self.global_mean_log_: Optional[float] = None
        self.target_encoders_: Dict[str, Dict] = {}
        self.region_stats_: Optional[pd.DataFrame] = None
        self.manuf_stats_: Optional[pd.DataFrame] = None
        self.global_region_: Optional[pd.Series] = None
        self.global_manuf_: Optional[pd.Series] = None
        self.scaler_: Optional[StandardScaler] = None
        self.selector_: Optional[SelectKBest] = None
        self.selected_features_: Optional[List[str]] = None
        self.cols_to_scale_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureEngineerPipeline":
        y_log = np.log1p(y)

        # 1. Basic numeric/categorical transforms
        self.year_mode_ = float(X["year"].mode()[0])
        self.odo_mean_ = float(X["odometer"].mean())
        Xtr = basic_transforms(X, self.year_mode_, self.odo_mean_, CYLINDERS_MAP, CAT_COLS)

        # 2. Group rare categories (learn from train)
        model_counts = Xtr["model"].value_counts()
        self.rare_models_ = set(model_counts[model_counts < 20].index)
        Xtr["model_grouped"] = Xtr["model"].where(~Xtr["model"].isin(self.rare_models_), "other")

        self.group_manufacturer_ = Xtr["manufacturer"].nunique() > 50
        if self.group_manufacturer_:
            manuf_counts = Xtr["manufacturer"].value_counts()
            self.rare_manufacturers_ = set(manuf_counts[manuf_counts < 100].index)
            Xtr["manufacturer_grouped"] = Xtr["manufacturer"].where(
                ~Xtr["manufacturer"].isin(self.rare_manufacturers_), "other"
            )
        else:
            self.rare_manufacturers_ = set()
            Xtr["manufacturer_grouped"] = Xtr["manufacturer"]

        # 3. One-Hot Encoding — store column list
        Xtr_ohe = pd.get_dummies(Xtr, columns=LOW_CARD_COLS,
                                 prefix=LOW_CARD_COLS, drop_first=False, dtype=np.int8)
        self.train_ohe_cols_ = Xtr_ohe.columns.tolist()

        # 4. Target Encoding (fit on train only)
        self.global_mean_log_ = float(y_log.mean())
        for col in HIGH_CARD_COLS:
            if col not in Xtr_ohe.columns:
                continue
            temp = pd.Series(y_log.values, index=Xtr_ohe.index, name="target")
            cats = Xtr_ohe[col]
            stats = pd.concat([cats, temp], axis=1).groupby(col)["target"].agg(["mean", "count"])
            stats["smoothed"] = (
                stats["mean"] * stats["count"] + SMOOTHING * self.global_mean_log_
            ) / (stats["count"] + SMOOTHING)
            self.target_encoders_[col] = stats["smoothed"].to_dict()
            Xtr_ohe[f"{col}_target"] = (
                Xtr_ohe[col].map(self.target_encoders_[col])
                .fillna(self.global_mean_log_).astype(np.float32)
            )

        # 5. Region / manufacturer statistics
        self.region_stats_ = (
            Xtr_ohe[["region"]].join(y.rename("price"))
            .groupby("region")["price"].agg(["mean", "median", "std"])
            .rename(columns={"mean": "region_mean_price",
                             "median": "region_median_price",
                             "std": "region_price_std"})
        )
        self.global_region_ = self.region_stats_.mean()
        Xtr_ohe = Xtr_ohe.join(self.region_stats_, on="region")
        for col in ["region_mean_price", "region_median_price", "region_price_std"]:
            Xtr_ohe[col] = Xtr_ohe[col].fillna(self.global_region_[col]).astype(np.float32)

        self.manuf_stats_ = (
            Xtr_ohe[["manufacturer"]].join(y.rename("price"))
            .groupby("manufacturer")["price"].agg(["mean", "median", "std"])
            .rename(columns={"mean": "manuf_mean_price",
                             "median": "manuf_median_price",
                             "std": "manuf_price_std"})
        )
        self.global_manuf_ = self.manuf_stats_.mean()
        Xtr_ohe = Xtr_ohe.join(self.manuf_stats_, on="manufacturer")
        for col in ["manuf_mean_price", "manuf_median_price", "manuf_price_std"]:
            Xtr_ohe[col] = Xtr_ohe[col].fillna(self.global_manuf_[col]).astype(np.float32)

        # 6. Interaction features
        Xtr_ohe = add_interactions(Xtr_ohe)

        # 7. Drop raw columns
        Xtr_fe = Xtr_ohe.drop(columns=[c for c in COLS_TO_DROP if c in Xtr_ohe.columns])

        # 8. Scale numerical columns
        self.cols_to_scale_ = [c for c in NUMERICAL_COLS if c in Xtr_fe.columns]
        self.scaler_ = StandardScaler()
        self.scaler_.fit(Xtr_fe[self.cols_to_scale_].values.astype(np.float64))
        Xtr_fe[self.cols_to_scale_] = self.scaler_.transform(
            Xtr_fe[self.cols_to_scale_].values.astype(np.float64)
        ).astype(np.float32)

        # 9. Feature selection
        Xtr_clean = Xtr_fe.fillna(0)
        K = min(50, Xtr_clean.shape[1])
        self.selector_ = SelectKBest(score_func=f_regression, k=K)
        self.selector_.fit(Xtr_clean.values.astype(np.float32), y_log.values)
        mask = self.selector_.get_support()
        self.selected_features_ = Xtr_clean.columns[mask].tolist()

        return self

    # ------------------------------------------------------------------
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        check_is_fitted(self, ["scaler_", "selector_", "selected_features_"])

        Xte = basic_transforms(X, self.year_mode_, self.odo_mean_, CYLINDERS_MAP, CAT_COLS)

        # Rare grouping (apply train mappings)
        Xte["model_grouped"] = Xte["model"].where(~Xte["model"].isin(self.rare_models_), "other")
        if self.group_manufacturer_:
            Xte["manufacturer_grouped"] = Xte["manufacturer"].where(
                ~Xte["manufacturer"].isin(self.rare_manufacturers_), "other"
            )
        else:
            Xte["manufacturer_grouped"] = Xte["manufacturer"]

        # OHE — reindex to train columns
        Xte_ohe = pd.get_dummies(Xte, columns=LOW_CARD_COLS,
                                 prefix=LOW_CARD_COLS, drop_first=False, dtype=np.int8)
        Xte_ohe = Xte_ohe.reindex(columns=self.train_ohe_cols_, fill_value=0)

        # Target encoding
        for col in HIGH_CARD_COLS:
            if col not in Xte_ohe.columns or col not in self.target_encoders_:
                continue
            Xte_ohe[f"{col}_target"] = (
                Xte_ohe[col].map(self.target_encoders_[col])
                .fillna(self.global_mean_log_).astype(np.float32)
            )

        # Region / manufacturer statistics
        Xte_ohe = Xte_ohe.join(self.region_stats_, on="region")
        for col in ["region_mean_price", "region_median_price", "region_price_std"]:
            Xte_ohe[col] = Xte_ohe[col].fillna(self.global_region_[col]).astype(np.float32)

        Xte_ohe = Xte_ohe.join(self.manuf_stats_, on="manufacturer")
        for col in ["manuf_mean_price", "manuf_median_price", "manuf_price_std"]:
            Xte_ohe[col] = Xte_ohe[col].fillna(self.global_manuf_[col]).astype(np.float32)

        # Interactions
        Xte_ohe = add_interactions(Xte_ohe)

        # Drop raw columns
        Xte_fe = Xte_ohe.drop(columns=[c for c in COLS_TO_DROP if c in Xte_ohe.columns])
        Xte_fe = Xte_fe.reindex(columns=Xte_fe.columns.intersection(
            pd.Index(self.cols_to_scale_).union(Xte_fe.columns)
        ), fill_value=0)

        # Scale
        cols_present = [c for c in self.cols_to_scale_ if c in Xte_fe.columns]
        Xte_fe[cols_present] = self.scaler_.transform(
            Xte_fe[cols_present].values.astype(np.float64)
        ).astype(np.float32)

        # Feature selection
        Xte_clean = Xte_fe.fillna(0)
        Xte_clean = Xte_clean.reindex(
            columns=pd.Index(self.selected_features_), fill_value=0
        )

        return Xte_clean
