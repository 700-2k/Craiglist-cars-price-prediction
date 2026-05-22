from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted


class CraigslistPreprocessor(BaseEstimator, TransformerMixin):
    CYLINDERS_MAP: dict[str, int] = {
        "3 cylinders": 3,
        "4 cylinders": 4,
        "5 cylinders": 5,
        "6 cylinders": 6,
        "8 cylinders": 8,
        "10 cylinders": 10,
        "12 cylinders": 12,
        "other": 0,
    }

    FILL_UNKNOWN_COLS: tuple[str, ...] = (
        "type",
        "drive",
        "transmission",
        "fuel",
        "condition",
        "manufacturer",
        "title_status",
    )

    def __init__(
        self,
        *,
        model_min_count: int = 20,
        region_n_clusters: int = 10,
        paint_n_clusters: int = 5,
    ) -> None:
        self.model_min_count = model_min_count
        self.region_n_clusters = region_n_clusters
        self.paint_n_clusters = paint_n_clusters

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CraigslistPreprocessor":
        df = self._to_frame(X)
        price_for_fit = self._extract_price_series(y=y, index=df.index)

        self.year_imputer_ = SimpleImputer(strategy="most_frequent")
        self.odometer_imputer_ = SimpleImputer(strategy="mean")
        self.description_imputer_ = SimpleImputer(strategy="constant", fill_value="")
        self.model_imputer_ = SimpleImputer(strategy="constant", fill_value="unknown")
        self.unknown_imputers_ = {
            col: SimpleImputer(strategy="constant", fill_value="unknown")
            for col in self.FILL_UNKNOWN_COLS
            if col in df.columns
        }

        if "year" in df.columns:
            self.year_imputer_.fit(df[["year"]])
        if "odometer" in df.columns:
            self.odometer_imputer_.fit(df[["odometer"]])
        if "description" in df.columns:
            self.description_imputer_.fit(df[["description"]])
        if "model" in df.columns:
            self.model_imputer_.fit(df[["model"]])
        for col, imp in self.unknown_imputers_.items():
            imp.fit(df[[col]])

        if "model" in df.columns:
            model_counts = df["model"].value_counts(dropna=True)
            self.popular_models_ = set(model_counts[model_counts >= self.model_min_count].index.tolist())
        else:
            self.popular_models_ = set()

        self.state_lat_median_ = {}
        self.state_long_median_ = {}
        if {"state", "lat"}.issubset(df.columns):
            self.state_lat_median_ = df.groupby("state")["lat"].median().dropna().to_dict()
        if {"state", "long"}.issubset(df.columns):
            self.state_long_median_ = df.groupby("state")["long"].median().dropna().to_dict()

        self.global_lat_median_ = self._safe_median(df, "lat", default=0.0)
        self.global_long_median_ = self._safe_median(df, "long", default=0.0)

        self.region_to_cluster_, self.default_region_cluster_ = self._fit_price_cluster_map(
            df=df,
            price=price_for_fit,
            category_col="region",
            n_clusters=self.region_n_clusters,
        )
        self.paint_to_cluster_, self.default_paint_cluster_ = self._fit_price_cluster_map(
            df=df,
            price=price_for_fit,
            category_col="paint_color",
            n_clusters=self.paint_n_clusters,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            [
                "year_imputer_",
                "odometer_imputer_",
                "description_imputer_",
                "model_imputer_",
                "unknown_imputers_",
                "popular_models_",
                "state_lat_median_",
                "state_long_median_",
                "global_lat_median_",
                "global_long_median_",
                "region_to_cluster_",
                "default_region_cluster_",
                "paint_to_cluster_",
                "default_paint_cluster_",
            ],
        )

        df = self._to_frame(X)

        if "year" in df.columns:
            df["year"] = self.year_imputer_.transform(df[["year"]])[:, 0]
            df["car_age"] = 2022 - df["year"]

        if "odometer" in df.columns:
            df["odometer"] = self.odometer_imputer_.transform(df[["odometer"]])[:, 0]
            df["log_odometer"] = np.log1p(df["odometer"])

        if "model" in df.columns:
            df["model"] = self.model_imputer_.transform(df[["model"]])[:, 0]
            df["model"] = df["model"].where(
                df["model"].isin(self.popular_models_) | (df["model"] == "unknown"),
                "other",
            )

        if {"lat", "long"}.issubset(df.columns):
            df["lat_long_missing"] = (df["lat"].isna() | df["long"].isna()).astype(int)
            if "state" in df.columns:
                df["lat"] = df["lat"].fillna(df["state"].map(self.state_lat_median_))
                df["long"] = df["long"].fillna(df["state"].map(self.state_long_median_))
            df["lat"] = df["lat"].fillna(self.global_lat_median_)
            df["long"] = df["long"].fillna(self.global_long_median_)

        if "region" in df.columns:
            region = df["region"].fillna("unknown")
            region_cluster = region.map(self.region_to_cluster_).fillna(self.default_region_cluster_)
            df["region_price_cluster"] = pd.Categorical(region_cluster.astype(int))

        if "paint_color" in df.columns:
            paint = df["paint_color"].fillna("unknown")
            paint_cluster = paint.map(self.paint_to_cluster_).fillna(self.default_paint_cluster_)
            df["paint_color_price_cluster"] = pd.Categorical(paint_cluster.astype(int))

        for col, imp in self.unknown_imputers_.items():
            df[col] = imp.transform(df[[col]])[:, 0]

        if "cylinders" in df.columns:
            df["cylinders_other"] = (df["cylinders"] == "other").astype(int)
            df["cylinders"] = df["cylinders"].map(self.CYLINDERS_MAP).fillna(0).astype(int)

        if "description" in df.columns:
            df["description"] = self.description_imputer_.transform(df[["description"]])[:, 0]

        drop_cols = [
            "posting_date",
            "posting_year",
            "posting_month",
            "state",
            "region",
            "paint_color",
        ]
        drop_cols = [col for col in drop_cols if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    def _fit_price_cluster_map(
        self,
        *,
        df: pd.DataFrame,
        price: pd.Series,
        category_col: str,
        n_clusters: int,
    ) -> tuple[dict[str, int], int]:
        if category_col not in df.columns:
            return {}, 0

        work = df.copy()
        work["price"] = price
        category_price = (
            work.assign(**{category_col: work[category_col].fillna("unknown")})
            .groupby(category_col)["price"]
            .median()
            .reset_index()
        )
        if category_price.empty:
            return {}, 0

        k = min(n_clusters, len(category_price))
        model = KMeans(n_clusters=k, n_init=10)
        category_price["cluster"] = model.fit_predict(category_price[["price"]])

        ordered_clusters = (
            category_price.groupby("cluster")["price"]
            .median()
            .sort_values()
            .index
        )
        order_map = {old: new for new, old in enumerate(ordered_clusters)}
        category_price["cluster"] = category_price["cluster"].map(order_map)

        mapping = dict(zip(category_price[category_col], category_price["cluster"]))
        default_cluster = int(mapping.get("unknown", 0))
        return mapping, default_cluster

    def _extract_price_series(self, *, y: Any, index: pd.Index) -> pd.Series:
        if y is None:
            raise ValueError("CraigslistPreprocessor.fit requires y=price.")

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y must be 1D target or single-column DataFrame.")
            y = y.iloc[:, 0]

        series = pd.Series(y, index=index)
        if len(series) != len(index):
            raise ValueError("Length mismatch: y and X must have the same number of rows.")
        return pd.to_numeric(series, errors="coerce")

    @staticmethod
    def _to_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CraigslistPreprocessor expects pandas.DataFrame input.")
        return X.copy()

    @staticmethod
    def _safe_median(df: pd.DataFrame, col: str, default: float) -> float:
        if col not in df.columns:
            return default
        value = df[col].median()
        if pd.isna(value):
            return default
        return float(value)
