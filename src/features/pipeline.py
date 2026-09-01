from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def split_feature_types(columns: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in columns]
    categorical = [c for c in columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric, categorical = split_feature_types(feature_columns)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )
