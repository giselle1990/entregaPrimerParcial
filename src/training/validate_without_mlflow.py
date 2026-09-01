"""Valida el pipeline y reproduce las métricas sin requerir MLflow.

Sirve como chequeo rápido del código. La entrega oficial debe ejecutarse con train.py
para generar Runs y Model Registry.
"""
from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load import load_training_data
from src.evaluation.metrics import evaluate_binary
from src.features.pipeline import build_preprocessor

RANDOM_STATE = 42

CONFIGS = [
    ("dummy_most_frequent", DummyClassifier(strategy="most_frequent"), 0.50),
    ("logreg_c0.5_t0.5", LogisticRegression(C=0.5, max_iter=1500, solver="liblinear", random_state=42), 0.50),
    ("logreg_c1_t0.5", LogisticRegression(C=1.0, max_iter=1500, solver="liblinear", random_state=42), 0.50),
    ("logreg_c1_t0.35", LogisticRegression(C=1.0, max_iter=1500, solver="liblinear", random_state=42), 0.35),
    ("rf_200_t0.5", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1), 0.50),
    ("rf_300_depth12_t0.5", RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1), 0.50),
]


def main() -> None:
    X, y = load_training_data("data/raw/customer_churn_historical.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    rows = []
    for name, model, threshold in CONFIGS:
        pipe = Pipeline([("preprocessor", build_preprocessor(list(X.columns))), ("model", model)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({"run_name": name, "threshold": threshold, **evaluate_binary(y_test, proba, threshold).to_dict()})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
