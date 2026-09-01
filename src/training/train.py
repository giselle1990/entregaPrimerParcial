from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
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
TEST_SIZE = 0.20
MODEL_NAME = "customer-churn-candidate"


def model_candidates() -> list[dict[str, Any]]:
    # Seis runs relevantes: baseline, modelos lineales, árbol y ajuste de threshold.
    return [
        {
            "run_name": "dummy_most_frequent",
            "model": DummyClassifier(strategy="most_frequent"),
            "threshold": 0.50,
            "family": "baseline",
        },
        {
            "run_name": "logreg_c0.5_t0.5",
            "model": LogisticRegression(C=0.5, max_iter=1500, solver="liblinear", random_state=RANDOM_STATE),
            "threshold": 0.50,
            "family": "linear",
        },
        {
            "run_name": "logreg_c1_t0.5",
            "model": LogisticRegression(C=1.0, max_iter=1500, solver="liblinear", random_state=RANDOM_STATE),
            "threshold": 0.50,
            "family": "linear",
        },
        {
            "run_name": "logreg_c1_t0.35",
            "model": LogisticRegression(C=1.0, max_iter=1500, solver="liblinear", random_state=RANDOM_STATE),
            "threshold": 0.35,
            "family": "linear_threshold",
        },
        {
            "run_name": "rf_200_t0.5",
            "model": RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "threshold": 0.50,
            "family": "tree",
        },
        {
            "run_name": "rf_300_depth12_t0.5",
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "threshold": 0.50,
            "family": "tree",
        },
    ]


def configure_mlflow(experiment: str) -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


def run_experiments(data_path: str, experiment: str, register_best: bool) -> pd.DataFrame:
    configure_mlflow(experiment)
    X, y = load_training_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    run_ids: dict[str, str] = {}

    for config in model_candidates():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(list(X.columns))),
                ("model", config["model"]),
            ]
        )

        with mlflow.start_run(run_name=config["run_name"]) as run:
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate_binary(y_test, probabilities, threshold=config["threshold"])

            mlflow.log_params(
                {
                    "family": config["family"],
                    "threshold": config["threshold"],
                    "random_state": RANDOM_STATE,
                    "test_size": TEST_SIZE,
                    "stratified_split": True,
                    **{f"model__{k}": v for k, v in config["model"].get_params(deep=False).items() if isinstance(v, (str, int, float, bool)) or v is None},
                }
            )
            mlflow.log_metrics(metrics.to_dict())
            mlflow.set_tags(
                {
                    "dataset": "customer_churn_historical.csv",
                    "target": "Churn",
                    "business_focus": "reduce_false_negatives",
                    "customerID_used_as_feature": "false",
                }
            )

            input_example = X_train.head(3)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=input_example,
            )

            row = {
                "run_name": config["run_name"],
                "run_id": run.info.run_id,
                "threshold": config["threshold"],
                **metrics.to_dict(),
            }
            results.append(row)
            fitted_models[config["run_name"]] = pipeline
            run_ids[config["run_name"]] = run.info.run_id

    results_df = pd.DataFrame(results)

    # Selección orientada al caso de negocio: se prioriza Recall para reducir falsos negativos;
    # F1 y ROC-AUC se utilizan para evitar una elección desbalanceada.
    eligible = results_df[results_df["run_name"] != "dummy_most_frequent"].copy()
    eligible["selection_score"] = 0.50 * eligible["recall"] + 0.30 * eligible["f1"] + 0.20 * eligible["roc_auc"]
    best = eligible.sort_values(["selection_score", "roc_auc"], ascending=False).iloc[0]
    best_name = str(best["run_name"])

    Path("results").mkdir(exist_ok=True)
    results_df.to_csv("results/experiment_results.csv", index=False)
    with open("results/selected_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": best_name,
                "run_id": run_ids[best_name],
                "threshold": float(best["threshold"]),
                "selection_score": float(best["selection_score"]),
                "criterion": "0.50*recall + 0.30*f1 + 0.20*roc_auc",
            },
            f,
            indent=2,
        )

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": fitted_models[best_name], "threshold": float(best["threshold"]), "run_id": run_ids[best_name]},
        "models/customer_churn_candidate.joblib",
    )

    if register_best:
        model_uri = f"runs:/{run_ids[best_name]}/model"
        registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        print(f"Modelo registrado: {MODEL_NAME}, versión={registered.version}, run_id={run_ids[best_name]}")

    print(results_df.sort_values("recall", ascending=False).to_string(index=False))
    print(f"\nCandidato seleccionado: {best_name} | run_id={run_ids[best_name]}")
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento reproducible del proyecto Customer Churn.")
    parser.add_argument("--data", default="data/raw/customer_churn_historical.csv")
    parser.add_argument("--experiment", default="customer-churn-entrega-1")
    parser.add_argument("--register-best", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args.data, args.experiment, args.register_best)
