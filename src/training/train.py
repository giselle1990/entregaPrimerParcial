from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.data.load import load_training_data
from src.evaluation.metrics import evaluate_binary
from src.features.pipeline import build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25  # 25% del 80% restante: 60/20/20 final.
MODEL_NAME = "customer-churn-candidate"
MODEL_PATH = Path("models/churn_pipeline.joblib")


def mlflow_input_example(features: pd.DataFrame, rows: int = 3) -> pd.DataFrame:
    """Build an MLflow example whose numeric schema accepts missing values."""
    example = features.head(rows).copy()
    integer_columns = example.select_dtypes(include=["integer"]).columns
    example[integer_columns] = example[integer_columns].astype("float64")
    return example


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def git_is_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return True


def run_experiments(data_path: str, experiment: str, register_best: bool) -> pd.DataFrame:
    configure_mlflow(experiment)
    X, y = load_training_data(data_path)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    results: list[dict[str, Any]] = []
    dataset_hash = file_sha256(data_path)
    commit = git_commit()
    dirty_worktree = git_is_dirty()

    for config in model_candidates():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(list(X.columns))),
                ("model", config["model"]),
            ]
        )

        with mlflow.start_run(run_name=config["run_name"]) as run:
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_val)[:, 1]
            metrics = evaluate_binary(y_val, probabilities, threshold=config["threshold"])

            mlflow.log_params(
                {
                    "family": config["family"],
                    "threshold": config["threshold"],
                    "random_state": RANDOM_STATE,
                    "test_size": TEST_SIZE,
                    "validation_size_within_train": VALIDATION_SIZE,
                    "stratified_split": True,
                    **{f"model__{k}": v for k, v in config["model"].get_params(deep=False).items() if isinstance(v, (str, int, float, bool)) or v is None},
                }
            )
            mlflow.log_metrics({f"val_{key}": value for key, value in metrics.to_dict().items()})
            mlflow.set_tags(
                {
                    "dataset": "customer_churn_historical.csv",
                    "dataset_sha256": dataset_hash,
                    "git_commit": commit,
                    "git_worktree_dirty": str(dirty_worktree).lower(),
                    "target": "Churn",
                    "business_focus": "reduce_false_negatives",
                    "customerID_used_as_feature": "false",
                }
            )

            input_example = mlflow_input_example(X_train)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=input_example,
                signature=infer_signature(input_example),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            row = {
                "run_name": config["run_name"],
                "run_id": run.info.run_id,
                "evaluation_split": "validation",
                "threshold": config["threshold"],
                **metrics.to_dict(),
            }
            results.append(row)

    results_df = pd.DataFrame(results)

    # Selección orientada al caso de negocio: se prioriza Recall para reducir falsos negativos;
    # F1 y ROC-AUC se utilizan para evitar una elección desbalanceada.
    eligible = results_df[results_df["run_name"] != "dummy_most_frequent"].copy()
    eligible["selection_score"] = 0.50 * eligible["recall"] + 0.30 * eligible["f1"] + 0.20 * eligible["roc_auc"]
    best = eligible.sort_values(["selection_score", "roc_auc"], ascending=False).iloc[0]
    best_name = str(best["run_name"])
    best_config = next(config for config in model_candidates() if config["run_name"] == best_name)

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(list(X.columns))),
            ("model", clone(best_config["model"])),
        ]
    )

    with mlflow.start_run(run_name=f"final_{best_name}") as final_run:
        final_pipeline.fit(X_train_val, y_train_val)
        test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_binary(
            y_test, test_probabilities, threshold=float(best["threshold"])
        )
        final_run_id = final_run.info.run_id

        mlflow.log_params(
            {
                "selected_candidate": best_name,
                "threshold": float(best["threshold"]),
                "selection_criterion": "0.50*recall + 0.30*f1 + 0.20*roc_auc",
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                **{
                    f"model__{key}": value
                    for key, value in best_config["model"].get_params(deep=False).items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            }
        )
        mlflow.log_metrics(
            {f"test_{key}": value for key, value in test_metrics.to_dict().items()}
        )
        mlflow.set_tags(
            {
                "stage": "final_candidate",
                "dataset": "customer_churn_historical.csv",
                "dataset_sha256": dataset_hash,
                "git_commit": commit,
                "git_worktree_dirty": str(dirty_worktree).lower(),
                "selected_on": "validation",
                "customerID_used_as_feature": "false",
            }
        )
        final_input_example = mlflow_input_example(X_train_val)
        final_model_info = mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="model",
            input_example=final_input_example,
            signature=infer_signature(final_input_example),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)

    output_dir = Path("results/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "experiment_results.csv", index=False)
    with open(output_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics.to_dict(), f, indent=2)
    with open(output_dir / "selected_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": best_name,
                "registered_model_name": MODEL_NAME,
                "run_id": final_run_id,
                "threshold": float(best["threshold"]),
                "selection_score": float(best["selection_score"]),
                "criterion": "0.50*recall + 0.30*f1 + 0.20*roc_auc",
                "selected_on": "validation",
                "final_evaluation": "test",
                "dataset_sha256": dataset_hash,
                "git_commit": commit,
                "git_worktree_dirty": dirty_worktree,
            },
            f,
            indent=2,
        )

    if register_best:
        registered = mlflow.register_model(
            model_uri=final_model_info.model_uri,
            name=MODEL_NAME,
        )
        print(f"Modelo registrado: {MODEL_NAME}, versión={registered.version}, run_id={final_run_id}")

    print("Resultados de validación:")
    print(results_df.sort_values("recall", ascending=False).to_string(index=False))
    print("\nEvaluación final sobre test aislado:")
    print(pd.Series(test_metrics.to_dict()).to_string())
    print(f"\nCandidato seleccionado: {best_name} | run_id final={final_run_id}")
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
