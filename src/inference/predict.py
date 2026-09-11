from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/churn_pipeline.joblib")


def predict(input_path: str, model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    customer_ids = data.get("customerID", pd.Series(data.index, name="customerID"))
    features = data.drop(columns=["customerID", "Churn"], errors="ignore")

    model = joblib.load(model_path)
    probabilities = model.predict_proba(features)[:, 1]

    return pd.DataFrame(
        {
            "customerID": customer_ids,
            "churn_probability": probabilities,
            "prediction": (probabilities >= 0.5).astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción de customer churn.")
    parser.add_argument("input", help="CSV con clientes para predecir")
    parser.add_argument("--output", default="results/predictions.csv")
    args = parser.parse_args()

    result = predict(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predicciones guardadas en {args.output}")


if __name__ == "__main__":
    main()
