from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.pipeline import ID_COLUMN, TARGET


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    required = {TARGET, ID_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("La variable Churn contiene valores fuera de {'Yes', 'No'}.")
    return X, y.astype(int)
