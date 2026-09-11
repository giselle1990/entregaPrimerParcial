import pandas as pd

from src.features.pipeline import build_preprocessor


def test_preprocessor_accepts_missing_and_unknown_values() -> None:
    training = pd.DataFrame(
        {
            "tenure": [1, 24, 60],
            "MonthlyCharges": [30.0, 65.0, 95.0],
            "TotalCharges": [30.0, None, 5700.0],
            "Contract": ["Month-to-month", "One year", "Two year"],
        }
    )
    new_customer = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [70.0],
            "TotalCharges": [None],
            "Contract": ["Nueva categoría"],
        }
    )

    preprocessor = build_preprocessor(training.columns.tolist())
    preprocessor.fit(training)
    transformed = preprocessor.transform(new_customer)

    assert transformed.shape[0] == 1
