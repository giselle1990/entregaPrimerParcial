import numpy as np

from src.evaluation.metrics import evaluate_binary


def test_evaluate_binary_returns_expected_confusion_matrix() -> None:
    metrics = evaluate_binary(
        y_true=np.array([0, 0, 1, 1]),
        probabilities=np.array([0.1, 0.8, 0.7, 0.2]),
        threshold=0.5,
    )

    assert (metrics.tn, metrics.fp, metrics.fn, metrics.tp) == (1, 1, 1, 1)
    assert metrics.accuracy == 0.5
