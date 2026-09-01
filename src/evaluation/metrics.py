from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_binary(y_true, probabilities: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
