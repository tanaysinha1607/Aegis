"""Evaluation metrics for binary risk scoring."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC; returns 0.5 if only one class is present."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def summarize_metrics(name: str, auc: float) -> Dict[str, float]:
    return {f"{name}_auc": auc}
