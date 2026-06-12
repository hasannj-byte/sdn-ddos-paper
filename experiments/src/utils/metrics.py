"""Evaluation metrics reported on the natural (imbalanced) test distribution.

This is the metric set that Table `tab:detection_results` expects: accuracy,
precision, recall, F1, PR-AUC, MCC, FPR -- plus model size and inference latency.
PR-AUC and MCC are included precisely because accuracy is misleading under class
imbalance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class DetectionMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    mcc: float
    fpr: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    n_params: int | None = None
    latency_ms: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> DetectionMetrics:
    """All detection metrics from ground truth and predicted probabilities."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # PR-AUC / ROC-AUC need both classes present.
    both = len(np.unique(y_true)) == 2
    return DetectionMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        pr_auc=float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        fpr=float(fpr),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def measure_latency_ms(predict_fn, X, n_warmup: int = 10, n_iter: int = 100) -> float:
    """Mean per-sample inference latency in milliseconds (single-sample path)."""
    sample = X[:1]
    for _ in range(n_warmup):
        predict_fn(sample)
    start = time.perf_counter()
    for _ in range(n_iter):
        predict_fn(sample)
    elapsed = time.perf_counter() - start
    return (elapsed / n_iter) * 1000.0


def forgetting_measure(acc_after: dict, acc_peak: dict) -> float:
    """Average forgetting across tasks: mean(peak_acc - final_acc) over tasks.

    acc_peak[t]  = best accuracy ever reached on family t
    acc_after[t] = accuracy on family t at the end of the stream
    """
    keys = [k for k in acc_peak if k in acc_after]
    if not keys:
        return float("nan")
    return float(np.mean([acc_peak[k] - acc_after[k] for k in keys]))
