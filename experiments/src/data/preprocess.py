"""Leakage-free preprocessing pipeline.

This is the corrected protocol described in the paper (Section: Data Collection
and Preprocessing). The order of operations is the whole point:

  1. split FIRST (stratified)            -> test set is quarantined
  2. fit the scaler on TRAIN only        -> no test statistics leak into training
  3. resample (SMOTE/undersample) TRAIN only -> no synthetic sample reaches test
  4. the TEST set keeps its natural class ratio

Applying SMOTE before the split (as in the original submission) leaks information
across the partition and inflates the metrics; this module exists to prevent that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def _resample(X, y, cfg):
    mode = cfg["preprocess"]["resample"]
    if mode == "none":
        return X, y
    if mode in ("smote", "smote_under"):
        X, y = SMOTE(
            sampling_strategy=cfg["preprocess"]["smote_sampling_strategy"],
            random_state=cfg["seed"],
        ).fit_resample(X, y)
    if mode == "smote_under":
        X, y = RandomUnderSampler(
            sampling_strategy=cfg["preprocess"]["under_sampling_strategy"],
            random_state=cfg["seed"],
        ).fit_resample(X, y)
    return X, y


def prepare(X: np.ndarray, y: np.ndarray, cfg: dict) -> DataSplits:
    """Run the full leakage-free pipeline and return flat (2D) arrays.

    Use `to_sequences` afterwards to reshape into the 3D form the CNN-LSTM needs.
    """
    seed = cfg["seed"]
    strat = y if cfg["split"]["stratify"] else None

    # 1. split first
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], random_state=seed, stratify=strat
    )
    strat_tr = y_tr if cfg["split"]["stratify"] else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=cfg["split"]["val_size"], random_state=seed, stratify=strat_tr
    )

    # 2. scaler fit on TRAIN only, applied to all partitions
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)

    # 3. resample TRAIN only  (4. test/val keep their natural ratio)
    X_tr, y_tr = _resample(X_tr, y_tr, cfg)

    return DataSplits(
        X_train=X_tr.astype(np.float32), y_train=y_tr.astype(np.int64),
        X_val=X_va.astype(np.float32), y_val=y_va.astype(np.int64),
        X_test=X_te.astype(np.float32), y_test=y_te.astype(np.int64),
        scaler=scaler,
    )


def to_sequences(X: np.ndarray, cfg: dict) -> np.ndarray:
    """Reshape (N, n_features) -> (N, timesteps, channels).

    With 8 features and channels=1 the features are treated as a length-8 sequence.
    """
    t, c = cfg["model"]["timesteps"], cfg["model"]["channels"]
    return X.reshape((X.shape[0], t, c)).astype(np.float32)


def apply_fitted(X: np.ndarray, scaler: StandardScaler, cfg: dict) -> np.ndarray:
    """Scale (with an already-fitted scaler) and reshape -- for foreign datasets."""
    return to_sequences(scaler.transform(X), cfg)
