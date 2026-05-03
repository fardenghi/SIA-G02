from pathlib import Path

import numpy as np
import pandas as pd

from common.datasets import k_fold_indices

_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _ROOT / "data" / "ej1_fraud"

FEATURE_COLS = [
    "timestamp",
    "amount_usd",
    "quantity_purchased",
    "session_duration_seconds",
    "days_since_last_purchase",
    "account_age_days",
    "device_screen_resolution",
    "time_since_last_login_s",
    "items_viewed_before_purchase",
]
TARGET_COL = "big_model_fraud_probability"
LABEL_COL = "flagged_fraud"


def load_fraud_dataset(path=None):
    """Load fraud_dataset.csv; returns (X, y, labels, feature_names).

    X: (N, 8) float64 – raw (unnormalized) features
    y: (N,) float64 – continuous fraud probability [0, 1]
    labels: (N,) int – binary flagged_fraud column
    """
    if path is None:
        path = _DATA_DIR / "fraud_dataset.csv"
    df = pd.read_csv(path)
    df = df.dropna()
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    labels = df[LABEL_COL].to_numpy(dtype=np.int32)
    return X, y, labels, FEATURE_COLS


def normalize_features(X, mean=None, std=None):
    """Z-score normalization.

    If mean/std are None, compute from X (fit mode).
    Returns (X_norm, mean, std).
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std = np.where(std == 0, 1.0, std)  # avoid division by zero
    return (X - mean) / std, mean, std


def train_val_split(X, y, labels=None, val_ratio=0.2, seed=42):
    """Random 80/20 split.  Returns train/val arrays."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if labels is not None:
        return (
            X[train_idx], y[train_idx], labels[train_idx],
            X[val_idx],   y[val_idx],   labels[val_idx],
        )
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def k_fold_split(X, y, labels, k=5, seed=42):
    """Yields (X_tr, y_tr, l_tr, X_v, y_v, l_v, val_idx) for each fold."""
    for train_idx, val_idx in k_fold_indices(len(X), k=k, seed=seed):
        yield (
            X[train_idx], y[train_idx], labels[train_idx],
            X[val_idx],   y[val_idx],   labels[val_idx],
            val_idx,
        )
