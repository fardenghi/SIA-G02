import numpy as np
import pytest

from src.ej1.data import load_fraud_dataset, normalize_features, train_val_split


def test_load_returns_correct_shapes():
    X, y, labels, feature_names = load_fraud_dataset()
    assert X.ndim == 2
    assert X.shape[1] == len(feature_names)
    assert y.shape == (X.shape[0],)
    assert labels.shape == (X.shape[0],)


def test_load_no_nans():
    X, y, labels, _ = load_fraud_dataset()
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_target_range():
    _, y, _, _ = load_fraud_dataset()
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_normalize_fit():
    X, _, _, _ = load_fraud_dataset()
    X_norm, mean, std = normalize_features(X)
    assert X_norm.shape == X.shape
    np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(X_norm.std(axis=0), 1.0, atol=1e-10)


def test_normalize_transform_uses_provided_stats():
    X, _, _, _ = load_fraud_dataset()
    _, mean, std = normalize_features(X)
    X_norm2, m2, s2 = normalize_features(X, mean=mean, std=std)
    assert m2 is mean and s2 is std
    np.testing.assert_allclose(X_norm2.mean(axis=0), 0.0, atol=1e-10)


def test_train_val_split_sizes():
    X, y, labels, _ = load_fraud_dataset()
    X_tr, y_tr, l_tr, X_v, y_v, l_v = train_val_split(X, y, labels, val_ratio=0.2)
    n = len(X)
    assert len(X_tr) + len(X_v) == n
    assert abs(len(X_v) / n - 0.2) < 0.01


def test_train_val_split_no_overlap():
    X, y, labels, _ = load_fraud_dataset()
    X_tr, y_tr, l_tr, X_v, y_v, l_v = train_val_split(X, y, labels, val_ratio=0.2)
    # Rows in train and val must be disjoint (check via y values as proxy)
    tr_set = set(map(tuple, X_tr.round(8)))
    vl_set = set(map(tuple, X_v.round(8)))
    assert len(tr_set & vl_set) == 0


def test_train_val_split_without_labels():
    X, y, _, _ = load_fraud_dataset()
    X_tr, y_tr, X_v, y_v = train_val_split(X, y, val_ratio=0.2)
    assert len(X_tr) + len(X_v) == len(X)
