import numpy as np
import pytest
from common.datasets import to_one_hot, xor_dataset


def test_xor_shape():
    X, y = xor_dataset()
    assert X.shape == (4, 2)
    assert y.shape == (4,)


def test_xor_values():
    X, y = xor_dataset()
    for xi, yi in zip(X, y):
        xor_result = 1.0 if xi[0] != xi[1] else -1.0
        assert yi == xor_result


def test_one_hot_zero_one_shape():
    y = np.array([0, 1, 2, 3, 9])
    Y = to_one_hot(y, n_classes=10, encoding="zero_one")
    assert Y.shape == (5, 10)


def test_one_hot_zero_one_values():
    y = np.array([0, 3, 9])
    Y = to_one_hot(y, n_classes=10, encoding="zero_one")
    assert Y[0, 0] == 1.0 and Y[0, 1:].sum() == 0.0
    assert Y[1, 3] == 1.0
    assert Y[2, 9] == 1.0
    assert np.all(np.isin(Y, [0.0, 1.0]))


def test_one_hot_signed_values():
    y = np.array([2])
    Y = to_one_hot(y, n_classes=5, encoding="signed")
    assert Y[0, 2] == 1.0
    assert np.all(Y[0, [0, 1, 3, 4]] == -1.0)


def test_one_hot_signed_sum():
    y = np.arange(10)
    Y = to_one_hot(y, n_classes=10, encoding="signed")
    # each row: one +1, nine -1 → sum = 1 - 9 = -8
    assert np.all(Y.sum(axis=1) == -8.0)
