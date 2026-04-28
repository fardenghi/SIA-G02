import numpy as np
import pytest
from common.losses import cross_entropy, cross_entropy_softmax_grad, mse, mse_grad


def test_mse_zero():
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert mse(y, y) == pytest.approx(0.0)


def test_mse_nonzero():
    y_true = np.array([[1.0]])
    y_pred = np.array([[0.0]])
    assert mse(y_true, y_pred) == pytest.approx(0.5)


def test_mse_grad_shape():
    y_true = np.zeros((4, 3))
    y_pred = np.ones((4, 3))
    g = mse_grad(y_true, y_pred)
    assert g.shape == (4, 3)


def test_mse_grad_formula():
    y_true = np.zeros((2, 2))
    y_pred = np.ones((2, 2))
    g = mse_grad(y_true, y_pred)
    np.testing.assert_allclose(g, np.full((2, 2), 0.5))


def test_cross_entropy_correct_low():
    # perfect prediction
    y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    y_pred = np.array([[0.99, 0.005, 0.005], [0.005, 0.99, 0.005]])
    assert cross_entropy(y_true, y_pred) < 0.05


def test_cross_entropy_wrong_high():
    y_true = np.array([[1, 0, 0]], dtype=float)
    y_pred = np.array([[0.01, 0.495, 0.495]])
    assert cross_entropy(y_true, y_pred) > 3.0


def test_cross_entropy_softmax_grad_shape():
    y_true = np.zeros((5, 4))
    y_pred = np.ones((5, 4)) / 4
    g = cross_entropy_softmax_grad(y_true, y_pred)
    assert g.shape == (5, 4)


def test_cross_entropy_softmax_grad_formula():
    y_true = np.array([[1, 0], [0, 1]], dtype=float)
    y_pred = np.array([[0.7, 0.3], [0.4, 0.6]])
    g = cross_entropy_softmax_grad(y_true, y_pred)
    expected = (y_pred - y_true) / 2
    np.testing.assert_allclose(g, expected)
