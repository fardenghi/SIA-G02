import numpy as np
import pytest
from common.activations import tanh_act, tanh_prime
from common.layers import DenseLayer


def make_layer(n_in=4, n_out=3, seed=0):
    rng = np.random.default_rng(seed)
    return DenseLayer(n_in, n_out, tanh_act, tanh_prime, weight_init="xavier", rng=rng)


def test_shapes_init():
    layer = make_layer(4, 3)
    assert layer.W.shape == (4, 3)
    assert layer.b.shape == (3,)


def test_W_not_zero():
    layer = make_layer(4, 3)
    assert not np.all(layer.W == 0)


def test_b_zero():
    layer = make_layer(4, 3)
    assert np.all(layer.b == 0)


def test_forward_shape():
    layer = make_layer(4, 3)
    X = np.random.randn(8, 4)
    out = layer.forward(X)
    assert out.shape == (8, 3)


def test_forward_deterministic():
    layer = make_layer(4, 3)
    X = np.random.randn(5, 4)
    out1 = layer.forward(X)
    out2 = layer.forward(X)
    np.testing.assert_array_equal(out1, out2)


def test_backward_shapes():
    layer = make_layer(4, 3)
    X = np.random.randn(8, 4)
    layer.forward(X)
    delta = np.random.randn(8, 3)
    dW, db, delta_out = layer.backward(delta, X)
    assert dW.shape == (4, 3)
    assert db.shape == (3,)
    assert delta_out.shape == (8, 4)


def test_backward_dW_formula():
    layer = make_layer(4, 3)
    X = np.random.randn(5, 4)
    layer.forward(X)
    delta = np.random.randn(5, 3)
    dW, db, _ = layer.backward(delta, X)
    np.testing.assert_allclose(dW, X.T @ delta)


def test_backward_db_formula():
    layer = make_layer(4, 3)
    X = np.random.randn(5, 4)
    layer.forward(X)
    delta = np.random.randn(5, 3)
    _, db, _ = layer.backward(delta, X)
    np.testing.assert_allclose(db, delta.sum(axis=0))
