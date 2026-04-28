import numpy as np
import pytest
from src.layers import DenseLayer
from src.activation import tanh_act, tanh_prime
from src.optimizers import SGD, Momentum, Adam


def fresh_layer(seed=0):
    rng = np.random.default_rng(seed)
    return DenseLayer(3, 2, tanh_act, tanh_prime, rng=rng)


def test_sgd_applies_gradient():
    layer = fresh_layer()
    W_before = layer.W.copy()
    b_before = layer.b.copy()
    dW = np.ones_like(layer.W)
    db = np.ones_like(layer.b)

    opt = SGD(lr=0.1)
    opt.step(layer, dW, db)

    np.testing.assert_allclose(layer.W, W_before - 0.1 * dW)
    np.testing.assert_allclose(layer.b, b_before - 0.1 * db)


def test_sgd_no_state():
    opt = SGD(lr=0.01)
    assert not hasattr(opt, "_state") or opt._state == {}  # no buffers


def test_momentum_accumulates_velocity():
    layer = fresh_layer()
    dW = np.ones_like(layer.W)
    db = np.ones_like(layer.b)
    opt = Momentum(lr=0.1, beta=0.9)

    opt.step(layer, dW, db)  # vW = 0.9*0 + 0.1*1 = 0.1
    state = opt._state[id(layer)]
    np.testing.assert_allclose(state["vW"], 0.1 * dW)

    opt.step(layer, dW, db)  # vW = 0.9*0.1 + 0.1 = 0.19
    np.testing.assert_allclose(state["vW"], 0.19 * np.ones_like(dW), rtol=1e-6)


def test_momentum_different_layers_independent():
    layer1 = fresh_layer(0)
    layer2 = fresh_layer(1)
    opt = Momentum(lr=0.1, beta=0.9)
    dW = np.ones_like(layer1.W)
    db = np.ones_like(layer1.b)

    opt.step(layer1, dW, db)
    assert id(layer2) not in opt._state


def test_adam_bias_correction_at_t1():
    layer = fresh_layer()
    dW = np.full_like(layer.W, 1.0)
    db = np.full_like(layer.b, 1.0)
    opt = Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    W_before = layer.W.copy()

    opt.step(layer, dW, db)

    # After t=1: mW_hat = 0.1/(1-0.9) = 1.0, vW_hat = 0.001/(1-0.999) ≈ 1.0
    # update = lr * 1.0 / (sqrt(1.0) + eps) ≈ lr
    expected_update = 0.001 / (np.sqrt(0.001 / (1 - 0.999)) + 1e-8)
    np.testing.assert_allclose(
        W_before - layer.W,
        np.full_like(dW, expected_update),
        rtol=1e-5,
    )


def test_adam_larger_grad_larger_update():
    layer1 = fresh_layer(0)
    layer2 = fresh_layer(0)
    opt1 = Adam(lr=0.001)
    opt2 = Adam(lr=0.001)

    dW_large = np.full_like(layer1.W, 10.0)
    dW_small = np.full_like(layer2.W, 0.01)

    W1_before = layer1.W.copy()
    W2_before = layer2.W.copy()

    opt1.step(layer1, dW_large, np.zeros_like(layer1.b))
    opt2.step(layer2, dW_small, np.zeros_like(layer2.b))

    diff1 = np.abs(W1_before - layer1.W).mean()
    diff2 = np.abs(W2_before - layer2.W).mean()
    assert diff1 > diff2, "Larger gradient should produce larger update for Adam"
