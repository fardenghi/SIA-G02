"""Tests de las capas espaciales (Fase 2.1): shapes, gradient-check numérico y casos a mano.

El gradient-check es el gate de correctitud de toda la matemática nueva: compara el backward
analítico (dx, dW, db) contra diferencias finitas centrales (diff relativa < 1e-5).
"""

import numpy as np
import pytest

from autoencoder.conv import Conv2D, Flatten, Reshape, Upsample2D, conv_out_size


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / (np.max(np.abs(a) + np.abs(b)) + 1e-12)


def _numgrad(loss, x, eps=1e-6):
    """Gradiente numérico de `loss()` respecto a `x` (mutado in-place), diferencia central."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = loss()
        x[idx] = orig - eps
        fm = loss()
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


# ------------------------------------------------------------------ forward shapes


@pytest.mark.parametrize("h,k,s,p,expected", [
    (28, 3, 2, 1, 14), (14, 3, 2, 1, 7), (8, 3, 2, 1, 4),
    (5, 3, 1, 1, 5), (5, 3, 1, 0, 3), (7, 1, 1, 0, 7),
])
def test_conv_out_size(h, k, s, p, expected):
    assert conv_out_size(h, k, s, p) == expected


def test_conv_forward_shape_stride2():
    rng = np.random.default_rng(0)
    conv = Conv2D(1, 16, kernel=3, stride=2, padding=1, activation="relu", rng=rng)
    x = rng.standard_normal((4, 1, 28, 28))
    assert conv.forward(x).shape == (4, 16, 14, 14)


def test_conv_forward_shape_same():
    rng = np.random.default_rng(0)
    conv = Conv2D(16, 32, kernel=3, stride=1, padding=1, activation="relu", rng=rng)
    x = rng.standard_normal((2, 16, 7, 7))
    assert conv.forward(x).shape == (2, 32, 7, 7)


# ------------------------------------------------------------------ gradient-check


def _conv_gradcheck(stride, padding):
    rng = np.random.default_rng(1)
    conv = Conv2D(2, 3, kernel=3, stride=stride, padding=padding, activation="linear", rng=rng)
    x = rng.standard_normal((2, 2, 5, 5))
    g = rng.standard_normal(conv.forward(x).shape)  # gradiente upstream fijo

    def loss():
        return float((conv.forward(x) * g).sum())

    conv.forward(x)
    dx = conv.backward(g)
    dW_ana, db_ana = conv.dW.copy(), conv.db.copy()

    assert _rel(dx, _numgrad(loss, x)) < 1e-5
    assert _rel(dW_ana, _numgrad(loss, conv.W)) < 1e-5
    assert _rel(db_ana, _numgrad(loss, conv.b)) < 1e-5


def test_conv_gradcheck_stride1_pad1():
    _conv_gradcheck(stride=1, padding=1)


def test_conv_gradcheck_stride2_pad1():
    _conv_gradcheck(stride=2, padding=1)


def test_conv_gradcheck_no_pad():
    _conv_gradcheck(stride=1, padding=0)


def test_conv_identity_filter():
    # Filtro 1×1 con peso 1 y bias 0, activación lineal: la salida iguala la entrada.
    rng = np.random.default_rng(2)
    conv = Conv2D(1, 1, kernel=1, stride=1, padding=0, activation="linear", rng=rng)
    conv.W = np.ones((1, 1, 1, 1))
    conv.b = np.zeros(1)
    x = rng.standard_normal((3, 1, 6, 6))
    assert np.allclose(conv.forward(x), x)


def test_upsample_forward():
    x = np.arange(4, dtype=float).reshape(1, 1, 2, 2)
    up = Upsample2D(scale=2)
    y = up.forward(x)
    assert y.shape == (1, 1, 4, 4)
    # Cada píxel repetido en un bloque 2×2.
    assert np.array_equal(y[0, 0, :2, :2], np.full((2, 2), x[0, 0, 0, 0]))
    assert np.array_equal(y[0, 0, 2:, 2:], np.full((2, 2), x[0, 0, 1, 1]))


def test_upsample_gradcheck():
    rng = np.random.default_rng(3)
    up = Upsample2D(scale=2)
    x = rng.standard_normal((2, 3, 4, 4))
    g = rng.standard_normal(up.forward(x).shape)

    def loss():
        return float((up.forward(x) * g).sum())

    up.forward(x)
    dx = up.backward(g)
    assert _rel(dx, _numgrad(loss, x)) < 1e-6


def test_upsample_backward_is_block_sum():
    # backward debe sumar cada bloque scale×scale (transpuesta del nearest-upsample).
    up = Upsample2D(scale=2)
    x = np.zeros((1, 1, 2, 2))
    up.forward(x)
    dout = np.ones((1, 1, 4, 4))
    dx = up.backward(dout)
    assert np.array_equal(dx, np.full((1, 1, 2, 2), 4.0))


def test_flatten_roundtrip_and_gradcheck():
    rng = np.random.default_rng(4)
    flat = Flatten()
    x = rng.standard_normal((2, 3, 4, 4))
    y = flat.forward(x)
    assert y.shape == (2, 3 * 4 * 4)
    # backward restaura la forma exacta.
    assert flat.backward(y).shape == x.shape
    g = rng.standard_normal(y.shape)

    def loss():
        return float((flat.forward(x) * g).sum())

    flat.forward(x)
    assert _rel(flat.backward(g), _numgrad(loss, x)) < 1e-9


def test_reshape_roundtrip():
    rng = np.random.default_rng(5)
    resh = Reshape((32, 7, 7))
    x = rng.standard_normal((4, 32 * 7 * 7))
    y = resh.forward(x)
    assert y.shape == (4, 32, 7, 7)
    assert np.array_equal(resh.backward(y), x)


def test_conv_pack_unpack_roundtrip():
    # W/b planos para pack/unpack (mismo patrón que Dense).
    rng = np.random.default_rng(6)
    conv = Conv2D(2, 4, kernel=3, stride=1, padding=1, rng=rng)
    flat = np.concatenate([conv.W.reshape(-1), conv.b.reshape(-1)])
    assert flat.size == conv.n_params
    conv2 = Conv2D(2, 4, kernel=3, stride=1, padding=1, rng=np.random.default_rng(99))
    conv2.W = flat[:conv.W.size].reshape(conv.W.shape)
    conv2.b = flat[conv.W.size:].reshape(conv.b.shape)
    assert np.array_equal(conv2.W, conv.W) and np.array_equal(conv2.b, conv.b)
