import numpy as np

from autoencoder import losses
from autoencoder.network import Autoencoder
from autoencoder.optim import Adam
from autoencoder.train import max_pixel_error, pixel_errors, train_once


def test_adam_decreases_loss():
    rng = np.random.default_rng(0)
    ae = Autoencoder([6, 4, 2], activation="tanh", output_activation="sigmoid",
                     init="xavier_normal", seed=0)
    X = (rng.random((8, 6)) > 0.5).astype(float)
    opt = Adam(lr=1e-2)
    params = ae.get_params()
    first = None
    last = None
    for epoch in range(200):
        ae.set_params(params)
        out = ae.forward(X)
        loss = losses.bce_value(out, X)
        ae.backward(losses.bce_grad(out, X))
        params = opt.step(params, ae.get_grads())
        if epoch == 0:
            first = loss
        last = loss
    assert last < first


def test_pixel_error_threshold():
    y_true = np.array([[1.0, 0.0, 1.0, 0.0]])
    y_pred = np.array([[0.6, 0.4, 0.9, 0.45]])  # todos correctos al umbral 0.5
    assert pixel_errors(y_true, y_pred).tolist() == [0]
    y_pred2 = np.array([[0.4, 0.4, 0.9, 0.45]])  # primer píxel mal
    assert pixel_errors(y_true, y_pred2).tolist() == [1]


def test_train_once_reduces_pixel_error():
    from autoencoder.data import load_font

    X = load_font("font/font.h")[:8]
    ae = Autoencoder([35, 20, 2], seed=3)
    final = train_once(ae, X, X, loss="bce", optimizer="adam", epochs=500, lr=5e-3)
    assert final["max_pixel_error"] <= max_pixel_error(X, np.full_like(X, 0.5))
