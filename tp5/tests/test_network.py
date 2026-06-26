import numpy as np

from autoencoder import losses
from autoencoder.network import Autoencoder


def test_mirror_decoder_sizes():
    ae = Autoencoder([35, 25, 15, 8, 2], seed=0)
    assert ae.encoder_sizes == [35, 25, 15, 8, 2]
    assert ae.decoder_sizes == [2, 8, 15, 25, 35]


def test_encode_shape_is_latent():
    ae = Autoencoder([35, 25, 15, 8, 2], seed=0)
    X = np.random.default_rng(1).random((10, 35))
    Z = ae.encode(X)
    assert Z.shape == (10, 2)


def test_forward_shape_and_composition():
    ae = Autoencoder([35, 25, 15, 8, 2], seed=0)
    X = np.random.default_rng(1).random((4, 35))
    out = ae.forward(X)
    assert out.shape == (4, 35)
    np.testing.assert_allclose(ae.decode(ae.encode(X)), ae.forward(X))


def test_output_activation_sigmoid_range():
    ae = Autoencoder([35, 10, 2], output_activation="sigmoid", seed=0)
    X = np.random.default_rng(2).random((8, 35))
    out = ae.forward(X)
    assert np.all(out > 0.0) and np.all(out < 1.0)


def test_init_reproducible_with_seed():
    a1 = Autoencoder([35, 10, 2], init="xavier_normal", seed=123)
    a2 = Autoencoder([35, 10, 2], init="xavier_normal", seed=123)
    np.testing.assert_array_equal(a1.get_params(), a2.get_params())


def _numeric_grad(ae, X, Y, loss_value, eps=1e-5):
    theta = ae.get_params().copy()
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        orig = theta[i]
        theta[i] = orig + eps
        ae.set_params(theta)
        lp = loss_value(ae.forward(X), Y)
        theta[i] = orig - eps
        ae.set_params(theta)
        lm = loss_value(ae.forward(X), Y)
        grad[i] = (lp - lm) / (2 * eps)
        theta[i] = orig
    ae.set_params(theta)
    return grad


def test_gradient_check_bce():
    rng = np.random.default_rng(7)
    ae = Autoencoder([6, 4, 2], activation="tanh", output_activation="sigmoid",
                     init="xavier_normal", seed=7)
    X = rng.random((5, 6))
    Y = (rng.random((5, 6)) > 0.5).astype(float)

    out = ae.forward(X)
    ae.backward(losses.bce_grad(out, Y))
    analytic = ae.get_grads()

    numeric = _numeric_grad(ae, X, Y, losses.bce_value)
    rel = np.linalg.norm(analytic - numeric) / (
        np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12
    )
    assert rel < 1e-5, f"diff relativa {rel}"


def test_gradient_check_leaky_relu():
    rng = np.random.default_rng(13)
    ae = Autoencoder([6, 4, 2], activation="leaky_relu", output_activation="sigmoid",
                     init="he_normal", seed=13)
    X = rng.random((5, 6))
    Y = (rng.random((5, 6)) > 0.5).astype(float)

    out = ae.forward(X)
    ae.backward(losses.bce_grad(out, Y))
    analytic = ae.get_grads()

    numeric = _numeric_grad(ae, X, Y, losses.bce_value)
    rel = np.linalg.norm(analytic - numeric) / (
        np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12
    )
    assert rel < 1e-5, f"diff relativa {rel}"


def test_gradient_check_mse():
    rng = np.random.default_rng(11)
    ae = Autoencoder([6, 5, 2], activation="tanh", output_activation="tanh",
                     init="xavier_normal", seed=11)
    X = rng.random((4, 6))
    Y = rng.random((4, 6))

    out = ae.forward(X)
    ae.backward(losses.mse_grad(out, Y))
    analytic = ae.get_grads()

    numeric = _numeric_grad(ae, X, Y, losses.mse_value)
    rel = np.linalg.norm(analytic - numeric) / (
        np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12
    )
    assert rel < 1e-5, f"diff relativa {rel}"


def test_pack_unpack_roundtrip():
    ae = Autoencoder([35, 10, 2], seed=5)
    theta = ae.get_params().copy()
    ae.set_params(theta)
    np.testing.assert_array_equal(ae.get_params(), theta)
