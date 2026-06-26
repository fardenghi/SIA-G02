"""Tests de la ConvVAE (Fase 2.2): shapes, gradient-check del ELBO completo y descenso.

El gradient-check del ELBO es el gate de correctitud (igual que con la MLP-VAE): valida la
orquestación nueva (decoder/encoder convolucional + split de la reparametrización + KL en las
cabezas) comparando `get_grads()` analítico contra diferencias finitas sobre `get_params()`.
"""

import numpy as np
import pytest

from autoencoder.conv_vae import ConvVAE
from autoencoder.vae_train import train_vae

SIZE = 8
DIM = SIZE * SIZE


def _convvae(latent=3, loss="bce", seed=0):
    return ConvVAE(size=SIZE, latent_dim=latent, conv_channels=[4, 8], dense_hidden=16,
                   activation="relu", output_activation="sigmoid", init="he_normal",
                   loss=loss, seed=seed)


def _data(n=4, seed=1):
    return np.random.default_rng(seed).uniform(0, 1, size=(n, DIM))


def test_encode_decode_forward_shapes():
    vae = _convvae(latent=5)
    X = _data(n=6)
    mu, logvar = vae.encode(X)
    assert mu.shape == (6, 5) and logvar.shape == (6, 5)
    z = vae.sample_prior(6)
    assert vae.decode(z).shape == (6, DIM)
    x_hat, mu, logvar, z = vae.forward(X, eps=np.zeros((6, 5)))
    assert x_hat.shape == (6, DIM) and z.shape == (6, 5)


def test_spatial_bookkeeping():
    # 28 → 14 → 7 (encoder) y 7 → 14 → 28 (decoder).
    vae = ConvVAE(size=28, latent_dim=2, conv_channels=[16, 32], dense_hidden=64, seed=0)
    assert vae.feat_h == 7 and vae.feat_ch == 32
    X = np.random.default_rng(0).uniform(0, 1, size=(3, 28 * 28))
    assert vae.decode(vae.sample_prior(3)).shape == (3, 28 * 28)
    assert vae.forward(X, eps=np.zeros((3, 2)))[0].shape == (3, 28 * 28)


def test_incompatible_size_raises():
    with pytest.raises(ValueError):
        ConvVAE(size=10, latent_dim=2, conv_channels=[16, 32])


def test_pack_unpack_roundtrip():
    vae = _convvae()
    p = vae.get_params()
    assert p.size == vae.n_params
    vae2 = _convvae(seed=123)
    vae2.set_params(p)
    assert np.allclose(vae2.get_params(), p)


def _rel_norm(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)


def _elbo_gradcheck(loss, beta=1.0, fd=1e-5):
    vae = _convvae(latent=3, loss=loss)
    X = _data(n=3)
    eps = np.random.default_rng(7).standard_normal((3, 3))  # ε fijo → ELBO determinista

    p0 = vae.get_params()
    vae.set_params(p0)
    vae.forward(X, eps=eps)
    vae.backward(beta)
    g_ana = vae.get_grads()

    def loss_at(p):
        vae.set_params(p)
        vae.forward(X, eps=eps)
        return vae.elbo(beta)[0]

    g_num = np.zeros_like(p0)
    for i in range(p0.size):
        pp = p0.copy(); pp[i] += fd; fp = loss_at(pp)
        pp = p0.copy(); pp[i] -= fd; fm = loss_at(pp)
        g_num[i] = (fp - fm) / (2 * fd)
    return _rel_norm(g_ana, g_num)


def test_elbo_gradcheck_bce():
    assert _elbo_gradcheck("bce") < 1e-5


def test_elbo_gradcheck_mse():
    assert _elbo_gradcheck("mse") < 1e-5


def test_elbo_decreases_with_adam():
    vae = _convvae(latent=4)
    X = _data(n=8)
    init = train_vae(vae, X, epochs=1, lr=1e-3, beta=1.0, seed=0)["elbo"]
    final = train_vae(vae, X, epochs=300, lr=1e-3, beta=1.0, seed=0)["elbo"]
    assert final < init
