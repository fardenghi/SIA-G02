"""Tests del núcleo VAE (Fase 1): shapes, reparametrización, KL y gradient-check del ELBO.

El gradient-check numérico del ELBO completo (recon + β·KL) es la verificación dura de que
la matemática nueva del VAE (reparametrización + KL + su backward) es correcta.
"""

import numpy as np
import pytest

from autoencoder.optim import Adam
from autoencoder.vae import VAE


def _small_vae(loss="bce", output_activation="sigmoid", seed=0):
    # tanh en las ocultas: suave y sin "kinks" -> apto para gradient-check numérico.
    return VAE([6, 5, 4], latent_dim=2, activation="tanh",
               output_activation=output_activation, init="xavier_normal",
               loss=loss, seed=seed)


def _data(n=8, d=6, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.95, size=(n, d))


def _fixed_eps(n=8, latent=2, seed=2):
    return np.random.default_rng(seed).standard_normal((n, latent))


# --------------------------------------------------------------------------- #
# Shapes y composición
# --------------------------------------------------------------------------- #


def test_encode_shapes():
    vae = _small_vae()
    mu, logvar = vae.encode(_data())
    assert mu.shape == (8, 2)
    assert logvar.shape == (8, 2)


def test_forward_shapes():
    vae = _small_vae()
    x_hat, mu, logvar, z = vae.forward(_data(), eps=_fixed_eps())
    assert x_hat.shape == (8, 6)
    assert mu.shape == (8, 2)
    assert logvar.shape == (8, 2)
    assert z.shape == (8, 2)


def test_decoder_mirror_sizes():
    vae = _small_vae()
    assert vae.decoder_sizes == [2, 4, 5, 6]
    assert vae.input_dim == 6
    assert vae.latent_dim == 2


def test_invalid_decoder_raises():
    with pytest.raises(ValueError):
        VAE([6, 4], latent_dim=2, decoder_layers=[3, 6])  # no arranca en latent_dim
    with pytest.raises(ValueError):
        VAE([6, 4], latent_dim=2, decoder_layers=[2, 5])  # no termina en input_dim


# --------------------------------------------------------------------------- #
# Reparametrización
# --------------------------------------------------------------------------- #


def test_reparameterize_deterministic_with_eps():
    mu = np.array([[1.0, -2.0]])
    logvar = np.array([[0.0, 0.5]])
    eps = np.array([[0.3, -0.7]])
    z1 = VAE.reparameterize(mu, logvar, eps)
    z2 = VAE.reparameterize(mu, logvar, eps)
    np.testing.assert_array_equal(z1, z2)


def test_reparameterize_zero_eps_returns_mu():
    mu = np.array([[1.0, -2.0]])
    logvar = np.array([[0.3, 0.5]])
    z = VAE.reparameterize(mu, logvar, np.zeros_like(mu))
    np.testing.assert_allclose(z, mu)


def test_reparameterize_tiny_std_collapses_to_mu():
    mu = np.array([[1.0, -2.0]])
    logvar = np.full_like(mu, -50.0)  # σ ~ 0
    z = VAE.reparameterize(mu, logvar, np.array([[5.0, -5.0]]))
    np.testing.assert_allclose(z, mu, atol=1e-9)


# --------------------------------------------------------------------------- #
# KL divergence
# --------------------------------------------------------------------------- #


def test_kl_zero_at_standard_normal():
    mu = np.zeros((4, 2))
    logvar = np.zeros((4, 2))
    assert VAE.kl_divergence(mu, logvar) == pytest.approx(0.0, abs=1e-12)


def test_kl_positive_off_prior():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(4, 2))
    logvar = rng.normal(size=(4, 2)) * 0.1
    assert VAE.kl_divergence(mu, logvar) > 0.0


def test_kl_grad_matches_numeric():
    # Gradiente analítico de la KL (lo que usa backward): dKL/dμ=μ/N, dKL/dlogσ²=0.5(e^{logσ²}-1)/N.
    rng = np.random.default_rng(3)
    n = 5
    mu = rng.normal(size=(n, 2))
    logvar = rng.normal(size=(n, 2)) * 0.3
    h = 1e-6
    for arr, ana in (
        (mu, mu / n),
        (logvar, 0.5 * (np.exp(logvar) - 1.0) / n),
    ):
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            up = arr.copy(); up[idx] += h
            dn = arr.copy(); dn[idx] -= h
            if arr is mu:
                num[idx] = (VAE.kl_divergence(up, logvar)
                            - VAE.kl_divergence(dn, logvar)) / (2 * h)
            else:
                num[idx] = (VAE.kl_divergence(mu, up)
                            - VAE.kl_divergence(mu, dn)) / (2 * h)
        np.testing.assert_allclose(num, ana, rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------- #
# Gradient-check del ELBO completo (recon + β·KL)
# --------------------------------------------------------------------------- #


def _numeric_grad(vae, X, eps, beta, h=1e-6):
    theta0 = vae.get_params().copy()
    num = np.zeros_like(theta0)
    for i in range(theta0.size):
        up = theta0.copy(); up[i] += h
        vae.set_params(up); vae.forward(X, eps=eps)
        f_up = vae.elbo(beta)[0]
        dn = theta0.copy(); dn[i] -= h
        vae.set_params(dn); vae.forward(X, eps=eps)
        f_dn = vae.elbo(beta)[0]
        num[i] = (f_up - f_dn) / (2 * h)
    vae.set_params(theta0)
    return num


def _analytic_grad(vae, X, eps, beta):
    vae.forward(X, eps=eps)
    vae.backward(beta)
    return vae.get_grads()


@pytest.mark.parametrize("loss,out_act,beta", [
    ("bce", "sigmoid", 1.0),
    ("mse", "linear", 0.5),
    ("bce", "sigmoid", 0.0),   # β=0 aísla la reconstrucción
])
def test_elbo_gradient_check(loss, out_act, beta):
    vae = _small_vae(loss=loss, output_activation=out_act, seed=7)
    X = _data()
    eps = _fixed_eps()
    num = _numeric_grad(vae, X, eps, beta)
    ana = _analytic_grad(vae, X, eps, beta)
    rel = np.linalg.norm(num - ana) / (np.linalg.norm(num) + np.linalg.norm(ana) + 1e-12)
    assert rel < 1e-5, f"gradient-check falló (rel={rel:.2e}, loss={loss}, beta={beta})"


# --------------------------------------------------------------------------- #
# Optimización y utilidades
# --------------------------------------------------------------------------- #


def test_elbo_decreases_with_adam():
    # Objetivo determinista (eps fijo) -> el ELBO debe bajar monótonamente con Adam.
    vae = _small_vae(seed=5)
    X = _data()
    eps = _fixed_eps()
    opt = Adam(lr=1e-2)
    params = vae.get_params()
    first = last = None
    for epoch in range(300):
        vae.set_params(params)
        vae.forward(X, eps=eps)
        total = vae.elbo(1.0)[0]
        vae.backward(1.0)
        params = opt.step(params, vae.get_grads())
        if epoch == 0:
            first = total
        last = total
    assert last < first


def test_params_roundtrip_and_count():
    vae = _small_vae()
    theta = vae.get_params()
    assert theta.size == vae.n_params
    vae.set_params(theta.copy())
    np.testing.assert_array_equal(vae.get_params(), theta)


def test_generate_and_sample_prior_shapes():
    vae = _small_vae()
    z = vae.sample_prior(7, rng=np.random.default_rng(0))
    assert z.shape == (7, 2)
    samples = vae.generate(7, rng=np.random.default_rng(0))
    assert samples.shape == (7, 6)


def test_sample_prior_deterministic_with_seed():
    vae = _small_vae()
    z1 = vae.sample_prior(4, rng=np.random.default_rng(123))
    z2 = vae.sample_prior(4, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(z1, z2)
