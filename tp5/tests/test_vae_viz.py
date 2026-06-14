"""Tests de las visualizaciones del VAE (Fase 4): smoke de generación de figuras."""

import numpy as np
import pytest

from autoencoder import vae_viz
from autoencoder.vae import VAE

SIZE = 8
DIM = SIZE * SIZE


def _vae(latent=2, seed=0):
    return VAE([DIM, 16], latent_dim=latent, activation="relu",
               output_activation="sigmoid", init="he_normal", seed=seed)


def _data(n=6):
    return np.random.default_rng(1).uniform(0, 1, size=(n, DIM))


def test_plot_image_grid_creates_file(tmp_path):
    p = tmp_path / "grid.png"
    vae_viz.plot_image_grid(_data(8), SIZE, ncols=4, path=p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_latent_means_creates_file(tmp_path):
    vae = _vae()
    mu, _ = vae.encode(_data())
    p = tmp_path / "means.png"
    vae_viz.plot_latent_means(mu, [f"e{i}" for i in range(mu.shape[0])], path=p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_manifold_creates_file(tmp_path):
    p = tmp_path / "manifold.png"
    vae_viz.plot_latent_manifold(_vae(latent=2), SIZE, n=5, path=p)
    assert p.exists() and p.stat().st_size > 0


def test_manifold_requires_2d_latent():
    with pytest.raises(ValueError):
        vae_viz.plot_latent_manifold(_vae(latent=3), SIZE, n=3)


def test_plot_samples_creates_file(tmp_path):
    p = tmp_path / "samples.png"
    vae_viz.plot_samples(_vae(), SIZE, n=9, rng=np.random.default_rng(0), path=p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_interpolation_creates_file(tmp_path):
    vae = _vae()
    X = _data()
    p = tmp_path / "interp.png"
    vae_viz.plot_interpolation(vae, X[0], X[1], SIZE, steps=6, path=p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_reconstruction_creates_file(tmp_path):
    vae = _vae()
    X = _data()
    mu, _ = vae.encode(X)
    x_hat = vae.decode(mu)
    p = tmp_path / "recon.png"
    vae_viz.plot_reconstruction_gray(X, x_hat, [f"e{i}" for i in range(X.shape[0])],
                                     size=SIZE, path=p)
    assert p.exists() and p.stat().st_size > 0
