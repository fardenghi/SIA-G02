"""Tests de los gráficos de diagnóstico del VAE (smoke + propiedad de kl_per_dim)."""

import numpy as np
import pandas as pd
import pytest

from autoencoder import vae_metrics_viz
from autoencoder.vae import VAE

SIZE = 8
DIM = SIZE * SIZE


def _vae(latent=2, seed=0):
    return VAE([DIM, 16], latent_dim=latent, activation="relu",
               output_activation="sigmoid", init="he_normal", seed=seed)


def _data(n=10):
    return np.random.default_rng(1).uniform(0, 1, size=(n, DIM))


def _metrics_df():
    epochs = np.arange(0, 500, 50)
    return pd.DataFrame({
        "run": "t", "epoch": epochs,
        "elbo": np.linspace(300, 100, len(epochs)),
        "recon": np.linspace(290, 95, len(epochs)),
        "kl": np.linspace(0, 6, len(epochs)),
        "beta": np.clip(epochs / 200, 0, 1),
    })


def test_kl_per_dim_sums_to_total_kl():
    vae = _vae()
    X = _data()
    per_dim = vae_metrics_viz.kl_per_dim(vae, X)
    assert per_dim.shape == (2,)
    mu, logvar = vae.encode(X)
    total = vae.kl_divergence(mu, logvar)
    # La suma sobre dimensiones coincide con la KL total (media por muestra).
    assert per_dim.sum() == pytest.approx(total, rel=1e-9)


def test_plot_training_curves_creates_file(tmp_path):
    p = tmp_path / "curves.png"
    vae_metrics_viz.plot_training_curves(_metrics_df(), path=p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_kl_per_dim_creates_file(tmp_path):
    p = tmp_path / "kld.png"
    vae_metrics_viz.plot_kl_per_dim(_vae(), _data(), path=p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_posterior_stats_creates_file(tmp_path):
    p = tmp_path / "post.png"
    vae_metrics_viz.plot_posterior_stats(_vae(), _data(), path=p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_aggregate_posterior_creates_file(tmp_path):
    p = tmp_path / "agg.png"
    vae_metrics_viz.plot_aggregate_posterior(_vae(latent=2), _data(), path=p)
    assert p.exists() and p.stat().st_size > 0


def test_aggregate_posterior_requires_2d():
    with pytest.raises(ValueError):
        vae_metrics_viz.plot_aggregate_posterior(_vae(latent=3), _data())


def test_plot_beta_sweep_creates_file(tmp_path):
    df = pd.DataFrame({"beta": [0.5, 1, 8, 784],
                       "recon": [311, 312, 319, 257], "kl": [7.9, 6.5, 3.2, 0.0]})
    p = tmp_path / "sweep.png"
    vae_metrics_viz.plot_beta_sweep(df, path=p)
    assert p.exists() and p.stat().st_size > 0
