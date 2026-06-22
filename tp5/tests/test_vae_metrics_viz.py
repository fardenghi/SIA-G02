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


def test_project_pca_shape():
    mu = np.random.default_rng(2).normal(size=(20, 8))
    proj, var_ratio = vae_metrics_viz.project_pca(mu, k=2)
    assert proj.shape == (20, 2)
    assert var_ratio.shape == (2,)


def test_project_pca_components_ordered_by_variance():
    mu = np.random.default_rng(3).normal(size=(50, 5))
    _, var_ratio = vae_metrics_viz.project_pca(mu, k=5)
    # Componentes ordenados por varianza explicada decreciente, sumando ~1.
    assert np.all(np.diff(var_ratio) <= 1e-12)
    assert var_ratio.sum() == pytest.approx(1.0, rel=1e-9)


def test_project_pca_full_is_invertible():
    rng = np.random.default_rng(4)
    mu = rng.normal(size=(30, 6))
    mean = mu.mean(axis=0)
    centered = mu - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    proj, _ = vae_metrics_viz.project_pca(mu, k=6)
    # Con todos los componentes la reconstrucción recupera μ: proj @ Vt + media == μ.
    recon = proj @ vt + mean
    assert np.allclose(recon, mu, atol=1e-8)


def test_active_units_counts_dims_over_threshold():
    # Caso sintético: dejar μ/logvar conocidos para controlar la KL por dim.
    vae = _vae(latent=4)
    X = _data()

    class _FakeEnc:
        # KL por dim ≈ [0.0, ~grande, 0.0, ~grande] con un μ controlado y σ=1 (logvar=0).
        def __call__(self, _X):
            n = _X.shape[0]
            mu = np.tile(np.array([0.0, 2.0, 0.0, 1.5]), (n, 1))
            logvar = np.zeros((n, 4))
            return mu, logvar

    vae.encode = _FakeEnc()
    # KL = 0.5*mu^2 con logvar=0: dims activas son las de mu != 0 (umbral 0.1).
    assert vae_metrics_viz.active_units(vae, X, threshold=0.1) == 2


def test_plot_kl_per_dim_high_latent_creates_file(tmp_path):
    p = tmp_path / "kld_hi.png"
    vae_metrics_viz.plot_kl_per_dim(_vae(latent=16), _data(), path=p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_latent_sweep_creates_file(tmp_path):
    df = pd.DataFrame({"latent_dim": [2, 4, 8, 16, 32],
                       "recon": [320, 280, 250, 240, 238],
                       "active_units": [2, 4, 6, 7, 7]})
    p = tmp_path / "latent_sweep.png"
    vae_metrics_viz.plot_latent_sweep(df, path=p)
    assert p.exists() and p.stat().st_size > 0


def test_ssim_identical_is_one():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(5, DIM))
    s = vae_metrics_viz.ssim(X, X, SIZE)
    assert s.shape == (5,)
    assert np.allclose(s, 1.0, atol=1e-6)


def test_ssim_degrades_with_noise():
    rng = np.random.default_rng(8)
    X = rng.uniform(0, 1, size=(5, DIM))
    noisy = np.clip(X + rng.normal(0, 0.3, size=X.shape), 0, 1)
    # El SSIM del par ruidoso es claramente menor que el del par idéntico (=1).
    assert vae_metrics_viz.ssim(noisy, X, SIZE).mean() < 0.99


def test_plot_sample_neighbors_creates_file(tmp_path):
    rng = np.random.default_rng(9)
    samples = rng.uniform(0, 1, size=(4, DIM))
    train = rng.uniform(0, 1, size=(12, DIM))
    p = tmp_path / "nn.png"
    vae_metrics_viz.plot_sample_neighbors(samples, train, SIZE, path=p)
    assert p.exists() and p.stat().st_size > 0


def test_as_image_infers_channels():
    flat_gray = np.zeros(DIM)
    flat_rgb = np.zeros(DIM * 3)
    assert vae_metrics_viz.as_image(flat_gray, SIZE).shape == (SIZE, SIZE)
    assert vae_metrics_viz.as_image(flat_rgb, SIZE).shape == (SIZE, SIZE, 3)


def test_ssim_color_identical_is_one():
    rng = np.random.default_rng(11)
    X = rng.uniform(0, 1, size=(5, DIM * 3))   # 3 canales (RGB) aplanados H·W·C
    s = vae_metrics_viz.ssim(X, X, SIZE)
    assert s.shape == (5,)
    assert np.allclose(s, 1.0, atol=1e-6)


def test_plot_sample_neighbors_color_creates_file(tmp_path):
    rng = np.random.default_rng(12)
    samples = rng.uniform(0, 1, size=(3, DIM * 3))
    train = rng.uniform(0, 1, size=(8, DIM * 3))
    p = tmp_path / "nn_color.png"
    vae_metrics_viz.plot_sample_neighbors(samples, train, SIZE, path=p)
    assert p.exists() and p.stat().st_size > 0
