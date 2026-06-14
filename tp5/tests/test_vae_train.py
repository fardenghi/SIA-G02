"""Tests del entrenamiento del VAE (Fase 3): beta_schedule, descenso del ELBO y tracker."""

import numpy as np
import pytest

from autoencoder.vae import VAE
from autoencoder.vae_train import (
    VAEMetricsTracker,
    _eval_recon,
    beta_schedule,
    train_vae,
)


def _synthetic(n=16, d=12, seed=1):
    # Datos estructurados (dos clusters) para que el VAE tenga algo que aprender.
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.9, size=(2, d))
    X = base[rng.integers(0, 2, size=n)] + rng.normal(0, 0.02, size=(n, d))
    return np.clip(X, 0.02, 0.98)


def _small_vae(seed=0, loss="bce", out="sigmoid"):
    return VAE([12, 8], latent_dim=2, activation="tanh", output_activation=out,
               init="xavier_normal", loss=loss, seed=seed)


# --------------------------------------------------------------------------- #
# beta_schedule
# --------------------------------------------------------------------------- #


def test_beta_schedule_no_warmup_constant():
    assert beta_schedule(0, 1.0, 0) == 1.0
    assert beta_schedule(999, 2.5, 0) == 2.5


def test_beta_schedule_linear_warmup():
    assert beta_schedule(0, 1.0, 100) == 0.0
    assert beta_schedule(50, 1.0, 100) == pytest.approx(0.5)
    assert beta_schedule(100, 1.0, 100) == 1.0
    # Clamp: nunca supera beta objetivo.
    assert beta_schedule(500, 1.0, 100) == 1.0


# --------------------------------------------------------------------------- #
# train_vae
# --------------------------------------------------------------------------- #


def test_train_vae_reduces_reconstruction():
    X = _synthetic()
    vae = _small_vae(seed=2)
    before = _eval_recon(vae, X)
    final = train_vae(vae, X, epochs=500, lr=1e-2, beta=0.1, seed=3)
    assert final["recon_det"] < before
    # La KL es una divergencia: siempre no negativa.
    assert final["kl"] >= -1e-9


def test_train_vae_returns_expected_keys():
    X = _synthetic()
    vae = _small_vae(seed=4)
    final = train_vae(vae, X, epochs=50, lr=1e-3, seed=5)
    assert set(final) == {"elbo", "recon", "kl", "recon_det"}


def test_tracker_logs_and_warmup_applied():
    X = _synthetic()
    vae = _small_vae(seed=6)
    tracker = VAEMetricsTracker(run_label="t")
    train_vae(vae, X, epochs=200, lr=1e-3, beta=1.0, beta_warmup=100,
              seed=7, tracker=tracker, log_every=1)
    df = tracker.df
    assert list(df.columns) == ["run", "epoch", "elbo", "recon", "kl", "beta"]
    assert len(df) == 200
    # Warmup: beta arranca en 0 y llega al objetivo tras el warmup.
    assert df.iloc[0]["beta"] == 0.0
    assert df.iloc[-1]["beta"] == 1.0


def test_train_vae_deterministic_with_seed():
    X = _synthetic()
    f1 = train_vae(_small_vae(seed=8), X, epochs=100, lr=1e-3, seed=9)
    f2 = train_vae(_small_vae(seed=8), X, epochs=100, lr=1e-3, seed=9)
    assert f1["elbo"] == pytest.approx(f2["elbo"])
    assert f1["recon_det"] == pytest.approx(f2["recon_det"])
