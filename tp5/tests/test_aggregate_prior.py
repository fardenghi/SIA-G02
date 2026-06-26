"""Tests del prior empírico sobre el posterior agregado (ex-post density estimation)."""

import numpy as np
import pytest

from autoencoder.aggregate_prior import AggregatePrior, posterior_prior_mismatch


def _blobs(seed=0):
    """Dos clusters separados en 4D (multimodal): caras vs animales, en miniatura."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((60, 4)) * 0.3 + np.array([3.0, 0, 0, 0])
    b = rng.standard_normal((60, 4)) * 0.3 + np.array([-3.0, 0, 0, 0])
    return np.vstack([a, b])


@pytest.mark.parametrize("kind", ["gaussian", "gmm"])
def test_sample_shape(kind):
    mu = _blobs()
    ap = AggregatePrior.fit(mu, kind=kind, k=4)
    z = ap.sample(25, np.random.default_rng(1))
    assert z.shape == (25, 4)
    assert np.isfinite(z).all()


def test_sample_is_deterministic_given_rng():
    mu = _blobs()
    ap = AggregatePrior.fit(mu, kind="gmm", k=4, seed=0)
    z1 = ap.sample(10, np.random.default_rng(7))
    z2 = ap.sample(10, np.random.default_rng(7))
    assert np.allclose(z1, z2)


def test_gaussian_recovers_mean():
    mu = _blobs()
    ap = AggregatePrior.fit(mu, kind="gaussian")
    z = ap.sample(20000, np.random.default_rng(0))
    assert np.allclose(z.mean(axis=0), mu.mean(axis=0), atol=0.1)


def test_gmm_samples_land_near_data_not_origin():
    """El sentido del truco: las muestras del GMM caen en los clusters, no en el hueco central
    (donde N(0,I) pondría casi toda su masa). Acá el origen está vacío entre los dos blobs."""
    mu = _blobs()
    ap = AggregatePrior.fit(mu, kind="gmm", k=4, seed=0)
    z = ap.sample(2000, np.random.default_rng(0))
    # Casi ninguna muestra cae en la zona vacía cerca del origen (|x0| < 1.5).
    frac_origin = np.mean(np.abs(z[:, 0]) < 1.5)
    assert frac_origin < 0.05


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        AggregatePrior.fit(_blobs(), kind="flow")


def test_mismatch_zero_for_standard_normal():
    rng = np.random.default_rng(0)
    mu = rng.standard_normal((5000, 8))
    m = posterior_prior_mismatch(mu)
    assert m["mean_norm"] < 0.2
    assert abs(m["std_mean"] - 1.0) < 0.1
