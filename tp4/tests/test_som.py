import numpy as np
import pytest
from kohonen.som import SOM


# --- helpers ---

def make_som(rows=3, cols=3, dim=4, **kwargs):
    defaults = dict(lr=0.5, lr_decay="exponential", radius=1.5,
                    radius_decay="exponential", neighborhood_fn="gaussian",
                    epochs=10, seed=42)
    defaults.update(kwargs)
    return SOM(rows, cols, dim, **defaults)


# --- initialization ---

def test_weights_shape():
    som = make_som(rows=3, cols=4, dim=5)
    assert som.weights.shape == (3, 4, 5)


def test_weights_reproducible_with_seed():
    s1 = make_som(seed=7)
    s2 = make_som(seed=7)
    np.testing.assert_array_equal(s1.weights, s2.weights)


def test_weights_differ_across_seeds():
    s1 = make_som(seed=1)
    s2 = make_som(seed=2)
    assert not np.allclose(s1.weights, s2.weights)


# --- BMU ---

def test_find_bmu_exact_match():
    som = make_som(rows=3, cols=3, dim=3)
    som.weights[1, 2] = np.array([1.0, 2.0, 3.0])
    bmu = som._find_bmu(np.array([1.0, 2.0, 3.0]))
    assert bmu == (1, 2)


def test_find_bmu_within_grid():
    som = make_som(rows=4, cols=5, dim=6)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.standard_normal(6)
        r, c = som._find_bmu(x)
        assert 0 <= r < 4 and 0 <= c < 5


# --- neighborhood ---

def test_gaussian_peak_at_bmu():
    som = make_som(rows=5, cols=5, dim=2)
    bmu = (2, 2)
    h = som._neighborhood(bmu, sigma=1.5, mode="gaussian")
    assert h.shape == (5, 5)
    assert h[2, 2] == pytest.approx(1.0)


def test_gaussian_decreases_with_distance():
    som = make_som(rows=5, cols=5, dim=2)
    bmu = (2, 2)
    h = som._neighborhood(bmu, sigma=1.5, mode="gaussian")
    assert h[2, 2] > h[2, 3] > h[2, 4]


def test_bubble_uniform_within_radius():
    som = make_som(rows=5, cols=5, dim=2)
    bmu = (2, 2)
    h = som._bubble(bmu, radius=1.5)
    # All cells with grid dist <= 1 should be 1.0, center included
    assert h[2, 2] == pytest.approx(1.0)
    assert h[2, 3] == pytest.approx(1.0)
    assert h[1, 2] == pytest.approx(1.0)


def test_bubble_zero_outside_radius():
    som = make_som(rows=5, cols=5, dim=2)
    bmu = (2, 2)
    h = som._bubble(bmu, radius=0.9)
    # Only center within radius < 1
    assert h[2, 2] == pytest.approx(1.0)
    assert h[2, 3] == pytest.approx(0.0)


# --- decay ---

def test_exponential_decay_at_t0():
    som = make_som()
    val = som._decay(1.0, t=0, T=100, mode="exponential")
    assert val == pytest.approx(1.0, rel=1e-3)


def test_exponential_decay_decreases():
    som = make_som()
    v0 = som._decay(1.0, t=0, T=100, mode="exponential")
    v50 = som._decay(1.0, t=50, T=100, mode="exponential")
    v99 = som._decay(1.0, t=99, T=100, mode="exponential")
    assert v0 > v50 > v99 > 0


def test_linear_decay_at_t0():
    som = make_som()
    val = som._decay(2.0, t=0, T=100, mode="linear")
    assert val == pytest.approx(2.0)


def test_linear_decay_reaches_near_zero():
    som = make_som()
    val = som._decay(1.0, t=99, T=100, mode="linear")
    assert val == pytest.approx(0.01)


# --- train / predict ---

def test_predict_within_grid():
    som = make_som(rows=3, cols=3, dim=7)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 7))
    som.train(X)
    coords = som.predict(X)
    assert coords.shape == (10, 2)
    assert np.all((coords[:, 0] >= 0) & (coords[:, 0] < 3))
    assert np.all((coords[:, 1] >= 0) & (coords[:, 1] < 3))


def test_train_reduces_quantization_error():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    som = make_som(rows=4, cols=4, dim=4, epochs=200, lr=0.5, radius=2.0)
    error_before = _quantization_error(som, X)
    som.train(X)
    error_after = _quantization_error(som, X)
    assert error_after < error_before


def _quantization_error(som, X):
    total = 0.0
    for x in X:
        r, c = som._find_bmu(x)
        total += np.linalg.norm(x - som.weights[r, c])
    return total / len(X)


# --- u-matrix ---

def test_u_matrix_shape():
    som = make_som(rows=3, cols=4, dim=5)
    u = som.u_matrix()
    assert u.shape == (3, 4)


def test_u_matrix_nonnegative():
    som = make_som(rows=3, cols=4, dim=5)
    u = som.u_matrix()
    assert np.all(u >= 0)
