"""Tests adicionales para SOM: get_quantization_error, get_topological_error,
entrenamiento con record_te=True, y casos extremos del decay."""
import numpy as np
import pytest
from kohonen.som import SOM


def make_som(rows=3, cols=3, dim=4, **kwargs):
    defaults = dict(
        lr=0.5, lr_decay="exponential", radius=1.5,
        radius_decay="exponential", neighborhood_fn="gaussian",
        epochs=10, seed=42,
    )
    defaults.update(kwargs)
    return SOM(rows, cols, dim, **defaults)


# ---------------------------------------------------------------------------
# get_quantization_error()
# ---------------------------------------------------------------------------

def test_quantization_error_nonnegative():
    """El QE debe ser ≥ 0 siempre."""
    som = make_som()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    som.train(X)
    qe = som.get_quantization_error(X)
    assert qe >= 0.0


def test_quantization_error_decreases_after_training():
    """El QE después del entrenamiento debe ser menor o igual al inicial."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 4))
    som = make_som(rows=4, cols=4, dim=4, epochs=100, lr=0.5, radius=2.0)
    qe_before = som.get_quantization_error(X)
    som.train(X)
    qe_after = som.get_quantization_error(X)
    assert qe_after <= qe_before


def test_quantization_error_single_sample():
    """Con un único sample, el QE debe ser finito y no negativo."""
    som = make_som()
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    som.train(X)
    qe = som.get_quantization_error(X)
    assert np.isfinite(qe) and qe >= 0.0


# ---------------------------------------------------------------------------
# get_topological_error()
# ---------------------------------------------------------------------------

def test_topological_error_in_range():
    """El TE debe estar en [0, 1]."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    som = make_som(rows=4, cols=4, dim=4, epochs=50, lr=0.5, radius=2.0)
    som.train(X)
    te = som.get_topological_error(X)
    assert 0.0 <= te <= 1.0


def test_topological_error_type_float():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    som = make_som()
    som.train(X)
    te = som.get_topological_error(X)
    assert isinstance(te, float)


# ---------------------------------------------------------------------------
# train() con record_te=True
# ---------------------------------------------------------------------------

def test_train_record_te_returns_two_lists():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    som = make_som(epochs=5)
    result = som.train(X, record_te=True)
    assert isinstance(result, tuple) and len(result) == 2
    qe_hist, te_hist = result
    assert len(qe_hist) == 5
    assert len(te_hist) == 5


def test_train_record_te_false_returns_list():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    som = make_som(epochs=5)
    result = som.train(X, record_te=False)
    assert isinstance(result, list)
    assert len(result) == 5


def test_train_qe_history_all_nonnegative():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    som = make_som(epochs=8)
    qe_hist = som.train(X)
    assert all(q >= 0.0 for q in qe_hist)


# ---------------------------------------------------------------------------
# decay() — casos límite
# ---------------------------------------------------------------------------

def test_exponential_decay_never_zero_before_T():
    """El decaimiento exponencial no debe llegar exactamente a 0 antes de T."""
    som = make_som()
    for t in range(99):
        v = som._decay(1.0, t, T=100, mode="exponential")
        assert v > 0


def test_linear_decay_at_half_T():
    """A t=T/2, el valor lineal debe ser v₀/2."""
    som = make_som()
    v = som._decay(2.0, t=50, T=100, mode="linear")
    assert v == pytest.approx(1.0)


def test_exponential_decay_large_radius():
    """Para radius > 1 se usa τ = T/log(radius); debe decaer monotónicamente."""
    som = make_som()
    vals = [som._decay(3.0, t, T=100, mode="exponential") for t in range(100)]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-9  # monótonamente decreciente


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def test_predict_dtype_int():
    """Las coordenadas de predict deben ser enteros (índices de grilla)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4))
    som = make_som()
    som.train(X)
    coords = som.predict(X)
    assert coords.dtype in (np.int32, np.int64, np.intp)


def test_predict_single_sample():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1, 4))
    som = make_som()
    som.train(X)
    coords = som.predict(X)
    assert coords.shape == (1, 2)


# ---------------------------------------------------------------------------
# neighborhood — bubble con radio cero
# ---------------------------------------------------------------------------

def test_bubble_radius_zero_only_center():
    """Con radius=0, solo la neurona central debe tener h=1."""
    som = make_som(rows=5, cols=5, dim=2)
    bmu = (2, 2)
    h = som._bubble(bmu, radius=0.0)
    # Solo el centro tiene distancia 0 (≤ 0 solo si radius=0 con ≤)
    assert h[2, 2] == pytest.approx(1.0)
    # Todos los vecinos deben ser 0
    assert h[2, 3] == pytest.approx(0.0)
    assert h[1, 2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SOM con bubble neighborhood — integración mínima
# ---------------------------------------------------------------------------

def test_train_with_bubble_neighborhood():
    """El SOM debe entrenarse sin error con neighborhood_fn='bubble'."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    som = make_som(rows=3, cols=3, dim=3, neighborhood_fn="bubble", epochs=5)
    qe = som.train(X)
    assert all(np.isfinite(q) for q in qe)


def test_train_with_linear_decays():
    """El SOM debe entrenarse sin error con ambos decays en modo 'linear'."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    som = make_som(
        rows=3, cols=3, dim=3,
        lr_decay="linear", radius_decay="linear",
        epochs=5,
    )
    qe = som.train(X)
    assert all(np.isfinite(q) for q in qe)
