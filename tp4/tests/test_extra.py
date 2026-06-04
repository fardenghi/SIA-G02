"""Tests adicionales para OjaNetwork y kohonen_europe:
- OjaNetwork: train con un sample, predict con un sample, lr muy alto/bajo
- plot_component_planes: genera el archivo
- kohonen_europe: funciones de color de celdas
"""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from oja.oja import OjaNetwork
from kohonen.som import SOM
from kohonen.kohonen_europe import (
    _cell_color,
    _DEFAULT_CELL_COLOR,
    _COUNTRY_GROUPS,
    build_assignments,
    plot_component_planes,
)


# ---------------------------------------------------------------------------
# OjaNetwork — casos extra
# ---------------------------------------------------------------------------

def test_train_single_sample_does_not_explode():
    """Con un único sample de entrenamiento, los pesos deben seguir siendo finitos."""
    net = OjaNetwork(input_dim=3, lr=0.1, epochs=5, seed=0)
    X = np.array([[1.0, 2.0, -1.0]])
    net.train(X)
    assert np.all(np.isfinite(net.weights))


def test_predict_single_sample():
    """predict debe funcionar con un array de un solo sample."""
    net = OjaNetwork(input_dim=4, lr=0.1, epochs=10, seed=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    net.train(X)
    x_single = rng.standard_normal((1, 4))
    scores = net.predict(x_single)
    assert scores.shape == (1,)


def test_lr_very_low_still_converges_direction():
    """Con lr muy bajo, la dirección del peso debe acercarse al PC1 de sklearn."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(1)
    n = 300
    signal = rng.standard_normal(n) * 5
    X = np.column_stack([signal, rng.standard_normal(n) * 0.1])
    X -= X.mean(axis=0)

    sk_w = PCA(n_components=1).fit(X).components_[0]
    net = OjaNetwork(input_dim=2, lr=0.01, epochs=50, seed=0)
    net.train(X)
    w = net.component()
    if np.dot(w, sk_w) < 0:
        w = -w
    assert float(np.dot(w, sk_w)) > 0.90


def test_history_length_matches_epochs():
    """history debe tener epochs+1 entradas (incluye estado inicial)."""
    epochs = 7
    net = OjaNetwork(input_dim=3, lr=0.1, epochs=epochs, seed=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    net.train(X)
    assert len(net.history) == epochs + 1


def test_component_same_direction_as_weights():
    """component() debe apuntar en la misma dirección que weights."""
    net = OjaNetwork(input_dim=4, lr=0.1, epochs=10, seed=0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    net.train(X)
    c = net.component()
    # Coseno entre weights y component debe ser positivo (misma dirección)
    cos = float(np.dot(net.weights, c)) / (np.linalg.norm(net.weights) * np.linalg.norm(c))
    assert cos > 0.99


# ---------------------------------------------------------------------------
# plot_component_planes()
# ---------------------------------------------------------------------------

def test_plot_component_planes_creates_file():
    """plot_component_planes debe crear un PNG válido."""
    som = SOM(
        grid_rows=3, grid_cols=3, input_dim=3,
        lr=0.3, lr_decay="exponential",
        radius=1.0, radius_decay="exponential",
        neighborhood_fn="gaussian",
        epochs=3, seed=0,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3))
    som.train(X)
    features = ["F1", "F2", "F3"]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "component_planes.png")
        plot_component_planes(som, features, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_component_planes_seven_features():
    """Debe funcionar con las 7 features del dataset europa."""
    som = SOM(
        grid_rows=2, grid_cols=2, input_dim=7,
        lr=0.3, lr_decay="exponential",
        radius=1.0, radius_decay="exponential",
        neighborhood_fn="gaussian",
        epochs=2, seed=0,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 7))
    som.train(X)
    features = ["Area", "GDP", "Inflation", "Life.expect",
                "Military", "Pop.growth", "Unemployment"]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "planes.png")
        plot_component_planes(som, features, path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# _cell_color() — lógica de color de celdas
# ---------------------------------------------------------------------------

def test_cell_color_empty_returns_default():
    """Con lista vacía de países debe devolver el color por defecto."""
    color = _cell_color([])
    assert color == _DEFAULT_CELL_COLOR


def test_cell_color_known_country_returns_group_color():
    """Un país conocido en un grupo debe devolver el color de ese grupo."""
    # Elegimos un país del primer grupo
    country = next(iter(_COUNTRY_GROUPS[0]["countries"]))
    color = _cell_color([country])
    assert color == _COUNTRY_GROUPS[0]["color"]


def test_cell_color_unknown_country_returns_default():
    """Un país no clasificado en ningún grupo debe devolver el color default."""
    color = _cell_color(["PaisDesconocido"])
    assert color == _DEFAULT_CELL_COLOR


def test_cell_color_multiple_countries_picks_majority():
    """Con varios países del mismo grupo, el color debe ser el de ese grupo."""
    group = _COUNTRY_GROUPS[1]
    countries = list(group["countries"])[:2]
    color = _cell_color(countries)
    assert color == group["color"]


# ---------------------------------------------------------------------------
# build_assignments() — extra
# ---------------------------------------------------------------------------

def test_build_assignments_no_duplicates():
    """Cada país debe aparecer exactamente una vez en los assignments."""
    countries = ["Germany", "France", "Spain"]
    coords = np.array([[0, 1], [0, 1], [2, 3]])
    assignments = build_assignments(countries, coords)
    all_countries = [c for cell in assignments.values() for c in cell]
    assert sorted(all_countries) == sorted(countries)


def test_build_assignments_multiple_in_same_cell():
    """Varios países en la misma celda deben agruparse juntos."""
    countries = ["A", "B", "C"]
    coords = np.array([[1, 1], [1, 1], [0, 0]])
    assignments = build_assignments(countries, coords)
    assert len(assignments[(1, 1)]) == 2
    assert len(assignments[(0, 0)]) == 1
