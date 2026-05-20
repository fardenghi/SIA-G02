import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from oja.oja_europe import (
    align_sign,
    compute_sklearn_pc1,
    load_data,
    plot_convergence,
    plot_country_scores,
    plot_loadings,
    run,
)


CONFIG = "configs/oja_europe.json"


def _load_cfg(epochs: int | None = None) -> dict:
    with open(CONFIG) as f:
        cfg = json.load(f)
    if epochs is not None:
        cfg["epochs"] = epochs
    return cfg


# --- load_data ---

def test_load_data_shape():
    countries, X, features = load_data("data/europe.csv")
    assert X.shape == (28, 7)
    assert len(countries) == 28
    assert len(features) == 7


def test_load_data_is_standardized():
    _, X, _ = load_data("data/europe.csv")
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(X.std(axis=0, ddof=0), 1, atol=1e-10)


# --- run ---

def test_run_returns_trained_net():
    cfg = _load_cfg(epochs=5)
    _, X, _ = load_data(cfg["data"])
    net = run(cfg, X)
    assert net.weights.shape == (X.shape[1],)
    assert np.all(np.isfinite(net.weights))


# --- sklearn comparison ---

def test_compute_sklearn_pc1_returns_unit_loading():
    _, X, _ = load_data("data/europe.csv")
    loading, scores, explained = compute_sklearn_pc1(X)
    assert loading.shape == (X.shape[1],)
    assert scores.shape == (X.shape[0],)
    assert np.linalg.norm(loading) == pytest.approx(1.0)
    assert 0 < explained <= 1


def test_align_sign_flips_when_anti_parallel():
    w = np.array([1.0, 0.0, 0.0])
    ref = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(align_sign(w, ref), -w)


def test_align_sign_keeps_when_parallel():
    w = np.array([1.0, 2.0, 3.0])
    ref = np.array([0.5, 1.0, 1.5])
    assert np.allclose(align_sign(w, ref), w)


def test_oja_matches_sklearn_pc1_on_europe():
    cfg = _load_cfg(epochs=200)
    _, X, _ = load_data(cfg["data"])
    net = run(cfg, X)
    oja_w = net.component()
    sk_w, _, _ = compute_sklearn_pc1(X)
    if np.dot(oja_w, sk_w) < 0:
        oja_w = -oja_w
    assert float(np.dot(oja_w, sk_w)) > 0.95


# --- plots ---

@pytest.fixture(scope="module")
def trained():
    cfg = _load_cfg(epochs=20)
    countries, X, features = load_data(cfg["data"])
    net = run(cfg, X)
    sk_w, sk_scores, _ = compute_sklearn_pc1(X)
    oja_w = align_sign(net.component(), sk_w)
    oja_scores = X @ oja_w
    return cfg, countries, X, features, net, oja_w, sk_w, oja_scores, sk_scores


def test_plot_loadings_creates_file(trained):
    _, _, _, features, _, oja_w, sk_w, _, _ = trained
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "loadings.png")
        plot_loadings(features, oja_w, sk_w, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_country_scores_creates_file(trained):
    _, countries, _, _, _, _, _, oja_scores, _ = trained
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "country_scores.png")
        plot_country_scores(countries, oja_scores, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_convergence_creates_file(trained):
    _, _, _, _, net, _, sk_w, _, _ = trained
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "convergence.png")
        plot_convergence(net.history, sk_w, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


# --- CLI ---

def test_cli_generates_all_plots(tmp_path):
    cfg = _load_cfg(epochs=10)
    cfg["output_dir"] = str(tmp_path)
    tmp_cfg = tmp_path / "cfg.json"
    tmp_cfg.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "oja.oja_europe", "--config", str(tmp_cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "loadings.png").exists()
    assert (tmp_path / "country_scores.png").exists()
    assert (tmp_path / "convergence.png").exists()
