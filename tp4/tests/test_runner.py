import json
import subprocess
import sys

import numpy as np

from kohonen.kohonen_europe import build_assignments, load_data, run


CONFIG = "configs/kohonen_europe.json"


def _load_cfg(epochs: int | None = None) -> dict:
    with open(CONFIG) as f:
        cfg = json.load(f)
    if epochs is not None:
        cfg["epochs"] = epochs
    return cfg


# --- load_data ---

def test_load_data_shape():
    countries, X = load_data("data/europe.csv")
    assert X.shape == (28, 7)


def test_load_data_countries_count():
    countries, _ = load_data("data/europe.csv")
    assert len(countries) == 28


def test_load_data_scaled_mean_near_zero():
    _, X = load_data("data/europe.csv")
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)


def test_load_data_scaled_std_near_one():
    _, X = load_data("data/europe.csv")
    np.testing.assert_allclose(X.std(axis=0, ddof=0), 1, atol=1e-10)


# --- run ---

def test_run_returns_som_and_coords():
    cfg = _load_cfg(epochs=5)
    _, X = load_data(cfg["data"])
    som, coords = run(cfg, X)
    assert coords.shape == (28, 2)
    assert som.weights.shape == (cfg["grid_rows"], cfg["grid_cols"], X.shape[1])


def test_run_coords_within_grid():
    cfg = _load_cfg(epochs=5)
    _, X = load_data(cfg["data"])
    _, coords = run(cfg, X)
    assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < cfg["grid_rows"])
    assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < cfg["grid_cols"])


# --- assignments ---

def test_assignments_covers_all_countries():
    cfg = _load_cfg(epochs=5)
    countries, X = load_data(cfg["data"])
    _, coords = run(cfg, X)
    assignments = build_assignments(countries, coords)
    assigned = [c for cells in assignments.values() for c in cells]
    assert sorted(assigned) == sorted(countries)


def test_assignments_keys_are_tuples():
    cfg = _load_cfg(epochs=5)
    countries, X = load_data(cfg["data"])
    _, coords = run(cfg, X)
    assignments = build_assignments(countries, coords)
    for key in assignments:
        assert isinstance(key, tuple) and len(key) == 2


# --- CLI ---

def test_cli_exits_zero(tmp_path):
    cfg = _load_cfg(epochs=5)
    cfg["output_dir"] = str(tmp_path)
    tmp_cfg = tmp_path / "cfg.json"
    tmp_cfg.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "kohonen.kohonen_europe", "--config", str(tmp_cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_prints_country_names(tmp_path):
    cfg = _load_cfg(epochs=5)
    cfg["output_dir"] = str(tmp_path)
    tmp_cfg = tmp_path / "cfg.json"
    tmp_cfg.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "kohonen.kohonen_europe", "--config", str(tmp_cfg)],
        capture_output=True, text=True,
    )
    assert "Germany" in result.stdout
    assert "Spain" in result.stdout
