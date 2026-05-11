import subprocess
import sys
import numpy as np
import pytest


# --- load_data ---

def test_load_data_shape():
    from kohonen_europe import load_data
    countries, X = load_data("data/europe.csv")
    assert X.shape == (28, 7)


def test_load_data_countries_count():
    from kohonen_europe import load_data
    countries, _ = load_data("data/europe.csv")
    assert len(countries) == 28


def test_load_data_scaled_mean_near_zero():
    from kohonen_europe import load_data
    _, X = load_data("data/europe.csv")
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)


def test_load_data_scaled_std_near_one():
    from kohonen_europe import load_data
    _, X = load_data("data/europe.csv")
    np.testing.assert_allclose(X.std(axis=0, ddof=0), 1, atol=1e-10)


# --- run ---

def test_run_returns_som_and_coords():
    from kohonen_europe import load_data, run
    import json
    with open("configs/kohonen_europe.json") as f:
        cfg = json.load(f)
    cfg["epochs"] = 5  # fast for test
    countries, X = load_data(cfg["data"])
    som, coords = run(cfg, X)
    assert coords.shape == (28, 2)


def test_run_coords_within_grid():
    from kohonen_europe import load_data, run
    import json
    with open("configs/kohonen_europe.json") as f:
        cfg = json.load(f)
    cfg["epochs"] = 5
    countries, X = load_data(cfg["data"])
    som, coords = run(cfg, X)
    assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < cfg["grid_rows"])
    assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < cfg["grid_cols"])


# --- assignments ---

def test_assignments_covers_all_countries():
    from kohonen_europe import load_data, run, build_assignments
    import json
    with open("configs/kohonen_europe.json") as f:
        cfg = json.load(f)
    cfg["epochs"] = 5
    countries, X = load_data(cfg["data"])
    _, coords = run(cfg, X)
    assignments = build_assignments(countries, coords)
    assigned = [c for cells in assignments.values() for c in cells]
    assert sorted(assigned) == sorted(countries)


def test_assignments_keys_are_tuples():
    from kohonen_europe import load_data, run, build_assignments
    import json
    with open("configs/kohonen_europe.json") as f:
        cfg = json.load(f)
    cfg["epochs"] = 5
    countries, X = load_data(cfg["data"])
    _, coords = run(cfg, X)
    assignments = build_assignments(countries, coords)
    for key in assignments:
        assert isinstance(key, tuple) and len(key) == 2


# --- CLI ---

def test_cli_exits_zero():
    result = subprocess.run(
        [sys.executable, "kohonen_europe.py", "--config", "configs/kohonen_europe.json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_prints_country_names():
    result = subprocess.run(
        [sys.executable, "kohonen_europe.py", "--config", "configs/kohonen_europe.json"],
        capture_output=True, text=True,
    )
    assert "Germany" in result.stdout
    assert "Spain" in result.stdout
