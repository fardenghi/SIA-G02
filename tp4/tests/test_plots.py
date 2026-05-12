import json
import os
import subprocess
import sys
import tempfile

import pytest

import matplotlib
matplotlib.use("Agg")

from kohonen.kohonen_europe import (
    build_assignments,
    load_data,
    plot_country_map,
    plot_hit_map,
    plot_u_matrix,
    run,
)


@pytest.fixture(scope="module")
def trained():
    with open("configs/kohonen_europe.json") as f:
        cfg = json.load(f)
    cfg["epochs"] = 10
    countries, X = load_data(cfg["data"])
    som, coords = run(cfg, X)
    assignments = build_assignments(countries, coords)
    return cfg, countries, X, som, coords, assignments


# --- country_map ---

def test_country_map_creates_file(trained):
    cfg, _, _, _, _, assignments = trained
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "country_map.png")
        plot_country_map(assignments, cfg["grid_rows"], cfg["grid_cols"], path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# --- u_matrix ---

def test_u_matrix_plot_creates_file(trained):
    _, _, _, som, _, _ = trained
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u_matrix.png")
        plot_u_matrix(som, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# --- hit_map ---

def test_hit_map_creates_file(trained):
    cfg, countries, _, _, _, assignments = trained
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hit_map.png")
        plot_hit_map(assignments, cfg["grid_rows"], cfg["grid_cols"], len(countries), path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_hit_map_counts_sum_to_n_countries(trained):
    _, countries, _, _, _, assignments = trained
    total = sum(len(v) for v in assignments.values())
    assert total == len(countries)


# --- main script ---

def test_main_generates_all_plots(tmp_path):
    with open("configs/kohonen_europe.json") as f:
        cfg = json.load(f)
    cfg["epochs"] = 10
    cfg["output_dir"] = str(tmp_path)
    tmp_cfg = tmp_path / "test_cfg.json"
    tmp_cfg.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "kohonen.kohonen_europe", "--config", str(tmp_cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "country_map.png").exists()
    assert (tmp_path / "u_matrix.png").exists()
    assert (tmp_path / "hit_map.png").exists()
