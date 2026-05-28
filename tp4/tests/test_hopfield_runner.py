"""Tests del runner de Hopfield: pick_letters, run_recall, y funciones de capacidad."""
import json
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from hopfield.alphabet import letter_vector, LETTERS
from hopfield.hopfield import HopfieldNetwork, add_noise
from hopfield.hopfield_runner import pick_letters, run_recall
from hopfield.capacity import (
    _hamming,
    evaluate_set,
    pick_combo,
    run_sweep,
)


# ---------------------------------------------------------------------------
# pick_letters()
# ---------------------------------------------------------------------------

def test_pick_letters_from_explicit_config():
    """Si 'letters' está en el config, debe devolver exactamente esas letras."""
    cfg = {"letters": ["G", "R", "T", "V"]}
    result = pick_letters(cfg)
    assert result == ["G", "R", "T", "V"]


def test_pick_letters_uppercase_normalization():
    """pick_letters() debe normalizar a mayúsculas."""
    cfg = {"letters": ["a", "b", "c", "d"]}
    result = pick_letters(cfg)
    assert result == ["A", "B", "C", "D"]


def test_pick_letters_auto_selects_k_letters():
    """Sin 'letters' en config, debe elegir automáticamente k letras."""
    cfg = {"k": 3}
    result = pick_letters(cfg)
    assert len(result) == 3
    assert all(c in LETTERS for c in result)


def test_pick_letters_default_k_is_4():
    """Sin 'k' ni 'letters', el default es k=4."""
    result = pick_letters({})
    assert len(result) == 4


def test_pick_letters_auto_chooses_orthogonal():
    """El subset auto elegido debe tener max|dot| bajo (mejor ortogonalidad)."""
    from hopfield.orthogonality import pairwise_dot_matrix, rank_combinations
    cfg = {"k": 4}
    result = pick_letters(cfg)
    dot = pairwise_dot_matrix()
    df = rank_combinations(4, dot)
    best_combo = set(df.iloc[0]["combo"])
    chosen = set(result)
    # El combo elegido debe ser el mejor (o igual al mejor en max_abs_dot)
    best_max_dot = df.iloc[0]["max_abs_dot"]
    from hopfield.orthogonality import combo_metrics
    chosen_max_dot, _ = combo_metrics(tuple(result), dot)
    assert chosen_max_dot <= best_max_dot + 1  # tolerancia de 1 por empates


# ---------------------------------------------------------------------------
# run_recall()
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_net():
    """Red de Hopfield con GRTV (el subconjunto más ortogonal)."""
    letters = ["G", "R", "T", "V"]
    patterns = np.stack([letter_vector(c) for c in letters])
    net = HopfieldNetwork(n_units=25)
    net.store(patterns)
    return net, letters


def test_run_recall_returns_dict_with_expected_keys(small_net, tmp_path):
    net, letters = small_net
    rng = np.random.default_rng(0)
    result = run_recall(
        net, letters[0],
        noise=0.0,
        mode="sync",
        max_steps=20,
        rng=rng,
        output_dir=str(tmp_path),
        tag="test",
    )
    for key in ("letter", "noise", "steps", "converged", "verdict", "final_energy"):
        assert key in result, f"Falta la clave '{key}' en el resultado"


def test_run_recall_zero_noise_recovers_exactly(small_net, tmp_path):
    """Con ruido 0 el patrón almacenado debe recuperarse como 'ok'."""
    net, letters = small_net
    rng = np.random.default_rng(0)
    result = run_recall(
        net, letters[0],
        noise=0.0,
        mode="sync",
        max_steps=20,
        rng=rng,
        output_dir=str(tmp_path),
        tag="zero_noise",
    )
    assert result["verdict"] == "ok"
    assert result["converged"]


def test_run_recall_creates_output_file(small_net, tmp_path):
    """run_recall debe crear un PNG en output_dir."""
    net, letters = small_net
    rng = np.random.default_rng(0)
    run_recall(
        net, letters[0],
        noise=0.0,
        mode="sync",
        max_steps=10,
        rng=rng,
        output_dir=str(tmp_path),
        tag="file_test",
    )
    expected = tmp_path / f"recall_file_test_{letters[0]}.png"
    assert expected.exists() and expected.stat().st_size > 0


def test_run_recall_letter_field_matches(small_net, tmp_path):
    """El campo 'letter' del resultado debe coincidir con la letra pedida."""
    net, letters = small_net
    rng = np.random.default_rng(0)
    for letter in letters:
        result = run_recall(
            net, letter,
            noise=0.0,
            mode="sync",
            max_steps=10,
            rng=rng,
            output_dir=str(tmp_path),
            tag="match",
        )
        assert result["letter"] == letter


def test_run_recall_noise_field_matches(small_net, tmp_path):
    net, letters = small_net
    rng = np.random.default_rng(0)
    noise = 0.15
    result = run_recall(
        net, letters[0],
        noise=noise,
        mode="sync",
        max_steps=10,
        rng=rng,
        output_dir=str(tmp_path),
        tag="noise_match",
    )
    assert result["noise"] == pytest.approx(noise)


def test_run_recall_accepts_noisy_input(small_net, tmp_path):
    """Debe aceptar un patrón ruidoso pre-construido vía noisy_input."""
    net, letters = small_net
    rng = np.random.default_rng(0)
    p = letter_vector(letters[0])
    noisy = add_noise(p, 0.1, rng)
    result = run_recall(
        net, letters[0],
        noise=0.0,  # irrelevante porque noisy_input está presente
        mode="sync",
        max_steps=10,
        rng=rng,
        output_dir=str(tmp_path),
        tag="prebuilt",
        noisy_input=noisy,
    )
    assert result["letter"] == letters[0]


# ---------------------------------------------------------------------------
# _hamming()
# ---------------------------------------------------------------------------

def test_hamming_identical():
    a = np.array([1, -1, 1, -1], dtype=np.int8)
    assert _hamming(a, a) == 0


def test_hamming_completely_different():
    a = np.array([1, 1, -1, -1], dtype=np.int8)
    b = -a
    assert _hamming(a, b) == 4


def test_hamming_one_bit():
    a = np.array([1, 1, 1, 1], dtype=np.int8)
    b = np.array([1, 1, 1, -1], dtype=np.int8)
    assert _hamming(a, b) == 1


# ---------------------------------------------------------------------------
# evaluate_set()
# ---------------------------------------------------------------------------

def test_evaluate_set_returns_list_of_dicts():
    """evaluate_set debe retornar una lista de dicts con las métricas esperadas."""
    rng = np.random.default_rng(0)
    rows = evaluate_set(
        letters=["G", "R"],
        noise_levels=[0.0, 0.1],
        n_trials=2,
        max_steps=10,
        rng=rng,
    )
    assert len(rows) == 2  # un dict por noise_level
    for row in rows:
        for key in ("n_patterns", "scale", "n_units", "noise",
                    "recall_accuracy", "spurious_rate", "avg_hamming_to_original"):
            assert key in row, f"Falta '{key}' en el resultado"


def test_evaluate_set_recall_accuracy_in_range():
    rng = np.random.default_rng(0)
    rows = evaluate_set(
        letters=["G", "R", "T", "V"],
        noise_levels=[0.0],
        n_trials=3,
        max_steps=20,
        rng=rng,
    )
    acc = rows[0]["recall_accuracy"]
    assert 0.0 <= acc <= 1.0


def test_evaluate_set_zero_noise_high_accuracy():
    """Con ruido=0 y el mejor subconjunto ortogonal, el recall debe ser alto."""
    rng = np.random.default_rng(42)
    rows = evaluate_set(
        letters=["G", "R", "T", "V"],
        noise_levels=[0.0],
        n_trials=5,
        max_steps=20,
        rng=rng,
    )
    assert rows[0]["recall_accuracy"] >= 0.8


def test_evaluate_set_n_patterns_matches():
    rng = np.random.default_rng(0)
    rows = evaluate_set(
        letters=["A", "B", "C"],
        noise_levels=[0.1],
        n_trials=1,
        max_steps=10,
        rng=rng,
    )
    assert rows[0]["n_patterns"] == 3


def test_evaluate_set_hamming_nonnegative():
    rng = np.random.default_rng(0)
    rows = evaluate_set(
        letters=["G", "V"],
        noise_levels=[0.2],
        n_trials=2,
        max_steps=10,
        rng=rng,
    )
    assert rows[0]["avg_hamming_to_original"] >= 0.0


# ---------------------------------------------------------------------------
# pick_combo()
# ---------------------------------------------------------------------------

def test_pick_combo_best_returns_k_letters():
    from hopfield.orthogonality import pairwise_dot_matrix
    dot = pairwise_dot_matrix()
    rng = np.random.default_rng(0)
    combo = pick_combo(4, "best", dot, rng)
    assert len(combo) == 4
    assert all(c in LETTERS for c in combo)


def test_pick_combo_worst_returns_k_letters():
    from hopfield.orthogonality import pairwise_dot_matrix
    dot = pairwise_dot_matrix()
    rng = np.random.default_rng(0)
    combo = pick_combo(3, "worst", dot, rng)
    assert len(combo) == 3


def test_pick_combo_random_returns_k_unique():
    from hopfield.orthogonality import pairwise_dot_matrix
    dot = pairwise_dot_matrix()
    rng = np.random.default_rng(7)
    combo = pick_combo(5, "random", dot, rng)
    assert len(combo) == 5
    assert len(set(combo)) == 5  # sin repetición


def test_pick_combo_first_returns_first_k():
    from hopfield.orthogonality import pairwise_dot_matrix
    dot = pairwise_dot_matrix()
    rng = np.random.default_rng(0)
    combo = pick_combo(4, "first", dot, rng)
    assert combo == LETTERS[:4]


def test_pick_combo_invalid_mode_raises():
    from hopfield.orthogonality import pairwise_dot_matrix
    dot = pairwise_dot_matrix()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="mode inválido"):
        pick_combo(3, "nonexistent", dot, rng)


def test_pick_combo_best_is_more_orthogonal_than_worst():
    """El combo 'best' debe tener max|dot| ≤ 'worst'."""
    from hopfield.orthogonality import pairwise_dot_matrix, combo_metrics
    dot = pairwise_dot_matrix()
    rng = np.random.default_rng(0)
    best = tuple(pick_combo(4, "best", dot, rng))
    worst = tuple(pick_combo(4, "worst", dot, rng))
    max_best, _ = combo_metrics(best, dot)
    max_worst, _ = combo_metrics(worst, dot)
    assert max_best <= max_worst


# ---------------------------------------------------------------------------
# run_sweep()
# ---------------------------------------------------------------------------

def test_run_sweep_returns_dataframe():
    df = run_sweep(
        k_max=2,
        modes=["best"],
        noise_levels=[0.1],
        n_trials=1,
        max_steps=5,
        seed=42,
        adaptive=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_run_sweep_columns_present():
    df = run_sweep(
        k_max=2,
        modes=["first"],
        noise_levels=[0.1],
        n_trials=1,
        max_steps=5,
        seed=0,
    )
    expected_cols = {"n_patterns", "noise", "recall_accuracy",
                     "spurious_rate", "avg_hamming_to_original", "mode"}
    assert expected_cols.issubset(set(df.columns))


def test_run_sweep_n_patterns_range():
    df = run_sweep(
        k_max=3,
        modes=["first"],
        noise_levels=[0.0],
        n_trials=1,
        max_steps=5,
        seed=0,
    )
    assert set(df["n_patterns"].unique()) == {1, 2, 3}


def test_run_sweep_mode_column_matches_input():
    df = run_sweep(
        k_max=2,
        modes=["best", "worst"],
        noise_levels=[0.1],
        n_trials=1,
        max_steps=5,
        seed=0,
    )
    assert set(df["mode"].unique()) == {"best", "worst"}


# ---------------------------------------------------------------------------
# CLI — runner de Hopfield
# ---------------------------------------------------------------------------

def test_hopfield_runner_cli_exits_zero(tmp_path):
    """El CLI de hopfield_runner debe completar sin error."""
    cfg = {
        "letters": ["G", "R", "T", "V"],
        "k": 4,
        "noise": 0.15,
        "high_noise": 0.40,
        "mode": "sync",
        "max_steps": 5,
        "spurious_attempts": 2,
        "seed": 42,
        "output_dir": str(tmp_path),
        "noise_levels_analysis": [0.0, 0.1],
        "n_trials": 2,
    }
    tmp_cfg = tmp_path / "cfg.json"
    tmp_cfg.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "hopfield.hopfield_runner", "--config", str(tmp_cfg)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_hopfield_runner_cli_creates_output_files(tmp_path):
    """El CLI debe generar al menos crosstalk.png y recovery_rate.png."""
    cfg = {
        "letters": ["G", "R", "T", "V"],
        "k": 4,
        "noise": 0.1,
        "high_noise": 0.4,
        "mode": "sync",
        "max_steps": 5,
        "spurious_attempts": 2,
        "seed": 42,
        "output_dir": str(tmp_path),
        "noise_levels_analysis": [0.0, 0.1],
        "n_trials": 2,
    }
    tmp_cfg = tmp_path / "cfg.json"
    tmp_cfg.write_text(json.dumps(cfg))

    subprocess.run(
        [sys.executable, "-m", "hopfield.hopfield_runner", "--config", str(tmp_cfg)],
        capture_output=True, text=True,
        check=True,
    )
    assert (tmp_path / "crosstalk.png").exists()
    assert (tmp_path / "recovery_rate.png").exists()
