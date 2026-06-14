import json

import numpy as np
import pytest

from autoencoder.config import ConfigError, load_config
from autoencoder.data import add_noise, load_font
from autoencoder.network import Autoencoder
from autoencoder.train import (
    VAL_REPEATS,
    VAL_SEED,
    _corrupt_for_epoch,
    _sample_level,
    evaluate_denoising,
    train_multi_restart,
    train_once,
)


def test_noise_level_zero_identity():
    X = load_font("font/font.h")
    np.testing.assert_array_equal(add_noise(X, "salt_pepper", 0.0), X)


def test_online_corruption_differs_across_epochs():
    """Con corrupción online, dos épocas ven realizaciones de ruido distintas."""
    X = load_font("font/font.h")[:8]
    spec = {"noise_type": "salt_pepper", "level": 0.2, "train_level_range": None,
            "rng": np.random.default_rng(0)}
    a = _corrupt_for_epoch(spec, X)
    b = _corrupt_for_epoch(spec, X)
    assert not np.array_equal(a, b)


def test_train_level_range_samples_within_bounds_and_fixed_when_null():
    rng = np.random.default_rng(0)
    levels = [_sample_level(rng, 0.1, [0.0, 0.3]) for _ in range(200)]
    assert all(0.0 <= lv <= 0.3 for lv in levels)
    assert len(set(np.round(levels, 6))) > 1  # variabilidad real
    # Sin rango -> siempre el nivel fijo.
    assert _sample_level(rng, 0.1, None) == 0.1


def test_lbfgs_with_online_corruption_raises_in_train_once():
    X = load_font("font/font.h")[:8]
    net = Autoencoder([35, 20, 2], seed=0)
    spec = {"noise_type": "salt_pepper", "level": 0.1, "train_level_range": None,
            "rng": np.random.default_rng(0)}
    with pytest.raises(ValueError):
        train_once(net, X, X, optimizer="lbfgs", epochs=2, denoise=spec)


def test_lbfgs_with_online_corruption_raises_in_config(tmp_path):
    cfg = {
        "name": "bad",
        "architecture": {"encoder_layers": [35, 20, 2]},
        "training": {"optimizer": "lbfgs", "epochs": 10},
        "denoising": {"enabled": True, "resample_per_epoch": True},
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ConfigError):
        load_config(p)


def test_evaluate_denoising_reports_perfect_pct():
    X = load_font("font/font.h")[:8]
    net = Autoencoder([35, 20, 2], seed=0)
    sweep = evaluate_denoising(net, X, "salt_pepper", levels=[0.05, 0.1], seed=0,
                               repeats=2)
    assert "perfect_pct" in sweep.columns
    assert (sweep["perfect_pct"] >= 0).all() and (sweep["perfect_pct"] <= 100).all()


def test_denoising_trains_against_clean_target():
    X = load_font("font/font.h")[:8]
    net, tracker, summary = train_multi_restart(
        X, encoder_layers=[35, 20, 2], epochs=200, lr=5e-3, restarts=1, seed=1,
        denoising={"enabled": True, "noise_type": "salt_pepper", "level": 0.1},
        verbose=False, log_every=50,
    )
    # El tracker compara forward(X̃) contra X (objetivo limpio).
    assert "max_pixel_error" in tracker.df.columns
    assert net.forward(X).shape == X.shape


def test_select_by_denoising_never_worse_than_clean_selection():
    """El restart elegido por denoising no puede denoisear peor que el elegido por
    reconstrucción limpia (el segundo está en el pool del primero), medido con el
    criterio de selección: mean_pixel_error sobre el set de validación fijo."""
    X = load_font("font/font.h")
    den = {"enabled": True, "noise_type": "salt_pepper", "level": 0.1,
           "resample_per_epoch": True, "train_level_range": [0.0, 0.3],
           "sweep_levels": [0.05, 0.1, 0.2, 0.3]}
    kw = dict(encoder_layers=[35, 25, 15, 8, 2], epochs=1500, lr=3e-3, restarts=4,
              seed=7, denoising=den, verbose=False, log_every=10**9, stop_at=None)
    net_clean, _, _ = train_multi_restart(X, select_by_denoising=False, **kw)
    net_den, _, _ = train_multi_restart(X, select_by_denoising=True, **kw)
    score = lambda n: evaluate_denoising(
        n, X, "salt_pepper", levels=den["sweep_levels"], seed=VAL_SEED,
        repeats=VAL_REPEATS)["mean_pixel_error"].mean()
    assert score(net_den) <= score(net_clean) + 1e-9


def test_selection_validation_seed_independent_of_eval_seed():
    """La validación (selección de restarts) y la evaluación final usan seeds fijas y
    distintas: la selección no puede ajustarse al set de reporte."""
    from autoencoder.train import EVAL_SEED

    assert VAL_SEED != EVAL_SEED


def test_corrupt_for_epoch_replicas_stacks_independent_corruptions():
    """Con replicas=k el batch corrupto apila k copias de cada patrón, corrompidas
    de forma independiente."""
    X = load_font("font/font.h")[:8]
    spec = {"noise_type": "salt_pepper", "level": 0.3, "train_level_range": None,
            "replicas": 3, "rng": np.random.default_rng(0)}
    out = _corrupt_for_epoch(spec, X)
    assert out.shape == (24, 35)
    # Las réplicas del mismo patrón difieren entre sí (corrupciones independientes).
    assert not np.array_equal(out[:8], out[8:16])


def test_train_once_with_replicas_trains_against_tiled_clean_target():
    X = load_font("font/font.h")[:8]
    net = Autoencoder([35, 20, 2], seed=0)
    spec = {"noise_type": "salt_pepper", "level": 0.1, "train_level_range": None,
            "replicas": 4, "rng": np.random.default_rng(0)}
    final = train_once(net, X, X, optimizer="adam", epochs=50, lr=3e-3, denoise=spec)
    # Las métricas finales se reportan sobre el set limpio sin tilear.
    assert net.forward(X).shape == X.shape
    assert np.isfinite(final["loss"])


def test_asymmetric_decoder_shapes_and_forward():
    net = Autoencoder([35, 20, 2], decoder_layers=[2, 20, 30, 35], seed=0)
    assert net.decoder_sizes == [2, 20, 30, 35]
    X = load_font("font/font.h")[:4]
    assert net.forward(X).shape == (4, 35)
    with pytest.raises(ValueError):
        Autoencoder([35, 20, 2], decoder_layers=[3, 20, 35], seed=0)


def test_leaky_relu_activation_available():
    net = Autoencoder([35, 20, 2], activation="leaky_relu", seed=0)
    X = load_font("font/font.h")[:4]
    assert net.forward(X).shape == (4, 35)


def test_evaluate_denoising_sweep_levels():
    X = load_font("font/font.h")[:8]
    net = Autoencoder([35, 20, 2], seed=0)
    sweep = evaluate_denoising(net, X, "salt_pepper", levels=[0.05, 0.1, 0.2, 0.3],
                               seed=0, repeats=3)
    assert list(sweep["level"]) == [0.05, 0.1, 0.2, 0.3]
    assert {"max_pixel_error", "mean_pixel_error"}.issubset(sweep.columns)
