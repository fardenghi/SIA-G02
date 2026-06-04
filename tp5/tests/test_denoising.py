import numpy as np

from autoencoder.data import add_noise, load_font
from autoencoder.network import Autoencoder
from autoencoder.train import evaluate_denoising, train_multi_restart


def test_noise_level_zero_identity():
    X = load_font("font/font.h")
    np.testing.assert_array_equal(add_noise(X, "salt_pepper", 0.0), X)


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


def test_evaluate_denoising_sweep_levels():
    X = load_font("font/font.h")[:8]
    net = Autoencoder([35, 20, 2], seed=0)
    sweep = evaluate_denoising(net, X, "salt_pepper", levels=[0.05, 0.1, 0.2, 0.3],
                               seed=0, repeats=3)
    assert list(sweep["level"]) == [0.05, 0.1, 0.2, 0.3]
    assert {"max_pixel_error", "mean_pixel_error"}.issubset(sweep.columns)
