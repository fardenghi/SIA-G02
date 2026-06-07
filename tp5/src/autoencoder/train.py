"""Entrenamiento full-batch, multi-restart, métricas de píxeles y MetricsTracker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import losses as losses_mod
from .data import add_noise
from .network import Autoencoder
from .optim import Adam, lbfgs_minimize


# --------------------------------------------------------------------------- #
# Métrica de píxeles (umbral 0.5, conteo por patrón, máximo sobre los patrones)
# --------------------------------------------------------------------------- #


def pixel_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Cantidad de píxeles distintos por patrón tras umbralizar `y_pred` a 0.5."""
    pred_bin = (y_pred >= 0.5).astype(int)
    true_bin = (y_true >= 0.5).astype(int)
    return np.sum(pred_bin != true_bin, axis=1)


def max_pixel_error(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return int(pixel_errors(y_true, y_pred).max())


def mean_pixel_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(pixel_errors(y_true, y_pred).mean())


# --------------------------------------------------------------------------- #
# MetricsTracker (pandas)
# --------------------------------------------------------------------------- #


class MetricsTracker:
    """Acumula métricas por época en un DataFrame; exporta a CSV y compara runs."""

    def __init__(self, run_label: str = "run"):
        self.run_label = run_label
        self._rows: list[dict] = []

    def log(self, epoch: int, loss: float, max_pix: int, mean_pix: float) -> None:
        self._rows.append({
            "run": self.run_label,
            "epoch": epoch,
            "loss": loss,
            "max_pixel_error": max_pix,
            "mean_pixel_error": mean_pix,
        })

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)

    @staticmethod
    def compare_runs(trackers_or_dfs) -> pd.DataFrame:
        """Concatena varios runs (MetricsTracker o DataFrame) etiquetados por `run`."""
        frames = []
        for item in trackers_or_dfs:
            frames.append(item.df if isinstance(item, MetricsTracker) else item)
        return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Entrenamiento de un modelo
# --------------------------------------------------------------------------- #


def train_once(
    net: Autoencoder,
    X_input: np.ndarray,
    X_target: np.ndarray,
    loss: str = "bce",
    optimizer: str = "adam",
    epochs: int = 20000,
    lr: float = 1e-3,
    tracker: MetricsTracker | None = None,
    log_every: int = 100,
    lr_schedule: str | None = None,
    lr_min: float = 0.0,
) -> dict:
    """Entrena `net` in-place sobre el par `(X_input, X_target)` en full-batch.

    Para denoising, `X_input` es la versión ruidosa y `X_target` la limpia.
    Devuelve un dict con métricas finales.
    """
    loss_value, loss_grad = losses_mod.get_loss(loss)

    def record(epoch: int, out: np.ndarray) -> None:
        if tracker is not None and (epoch % log_every == 0):
            tracker.log(epoch, loss_value(out, X_target),
                        max_pixel_error(X_target, out),
                        mean_pixel_error(X_target, out))

    if optimizer == "adam":
        opt = Adam(lr=lr)
        params = net.get_params()
        denom = max(1, epochs - 1)
        for epoch in range(epochs):
            # Scheduler de lr (solo Adam). Cosine annealing de `lr` a `lr_min`.
            if lr_schedule == "cosine":
                opt.lr = lr_min + 0.5 * (lr - lr_min) * (
                    1.0 + np.cos(np.pi * epoch / denom))
            net.set_params(params)
            out = net.forward(X_input)
            net.backward(loss_grad(out, X_target))
            grads = net.get_grads()
            params = opt.step(params, grads)
            record(epoch, out)
        net.set_params(params)

    elif optimizer == "lbfgs":
        state = {"epoch": 0}

        def fun_and_grad(theta: np.ndarray):
            net.set_params(theta)
            out = net.forward(X_input)
            val = loss_value(out, X_target)
            net.backward(loss_grad(out, X_target))
            return val, net.get_grads()

        def cb(theta: np.ndarray):
            net.set_params(theta)
            out = net.forward(X_input)
            record(state["epoch"], out)
            state["epoch"] += 1

        result = lbfgs_minimize(fun_and_grad, net.get_params(), maxiter=epochs,
                                callback=cb)
        net.set_params(result.x)
    else:
        raise ValueError(f"Optimizador desconocido: {optimizer!r}")

    out = net.forward(X_input)
    return {
        "loss": loss_value(out, X_target),
        "max_pixel_error": max_pixel_error(X_target, out),
        "mean_pixel_error": mean_pixel_error(X_target, out),
    }


# --------------------------------------------------------------------------- #
# Multi-restart
# --------------------------------------------------------------------------- #


def train_multi_restart(
    X: np.ndarray,
    encoder_layers: list[int],
    activation: str = "tanh",
    output_activation: str = "sigmoid",
    init: str = "xavier_normal",
    loss: str = "bce",
    optimizer: str = "adam",
    epochs: int = 20000,
    lr: float = 1e-3,
    restarts: int = 20,
    seed: int = 42,
    denoising: dict | None = None,
    log_every: int = 100,
    verbose: bool = True,
    stop_at: int | None = 0,
    latent_activation: str | None = None,
    lr_schedule: str | None = None,
    lr_min: float = 0.0,
):
    """Entrena `restarts` modelos con semillas distintas; conserva el de menor
    `max_pixel_error`. Devuelve `(mejor_red, mejor_tracker, resumen_df)`.

    `stop_at`: si un restart alcanza `max_pixel_error <= stop_at`, se abortan los
    restarts restantes (ya se encontró un modelo suficientemente bueno). Con
    `stop_at=0` (default) corta apenas un modelo reconstruye los 32/32 sin error;
    `None` desactiva el corte y corre los `restarts` completos.
    """
    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 2**31 - 1, size=restarts)

    best_net = None
    best_tracker = None
    best_mpe = np.inf
    summary_rows = []

    for r, s in enumerate(seeds):
        s = int(s)
        net = Autoencoder(encoder_layers, activation, output_activation, init, seed=s,
                          latent_activation=latent_activation)

        # Entrada/objetivo: con denoising, entrada ruidosa y objetivo limpio.
        if denoising and denoising.get("enabled"):
            noise_rng = np.random.default_rng(s + 1)
            X_input = add_noise(X, denoising.get("noise_type", "salt_pepper"),
                                denoising.get("level", 0.1), noise_rng)
        else:
            X_input = X
        X_target = X

        tracker = MetricsTracker(run_label=f"{optimizer}_r{r}")
        final = train_once(net, X_input, X_target, loss, optimizer, epochs, lr,
                           tracker=tracker, log_every=log_every,
                           lr_schedule=lr_schedule, lr_min=lr_min)
        summary_rows.append({"restart": r, "seed": s, **final})
        if verbose:
            print(f"  restart {r:2d} (seed {s}): "
                  f"loss={final['loss']:.4f} max_pix={final['max_pixel_error']} "
                  f"mean_pix={final['mean_pixel_error']:.3f}")

        if final["max_pixel_error"] < best_mpe:
            best_mpe = final["max_pixel_error"]
            best_net = net
            best_tracker = tracker

        # Corte temprano: si ya alcanzamos el umbral objetivo, no seguimos.
        if stop_at is not None and best_mpe <= stop_at:
            if verbose:
                print(f"  -> corte temprano en restart {r}: "
                      f"max_pixel_error={best_mpe} <= stop_at={stop_at} "
                      f"({r + 1}/{len(seeds)} restarts ejecutados)")
            break

    summary = pd.DataFrame(summary_rows)
    return best_net, best_tracker, summary


def evaluate_denoising(
    net: Autoencoder,
    X_clean: np.ndarray,
    noise_type: str = "salt_pepper",
    levels=(0.05, 0.1, 0.2, 0.3),
    seed: int = 0,
    repeats: int = 5,
) -> pd.DataFrame:
    """Barre niveles de ruido y mide el error de reconstrucción contra `X_clean`.

    Para cada nivel corrompe `X_clean`, reconstruye y mide `max`/`mean` pixel error,
    promediando sobre `repeats` realizaciones de ruido.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for level in levels:
        maxes, means = [], []
        for _ in range(repeats):
            noisy = add_noise(X_clean, noise_type, level, rng)
            recon = net.forward(noisy)
            maxes.append(max_pixel_error(X_clean, recon))
            means.append(mean_pixel_error(X_clean, recon))
        rows.append({
            "level": level,
            "max_pixel_error": float(np.mean(maxes)),
            "mean_pixel_error": float(np.mean(means)),
        })
    return pd.DataFrame(rows)
