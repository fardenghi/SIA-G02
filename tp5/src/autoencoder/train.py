"""Entrenamiento full-batch, multi-restart, métricas de píxeles y MetricsTracker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import losses as losses_mod
from .data import add_noise
from .network import Autoencoder
from .optim import Adam, lbfgs_minimize

# Seeds fijas e independientes de `training.seed`: la evaluación final y la validación
# (selección de restarts) usan siempre el MISMO set de corrupciones en todos los runs,
# y son sets distintos entre sí para que la selección no se ajuste al set de reporte.
VAL_SEED = 777
VAL_REPEATS = 30
EVAL_SEED = 1234
EVAL_REPEATS = 50


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


def _sample_level(rng: np.random.Generator, level: float,
                  train_level_range: list[float] | None) -> float:
    """Nivel de ruido por época: uniforme en el rango si está, sino el fijo `level`."""
    if train_level_range is not None:
        lo, hi = train_level_range
        return float(rng.uniform(lo, hi))
    return level


def _corrupt_for_epoch(denoise: dict, X_target: np.ndarray) -> np.ndarray:
    """Genera entradas ruidosas frescas a partir del objetivo limpio (corrupción online).

    Con `replicas=k > 1` apila `k` corrupciones independientes de cada patrón (batch
    `k*n`), cada réplica con su propio nivel sampleado: la red ve más realizaciones de
    ruido por paso de gradiente.
    """
    k = int(denoise.get("replicas") or 1)
    blocks = []
    for _ in range(k):
        lvl = _sample_level(denoise["rng"], denoise["level"],
                            denoise.get("train_level_range"))
        blocks.append(add_noise(X_target, denoise["noise_type"], lvl, denoise["rng"]))
    return blocks[0] if k == 1 else np.vstack(blocks)


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
    denoise: dict | None = None,
) -> dict:
    """Entrena `net` in-place sobre el par `(X_input, X_target)` en full-batch.

    Para denoising estático, `X_input` es la versión ruidosa y `X_target` la limpia.
    Si `denoise` (dict con `noise_type`, `level`, `train_level_range`, `rng`) está
    presente, se ignora `X_input` y se regenera una entrada ruidosa fresca por época a
    partir de `X_target` (corrupción online; requiere Adam). Devuelve métricas finales.
    """
    if denoise is not None and optimizer != "adam":
        raise ValueError(
            "La corrupción online (denoise) requiere optimizer='adam'; "
            "el objetivo estocástico es incompatible con L-BFGS.")
    loss_value, loss_grad = losses_mod.get_loss(loss)

    def record(epoch: int, out: np.ndarray, target: np.ndarray) -> None:
        if tracker is not None and (epoch % log_every == 0):
            tracker.log(epoch, loss_value(out, target),
                        max_pixel_error(target, out),
                        mean_pixel_error(target, out))

    if optimizer == "adam":
        opt = Adam(lr=lr)
        params = net.get_params()
        denom = max(1, epochs - 1)
        # Con réplicas ruidosas, el objetivo limpio se tilea para alinear con el batch.
        k = int(denoise.get("replicas") or 1) if denoise is not None else 1
        X_tgt_train = np.tile(X_target, (k, 1)) if k > 1 else X_target
        for epoch in range(epochs):
            # Scheduler de lr (solo Adam). Cosine annealing de `lr` a `lr_min`.
            if lr_schedule == "cosine":
                opt.lr = lr_min + 0.5 * (lr - lr_min) * (
                    1.0 + np.cos(np.pi * epoch / denom))
            net.set_params(params)
            # Corrupción online: ruido fresco por época contra el target limpio.
            X_in = _corrupt_for_epoch(denoise, X_target) if denoise is not None else X_input
            out = net.forward(X_in)
            net.backward(loss_grad(out, X_tgt_train))
            grads = net.get_grads()
            params = opt.step(params, grads)
            record(epoch, out, X_tgt_train)
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
            record(state["epoch"], out, X_target)
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
    select_by_denoising: bool = False,
    decoder_layers: list[int] | None = None,
    return_candidates: bool = False,
):
    """Entrena `restarts` modelos con semillas distintas; conserva el de menor
    `max_pixel_error`. Devuelve `(mejor_red, mejor_tracker, resumen_df)`.

    `stop_at`: si un restart alcanza `max_pixel_error <= stop_at`, se abortan los
    restarts restantes (ya se encontró un modelo suficientemente bueno). Con
    `stop_at=0` (default) corta apenas un modelo reconstruye los 32/32 sin error;
    `None` desactiva el corte y corre los `restarts` completos.

    `select_by_denoising`: con denoising habilitado, varios restarts reconstruyen el
    set limpio igual de bien (mismo `max_pixel_error`), pero denoisean distinto. Si está
    activo, entre los restarts con la mejor reconstrucción limpia se elige el de menor
    **mean_pixel_error** sobre un set de validación ruidoso FIJO (`VAL_SEED`,
    independiente de `seed` y del set de evaluación final; mejora gratis, sin
    reentrenar). Desactiva el corte temprano (necesita todos los restarts).

    `return_candidates`: si está activo devuelve además la lista de candidatos
    `(red, tracker, max_pixel_error_limpio)` de todos los restarts ejecutados.
    """
    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 2**31 - 1, size=restarts)

    best_net = None
    best_tracker = None
    best_mpe = np.inf
    summary_rows = []
    candidates = []  # (net, tracker, clean_max_pix) por restart, para selección final
    # La selección por denoising compara todos los restarts -> sin corte temprano.
    use_denoise_select = bool(
        select_by_denoising and denoising and denoising.get("enabled"))

    for r, s in enumerate(seeds):
        s = int(s)
        net = Autoencoder(encoder_layers, activation, output_activation, init, seed=s,
                          latent_activation=latent_activation,
                          decoder_layers=decoder_layers)

        # Entrada/objetivo: con denoising, entrada ruidosa y objetivo limpio.
        # `rng` de ruido determinista por restart (reproducibilidad).
        X_target = X
        X_input = X
        denoise_spec = None
        if denoising and denoising.get("enabled"):
            noise_rng = np.random.default_rng(s + 1)
            if denoising.get("resample_per_epoch", True):
                # Corrupción online: train_once regenera el ruido por época.
                denoise_spec = {
                    "noise_type": denoising.get("noise_type", "salt_pepper"),
                    "level": denoising.get("level", 0.1),
                    "train_level_range": denoising.get("train_level_range"),
                    "replicas": denoising.get("replicas", 1),
                    "rng": noise_rng,
                }
            else:
                # Corrupción estática: una única realización fija por restart.
                X_input = add_noise(X, denoising.get("noise_type", "salt_pepper"),
                                    denoising.get("level", 0.1), noise_rng)

        tracker = MetricsTracker(run_label=f"{optimizer}_r{r}")
        final = train_once(net, X_input, X_target, loss, optimizer, epochs, lr,
                           tracker=tracker, log_every=log_every,
                           lr_schedule=lr_schedule, lr_min=lr_min,
                           denoise=denoise_spec)
        summary_rows.append({"restart": r, "seed": s, **final})
        if verbose:
            print(f"  restart {r:2d} (seed {s}): "
                  f"loss={final['loss']:.4f} max_pix={final['max_pixel_error']} "
                  f"mean_pix={final['mean_pixel_error']:.3f}")

        candidates.append((net, tracker, final["max_pixel_error"]))
        if final["max_pixel_error"] < best_mpe:
            best_mpe = final["max_pixel_error"]
            best_net = net
            best_tracker = tracker

        # Corte temprano: si ya alcanzamos el umbral objetivo, no seguimos.
        # (La selección por denoising lo desactiva: necesita todos los restarts.)
        if not use_denoise_select and stop_at is not None and best_mpe <= stop_at:
            if verbose:
                print(f"  -> corte temprano en restart {r}: "
                      f"max_pixel_error={best_mpe} <= stop_at={stop_at} "
                      f"({r + 1}/{len(seeds)} restarts ejecutados)")
            break

    if use_denoise_select:
        # Entre los restarts que mejor reconstruyen el set limpio, quedarse con el de
        # menor error de píxeles sobre el set de validación ruidoso FIJO (VAL_SEED,
        # independiente del seed de entrenamiento y del set de evaluación final).
        best_clean = min(mpe for (_, _, mpe) in candidates)
        pool = [(net, tr) for (net, tr, mpe) in candidates if mpe == best_clean]
        levels = denoising.get("sweep_levels", [0.05, 0.1, 0.2, 0.3])
        ntype = denoising.get("noise_type", "salt_pepper")
        best_score = np.inf
        for net, tr in pool:
            sweep = evaluate_denoising(net, X, ntype, levels=levels, seed=VAL_SEED,
                                       repeats=VAL_REPEATS)
            score = float(sweep["mean_pixel_error"].mean())
            if score < best_score:
                best_score, best_net, best_tracker = score, net, tr
        if verbose:
            print(f"  -> selección por denoising: mejor de {len(pool)} restart(s) con "
                  f"reconstrucción limpia max_pix={best_clean} "
                  f"(mean_pixel_error validación={best_score:.3f})")

    summary = pd.DataFrame(summary_rows)
    if return_candidates:
        return best_net, best_tracker, summary, candidates
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
        maxes, means, perfects = [], [], []
        for _ in range(repeats):
            noisy = add_noise(X_clean, noise_type, level, rng)
            recon = net.forward(noisy)
            errs = pixel_errors(X_clean, recon)
            maxes.append(int(errs.max()))
            means.append(float(errs.mean()))
            # Fracción de patrones reconstruidos sin ningún píxel de error.
            perfects.append(float(np.mean(errs == 0)))
        rows.append({
            "level": level,
            "max_pixel_error": float(np.mean(maxes)),
            "mean_pixel_error": float(np.mean(means)),
            "perfect_pct": float(np.mean(perfects)) * 100.0,
        })
    return pd.DataFrame(rows)
