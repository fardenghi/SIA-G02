"""¿Cómo varía la métrica de recuperación según la cantidad de patrones almacenados?

Se mide:
- recall_accuracy: fracción de queries ruidosas que terminan en el patrón original.
- spurious_rate: fracción que termina en un estado fijo distinto a cualquier patrón.
- avg_hamming_to_original: distancia de Hamming promedio entre estado final y patrón original.

Para cada k=1..k_max:
- elegimos un subconjunto de k letras (modos: "best" según ortogonalidad, "random", o "first")
- almacenamos en la red
- repetimos n_trials veces: tomamos cada patrón, le aplicamos ruido, recuperamos
- promediamos sobre todas las semillas/ensayos
"""
from __future__ import annotations

import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hopfield.alphabet import ALPHABET, LETTERS, letter_vector
from hopfield.hopfield import HopfieldNetwork, add_noise
from hopfield.orthogonality import pairwise_dot_matrix, rank_combinations


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def evaluate_set(
    letters: list[str],
    noise_levels: list[float],
    n_trials: int,
    max_steps: int,
    rng: np.random.Generator,
) -> dict:
    patterns = np.stack([letter_vector(c) for c in letters])
    net = HopfieldNetwork(n_units=patterns.shape[1])
    net.store(patterns)

    rows = []
    for noise in noise_levels:
        recall_hits = 0
        spurious_hits = 0
        total_hamming = 0
        total = 0
        for _ in range(n_trials):
            for idx, p in enumerate(patterns):
                noisy = add_noise(p, noise, rng)
                final, _hist, _en, _conv = net.recall(noisy, mode="sync",
                                                     max_steps=max_steps)
                if np.array_equal(final, p):
                    recall_hits += 1
                elif net.is_stored(final) == -1:
                    spurious_hits += 1
                total_hamming += _hamming(final, p)
                total += 1
        rows.append({
            "n_patterns": len(letters),
            "noise": noise,
            "recall_accuracy": recall_hits / total,
            "spurious_rate": spurious_hits / total,
            "avg_hamming_to_original": total_hamming / total,
        })
    return rows


def pick_combo(k: int, mode: str, dot: np.ndarray, rng: np.random.Generator) -> list[str]:
    if mode == "best":
        df = rank_combinations(k, dot)
        return list(df.iloc[0]["combo"])
    if mode == "worst":
        df = rank_combinations(k, dot)
        return list(df.iloc[-1]["combo"])
    if mode == "random":
        idx = rng.choice(len(LETTERS), size=k, replace=False)
        return [LETTERS[i] for i in idx]
    if mode == "first":
        return LETTERS[:k]
    raise ValueError(f"mode inválido: {mode}")


def run_sweep(
    k_max: int,
    modes: list[str],
    noise_levels: list[float],
    n_trials: int,
    max_steps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dot = pairwise_dot_matrix()
    all_rows = []
    for mode in modes:
        for k in range(1, k_max + 1):
            letters = pick_combo(k, mode, dot, rng)
            rows = evaluate_set(letters, noise_levels, n_trials, max_steps, rng)
            for r in rows:
                r["mode"] = mode
                r["letters"] = "".join(letters)
            all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def plot_accuracy_vs_n(df: pd.DataFrame, output_dir: str) -> None:
    noises = sorted(df["noise"].unique())
    modes = df["mode"].unique()

    fig, axes = plt.subplots(1, len(noises), figsize=(4.5 * len(noises), 4.5),
                             sharey=True)
    if len(noises) == 1:
        axes = [axes]
    for ax, noise in zip(axes, noises):
        for mode in modes:
            sub = df[(df["noise"] == noise) & (df["mode"] == mode)] \
                .sort_values("n_patterns")
            ax.plot(sub["n_patterns"], sub["recall_accuracy"],
                    marker="o", label=mode)
        ax.set_title(f"Ruido = {noise:.0%}")
        ax.set_xlabel("Cantidad de patrones almacenados")
        ax.set_ylabel("Recall accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Recall accuracy vs cantidad de patrones almacenados")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_vs_n.png"), dpi=150)
    plt.close(fig)


def plot_spurious_vs_n(df: pd.DataFrame, output_dir: str) -> None:
    noises = sorted(df["noise"].unique())
    modes = df["mode"].unique()

    fig, axes = plt.subplots(1, len(noises), figsize=(4.5 * len(noises), 4.5),
                             sharey=True)
    if len(noises) == 1:
        axes = [axes]
    for ax, noise in zip(axes, noises):
        for mode in modes:
            sub = df[(df["noise"] == noise) & (df["mode"] == mode)] \
                .sort_values("n_patterns")
            ax.plot(sub["n_patterns"], sub["spurious_rate"],
                    marker="s", label=mode)
        ax.set_title(f"Ruido = {noise:.0%}")
        ax.set_xlabel("Cantidad de patrones almacenados")
        ax.set_ylabel("Tasa de estados espúreos")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Estados espúreos vs cantidad de patrones almacenados")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "spurious_vs_n.png"), dpi=150)
    plt.close(fig)


def plot_hamming_vs_n(df: pd.DataFrame, output_dir: str) -> None:
    noises = sorted(df["noise"].unique())
    modes = df["mode"].unique()

    fig, axes = plt.subplots(1, len(noises), figsize=(4.5 * len(noises), 4.5),
                             sharey=True)
    if len(noises) == 1:
        axes = [axes]
    for ax, noise in zip(axes, noises):
        for mode in modes:
            sub = df[(df["noise"] == noise) & (df["mode"] == mode)] \
                .sort_values("n_patterns")
            ax.plot(sub["n_patterns"], sub["avg_hamming_to_original"],
                    marker="^", label=mode)
        ax.set_title(f"Ruido = {noise:.0%}")
        ax.set_xlabel("Cantidad de patrones almacenados")
        ax.set_ylabel("Hamming promedio (bits)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Distancia de Hamming al patrón original vs cantidad de patrones")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hamming_vs_n.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Capacidad: métrica vs N de patrones")
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--modes", nargs="+", default=["best", "worst", "random"])
    parser.add_argument("--noise", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/hopfield/capacity")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = run_sweep(args.k_max, args.modes, args.noise, args.trials,
                   args.max_steps, args.seed)
    df.to_csv(os.path.join(args.output_dir, "capacity.csv"), index=False)

    plot_accuracy_vs_n(df, args.output_dir)
    plot_spurious_vs_n(df, args.output_dir)
    plot_hamming_vs_n(df, args.output_dir)

    print("\n=== Sweep de capacidad ===")
    print(df.to_string(index=False))
    print(f"\nResultados en {args.output_dir}/")


if __name__ == "__main__":
    main()
