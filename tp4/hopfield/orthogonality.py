"""Análisis de ortogonalidad entre letras del abecedario.

Para Hopfield la capacidad y la calidad de la recuperación dependen fuertemente
de cuán ortogonales sean los patrones almacenados entre sí. Aquí calculamos:

- Matriz 26x26 de productos internos entre todas las letras (heatmap).
- Para una cardinalidad k dada (típicamente 4), recorremos las C(26,k)
  combinaciones posibles y rankeamos según:
    * max_abs_dot: max |<xi, xj>| sobre todos los pares i<j  (menor = mejor)
    * mean_abs_dot: promedio de |<xi, xj>|
- Reporta los mejores y peores conjuntos y exporta CSV completo.
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


def alphabet_matrix() -> np.ndarray:
    return np.stack([letter_vector(ch) for ch in LETTERS]).astype(np.int32)


def pairwise_dot_matrix() -> np.ndarray:
    A = alphabet_matrix()
    return A @ A.T  # 26x26


def plot_dot_heatmap(matrix: np.ndarray, output: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    abs_off_diag = np.abs(matrix.copy())
    np.fill_diagonal(abs_off_diag, 0)
    im = ax.imshow(abs_off_diag, cmap="Reds", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="|<xi, xj>|")
    ax.set_xticks(range(len(LETTERS)))
    ax.set_yticks(range(len(LETTERS)))
    ax.set_xticklabels(LETTERS, fontsize=8)
    ax.set_yticklabels(LETTERS, fontsize=8)
    ax.set_title("Matriz de |<xi, xj>| entre letras (diag = 0)")
    for i in range(len(LETTERS)):
        for j in range(len(LETTERS)):
            if i != j:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=5, color="black")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def combo_metrics(combo: tuple[str, ...], dot: np.ndarray) -> tuple[int, float]:
    idx = [LETTERS.index(c) for c in combo]
    sub = dot[np.ix_(idx, idx)]
    abs_off = np.abs(sub.copy())
    np.fill_diagonal(abs_off, 0)
    pairs = abs_off[np.triu_indices_from(abs_off, k=1)]
    if pairs.size == 0:
        return 0, 0.0
    return int(pairs.max()), float(pairs.mean())


def rank_combinations(k: int, dot: np.ndarray) -> pd.DataFrame:
    rows = []
    for combo in itertools.combinations(LETTERS, k):
        max_abs, mean_abs = combo_metrics(combo, dot)
        rows.append({
            "combo": "".join(combo),
            "max_abs_dot": max_abs,
            "mean_abs_dot": mean_abs,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["max_abs_dot", "mean_abs_dot"]).reset_index(drop=True)
    return df


def plot_top_bottom(df: pd.DataFrame, k: int, output: str, n: int = 15) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=False)

    top = df.head(n).iloc[::-1]
    ax1.barh(top["combo"], top["max_abs_dot"], color="seagreen")
    ax1.set_xlabel("max |<xi, xj>|")
    ax1.set_title(f"Top {n} más ortogonales (k={k})")
    for i, (m_max, m_mean) in enumerate(zip(top["max_abs_dot"], top["mean_abs_dot"])):
        ax1.text(m_max + 0.1, i, f"μ={m_mean:.2f}", va="center", fontsize=8)

    bottom = df.tail(n).iloc[::-1]
    ax2.barh(bottom["combo"], bottom["max_abs_dot"], color="indianred")
    ax2.set_xlabel("max |<xi, xj>|")
    ax2.set_title(f"Bottom {n} menos ortogonales (k={k})")
    for i, (m_max, m_mean) in enumerate(zip(bottom["max_abs_dot"], bottom["mean_abs_dot"])):
        ax2.text(m_max + 0.1, i, f"μ={m_mean:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, k: int, output: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(df["max_abs_dot"], bins=range(0, int(df["max_abs_dot"].max()) + 2),
             color="steelblue", edgecolor="black")
    ax1.set_xlabel("max |<xi, xj>|")
    ax1.set_ylabel("Cantidad de combinaciones")
    ax1.set_title(f"Distribución max|dot| sobre C(26,{k})={len(df)}")
    ax1.grid(True, alpha=0.3)

    ax2.hist(df["mean_abs_dot"], bins=30, color="darkorange", edgecolor="black")
    ax2.set_xlabel("mean |<xi, xj>|")
    ax2.set_ylabel("Cantidad de combinaciones")
    ax2.set_title(f"Distribución mean|dot|")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Análisis de ortogonalidad del abecedario")
    parser.add_argument("--k", type=int, default=4, help="Tamaño del subconjunto")
    parser.add_argument("--output-dir", default="output/hopfield/orthogonality")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dot = pairwise_dot_matrix()
    plot_dot_heatmap(dot, os.path.join(args.output_dir, "dot_heatmap.png"))

    df = rank_combinations(args.k, dot)
    df.to_csv(os.path.join(args.output_dir, f"combos_k{args.k}.csv"), index=False)

    plot_top_bottom(df, args.k,
                    os.path.join(args.output_dir, f"top_bottom_k{args.k}.png"),
                    n=args.top)
    plot_distribution(df, args.k,
                      os.path.join(args.output_dir, f"distribution_k{args.k}.png"))

    print(f"\n=== Análisis de ortogonalidad (k={args.k}) ===")
    print(f"Combinaciones evaluadas: {len(df)}")
    print(f"\nTop {args.top} más ortogonales:")
    print(df.head(args.top).to_string(index=False))
    print(f"\nBottom {args.top} menos ortogonales:")
    print(df.tail(args.top).to_string(index=False))

    print(f"\nResultados en {args.output_dir}/")
    print(f"Mejor combinación (k={args.k}): {df.iloc[0]['combo']}  "
          f"(max|dot|={df.iloc[0]['max_abs_dot']}, mean|dot|={df.iloc[0]['mean_abs_dot']:.2f})")


if __name__ == "__main__":
    main()
