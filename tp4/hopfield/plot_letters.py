"""Grafica las letras del abecedario en grillas 5x5.

Uso:
    uv run python -m hopfield.plot_letters --start c --end h
        -> grafica c d e f g h

    uv run python -m hopfield.plot_letters --start a --end z --output output/hopfield/abecedario.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hopfield.alphabet import ALPHABET, letters_in_range


def plot_letter(ax, letter: str, matrix: np.ndarray) -> None:
    img = np.where(matrix == 1, 1.0, 0.0)
    ax.imshow(img, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(letter, fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_letters(letters: list[str], output: str, title: str | None = None) -> None:
    n = len(letters)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.atleast_2d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for ax, letter in zip(axes_flat, letters):
        ax.axis("on")
        plot_letter(ax, letter, ALPHABET[letter])

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Guardado: {output}  ({n} letras: {''.join(letters)})")


def main():
    parser = argparse.ArgumentParser(description="Plot del abecedario 5x5")
    parser.add_argument("--start", required=True, help="Letra inicial (ej: c)")
    parser.add_argument("--end", required=True, help="Letra final (ej: h)")
    parser.add_argument("--output", default=None, help="Path de salida del .png")
    args = parser.parse_args()

    letters = letters_in_range(args.start, args.end)
    output = args.output or f"output/hopfield/letters_{args.start.upper()}_{args.end.upper()}.png"
    plot_letters(letters, output, title=f"Letras {args.start.upper()} - {args.end.upper()}")


if __name__ == "__main__":
    main()
