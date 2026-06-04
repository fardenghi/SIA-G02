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

from hopfield.alphabet import ALPHABET, letters_in_range, scale_pattern


def plot_letter(ax, letter: str, matrix: np.ndarray) -> None:
    grid = matrix.shape[0]
    img = np.where(matrix == 1, 1.0, 0.0)
    ax.imshow(img, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(letter, fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, grid, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_letters(letters: list[str], output: str, title: str | None = None, scale: int = 1) -> None:
    n = len(letters)
    cols = min(n, 9)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.1))
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.atleast_2d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for ax, letter in zip(axes_flat, letters):
        ax.axis("on")
        matrix = scale_pattern(ALPHABET[letter], scale) if scale > 1 else ALPHABET[letter]
        plot_letter(ax, letter, matrix)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    n_last = n % cols
    if n_last != 0:
        x0 = axes_flat[0].get_position().x0
        x1 = axes_flat[cols - 1].get_position().x1
        ax_width = axes_flat[0].get_position().width
        spacing = (x1 - x0 - cols * ax_width) / (cols - 1) if cols > 1 else 0
        start_x = x0 + ((x1 - x0) - (n_last * ax_width + (n_last - 1) * spacing)) / 2
        for i in range(n_last):
            ax = axes_flat[(rows - 1) * cols + i]
            pos = ax.get_position()
            ax.set_position([start_x + i * (ax_width + spacing), pos.y0, ax_width, pos.height])

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Guardado: {output}  ({n} letras: {''.join(letters)})")


def main():
    parser = argparse.ArgumentParser(description="Plot del abecedario 5x5")
    parser.add_argument("--start", required=True, help="Letra inicial (ej: c)")
    parser.add_argument("--end", required=True, help="Letra final (ej: h)")
    parser.add_argument("--output", default=None, help="Path de salida del .png")
    parser.add_argument("--scale", type=int, default=1, help="Factor de escala (ej: 3 para 15x15)")
    args = parser.parse_args()

    letters = letters_in_range(args.start, args.end)
    output = args.output or f"output/hopfield/letters_{args.start.upper()}_{args.end.upper()}_k{args.scale}.png"
    plot_letters(letters, output, title=f"Letras {args.start.upper()} - {args.end.upper()}", scale=args.scale)


if __name__ == "__main__":
    main()
