"""Evalúa y compara los modelos arch_Nhidden y genera gráficos incluyendo accuracy vs capas.

Ejemplos:
    uv run python -m exercises.ej2_digits.compare_arch_hidden
    uv run python -m exercises.ej2_digits.compare_arch_hidden --only-comparison
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

from exercises.ej2_digits.compare_all import (
    _load_all_loss_curves,
    _load_all_metrics,
    _plot_accuracy_global,
    _plot_loss_curves,
    _plot_per_class_heatmap,
)
from exercises.ej2_digits.evaluate import run as evaluate_model

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _ROOT / "data" / "ej2_digits" / "digits_test.csv"
_OUT_DIR = _ROOT / "outputs" / "ej2_digits" / "evaluation" / "comparison_arch_hidden"

_MODELS = ["arch_1hidden", "arch_2hidden", "arch_3hidden_2", "arch_4hidden", "arch_5hidden"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara modelos arch_Nhidden y grafica accuracy vs cantidad de capas."
    )
    parser.add_argument("--only-comparison", action="store_true",
                        help="Omite las evaluaciones individuales y solo genera los gráficos.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET,
                        help="Path al dataset para la evaluación.")
    return parser.parse_args()


def _plot_accuracy_vs_layers(all_results, out_path):
    layers = []
    accs = []
    for name in _MODELS:
        if name not in all_results:
            continue
        n_layers = int(re.search(r"arch_(\d+)hidden", name).group(1))
        layers.append(n_layers)
        accs.append(all_results[name]["accuracy"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(layers, accs, marker="o", color="steelblue", linewidth=2, markersize=8)

    for x, y, name in zip(layers, accs, [n for n in _MODELS if n in all_results]):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xticks(layers)
    ax.set_xlabel("Cantidad de capas ocultas")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy en test vs cantidad de capas ocultas")
    ax.set_ylim(min(accs) - 0.002, max(accs) + 0.002)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def _plot_gap_vs_layers(curves, out_path):
    """Grafica el gap (val_loss_final - train_loss_final) por cantidad de capas ocultas."""
    layers = []
    gaps = []
    train_finals = []
    val_finals = []

    for name in _MODELS:
        if name not in curves:
            continue
        df = curves[name]
        if "loss_train" not in df.columns or "loss_val" not in df.columns:
            continue
        n_layers = int(re.search(r"arch_(\d+)hidden", name).group(1))
        train_final = float(df["loss_train"].iloc[-1])
        val_final = float(df["loss_val"].iloc[-1])
        layers.append(n_layers)
        train_finals.append(train_final)
        val_finals.append(val_final)
        gaps.append(val_final - train_final)

    if not layers:
        print("  [WARN] No hay datos de curvas para graficar el gap.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: train loss final y val loss final por capas
    axes[0].plot(layers, train_finals, marker="o", color="steelblue", linewidth=2,
                 markersize=8, label="Train loss final")
    axes[0].plot(layers, val_finals, marker="s", color="tomato", linewidth=2,
                 markersize=8, linestyle="--", label="Val loss final")
    axes[0].set_xticks(layers)
    axes[0].set_xlabel("Cantidad de capas ocultas")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Val loss final por arquitectura")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Subplot 2: gap val - train por capas
    colors = ["tomato" if g > 0 else "steelblue" for g in gaps]
    bars = axes[1].bar(layers, gaps, color=colors, width=0.4)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(layers)
    axes[1].set_xlabel("Cantidad de capas ocultas")
    axes[1].set_ylabel("Gap (val - train)")
    axes[1].set_title("Gap val−train loss final\n(rojo = val > train = posible overfitting)")
    axes[1].grid(axis="y", alpha=0.3)

    for bar, gap in zip(bars, gaps):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     gap + (0.0002 if gap >= 0 else -0.0004),
                     f"{gap:.4f}", ha="center", va="bottom" if gap >= 0 else "top", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def main():
    args = parse_args()

    print(f"Modelos: {', '.join(_MODELS)}")
    print("=" * 55)

    if not args.only_comparison:
        for name in _MODELS:
            print(f"\n[{name}]")
            try:
                evaluate_model(name, dataset_path=args.dataset)
            except FileNotFoundError as e:
                print(f"  [ERROR] {e}")

    print("\n" + "=" * 55)
    print("Generando gráficos de comparación...")

    all_results = _load_all_metrics(_MODELS)
    if not all_results:
        raise SystemExit("No hay metrics.json disponibles. Corré primero sin --only-comparison.")

    curves = _load_all_loss_curves(_MODELS)

    _plot_accuracy_global(all_results, _OUT_DIR / "accuracy_global.png")
    _plot_per_class_heatmap(all_results, _OUT_DIR / "accuracy_per_class_heatmap.png")
    _plot_loss_curves(curves, _OUT_DIR / "loss_curves.png")
    _plot_accuracy_vs_layers(all_results, _OUT_DIR / "accuracy_vs_layers.png")
    _plot_gap_vs_layers(curves, _OUT_DIR / "gap_vs_layers.png")

    print(f"\nComparación guardada en {_OUT_DIR}")


if __name__ == "__main__":
    main()
