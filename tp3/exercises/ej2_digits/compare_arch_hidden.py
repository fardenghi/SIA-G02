"""Evalúa y compara los modelos arch_Nhidden y genera gráficos incluyendo accuracy vs capas.

Ejemplos:
    uv run python -m exercises.ej2_digits.compare_arch_hidden
    uv run python -m exercises.ej2_digits.compare_arch_hidden --only-comparison
    uv run python -m exercises.ej2_digits.compare_arch_hidden --models arch_1hidden_128:1 arch_2hidden_128_64:2 arch_3hidden_128_64_32:3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from exercises.ej2_digits.compare_all import (
    _load_all_loss_curves,
    _load_all_metrics,
    _plot_loss_curves,
)
from exercises.ej2_digits.evaluate import run as evaluate_model

_N_CLASSES = 10


def _plot_accuracy_global(all_results, models, out_path):
    names = [n for n in models if n in all_results]
    accs = [all_results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.6)))
    bars = ax.barh(names, accs, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Comparación — Accuracy global por modelo")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def _plot_per_class_heatmap(all_results, models, out_path):
    names = [n for n in models if n in all_results]
    matrix = np.array([
        [all_results[n]["per_class_accuracy"].get(str(c), 0.0) for c in range(_N_CLASSES)]
        for n in names
    ])

    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.7)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xticks(range(_N_CLASSES))
    ax.set_xticklabels([str(c) for c in range(_N_CLASSES)])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Dígito")
    ax.set_title("Comparación — Accuracy por clase y modelo")

    for i, name in enumerate(names):
        for j in range(_N_CLASSES):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _ROOT / "data" / "ej2_digits" / "digits_test.csv"
_OUT_DIR = _ROOT / "outputs" / "ej2_digits" / "evaluation" / "comparison_arch_hidden"

_DEFAULT_MODELS = {
    "arch_1hidden": 1,
    "arch_2hidden": 2,
    "arch_3hidden_2": 3,
    "arch_4hidden": 4,
    "arch_5hidden": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara modelos arch_Nhidden y grafica accuracy vs cantidad de capas."
    )
    parser.add_argument("--only-comparison", action="store_true",
                        help="Omite las evaluaciones individuales y solo genera los gráficos.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET,
                        help="Path al dataset para la evaluación.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Modelos en formato nombre:n_capas, ej: arch_1hidden_128:1 arch_2hidden_128_64:2")
    parser.add_argument("--out-dir", type=Path, default=_OUT_DIR,
                        help="Directorio de salida para los gráficos.")
    return parser.parse_args()


def _parse_models(models_arg):
    result = {}
    for item in models_arg:
        if ":" not in item:
            raise SystemExit(f"Formato inválido '{item}'. Usar nombre:n_capas, ej: arch_1hidden_128:1")
        name, n_layers = item.rsplit(":", 1)
        result[name] = int(n_layers)
    return result


def _plot_accuracy_vs_layers(all_results, models, out_path):
    layers, accs, names = [], [], []
    for name, n_layers in models.items():
        if name not in all_results:
            continue
        layers.append(n_layers)
        accs.append(all_results[name]["accuracy"])
        names.append(name)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(layers, accs, marker="o", color="steelblue", linewidth=2, markersize=8)

    for x, y in zip(layers, accs):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xticks(layers)
    ax.set_xlabel("Cantidad de capas ocultas")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy en test vs cantidad de capas ocultas")
    ax.set_ylim(min(accs) - 0.05, max(accs) + 0.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def _plot_gap_vs_layers(curves, models, out_path):
    layers, gaps, train_finals, val_finals = [], [], [], []

    for name, n_layers in models.items():
        if name not in curves:
            continue
        df = curves[name]
        if "loss_train" not in df.columns or "loss_val" not in df.columns:
            continue
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

    colors = ["tomato" if g > 0 else "steelblue" for g in gaps]
    bars = axes[1].bar(layers, gaps, color=colors, width=0.4)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(layers)
    axes[1].set_xlabel("Cantidad de capas ocultas")
    axes[1].set_ylabel("Gap (val - train)")
    axes[1].set_title("Gap val−train loss final")
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
    models = _parse_models(args.models) if args.models else _DEFAULT_MODELS
    model_names = list(models.keys())
    out_dir = args.out_dir

    print(f"Modelos: {', '.join(model_names)}")
    print("=" * 55)

    if not args.only_comparison:
        for name in model_names:
            print(f"\n[{name}]")
            try:
                evaluate_model(name, dataset_path=args.dataset)
            except FileNotFoundError as e:
                print(f"  [ERROR] {e}")

    print("\n" + "=" * 55)
    print("Generando gráficos de comparación...")

    all_results = _load_all_metrics(model_names)
    if not all_results:
        raise SystemExit("No hay metrics.json disponibles. Corré primero sin --only-comparison.")

    curves = _load_all_loss_curves(model_names)

    _plot_accuracy_global(all_results, models, out_dir / "accuracy_global.png")
    _plot_per_class_heatmap(all_results, models, out_dir / "accuracy_per_class_heatmap.png")
    _plot_loss_curves(curves, out_dir / "loss_curves.png")
    _plot_accuracy_vs_layers(all_results, models, out_dir / "accuracy_vs_layers.png")
    _plot_gap_vs_layers(curves, models, out_dir / "gap_vs_layers.png")

    print(f"\nComparación guardada en {out_dir}")


if __name__ == "__main__":
    main()
