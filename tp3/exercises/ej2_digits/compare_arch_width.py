"""Evalúa y compara modelos de una sola capa oculta con distinto ancho.

Ejemplos:
    uv run python -m exercises.ej2_digits.compare_arch_width
    uv run python -m exercises.ej2_digits.compare_arch_width --only-comparison
"""

import argparse
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
_OUT_DIR = _ROOT / "outputs" / "ej2_digits" / "evaluation" / "comparison_arch_width"

# nombre → neuronas en la capa oculta
_MODELS = {
    "arch_narrow":   32,
    "arch_medium":   64,
    "arch_wide":     128,
    "arch_wider":    256,
    "arch_widest":   512,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara modelos de 1 capa oculta con distinto ancho."
    )
    parser.add_argument("--only-comparison", action="store_true",
                        help="Omite las evaluaciones individuales y solo genera los gráficos.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET,
                        help="Path al dataset para la evaluación.")
    return parser.parse_args()


def _plot_accuracy_vs_width(all_results, out_path):
    widths, accs, names = [], [], []
    for name, width in _MODELS.items():
        if name not in all_results:
            continue
        widths.append(width)
        accs.append(all_results[name]["accuracy"])
        names.append(name)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(widths, accs, marker="o", color="steelblue", linewidth=2, markersize=8)

    for x, y, name in zip(widths, accs, names):
        ax.annotate(f"{y:.4f}\n({name})", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)

    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.margins(x=0.15)
    ax.set_xlabel("Neuronas en la capa oculta")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy en test vs ancho de la capa oculta (1 capa)")
    ax.set_ylim(min(accs) - 0.002, max(accs) + 0.005)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def _plot_gap_vs_width(curves, out_path):
    widths, gaps, train_finals, val_finals = [], [], [], []

    for name, width in _MODELS.items():
        if name not in curves:
            continue
        df = curves[name]
        if "loss_train" not in df.columns or "loss_val" not in df.columns:
            continue
        train_final = float(df["loss_train"].iloc[-1])
        val_final = float(df["loss_val"].iloc[-1])
        widths.append(width)
        train_finals.append(train_final)
        val_finals.append(val_final)
        gaps.append(val_final - train_final)

    if not widths:
        print("  [WARN] No hay datos de curvas para graficar el gap.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(widths, train_finals, marker="o", color="steelblue", linewidth=2,
                 markersize=8, label="Train loss final")
    axes[0].plot(widths, val_finals, marker="s", color="tomato", linewidth=2,
                 markersize=8, linestyle="--", label="Val loss final")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(widths)
    axes[0].set_xticklabels([str(w) for w in widths])
    axes[0].set_xlabel("Neuronas en la capa oculta")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Val loss final por ancho")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    colors = ["tomato" if g > 0 else "steelblue" for g in gaps]
    bars = axes[1].bar([str(w) for w in widths], gaps, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Neuronas en la capa oculta")
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
    model_names = list(_MODELS.keys())

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

    _plot_accuracy_global(all_results, _OUT_DIR / "accuracy_global.png")
    _plot_per_class_heatmap(all_results, _OUT_DIR / "accuracy_per_class_heatmap.png")
    _plot_loss_curves(curves, _OUT_DIR / "loss_curves.png")
    _plot_accuracy_vs_width(all_results, _OUT_DIR / "accuracy_vs_width.png")
    _plot_gap_vs_width(curves, _OUT_DIR / "gap_vs_width.png")

    print(f"\nComparación guardada en {_OUT_DIR}")


if __name__ == "__main__":
    main()
