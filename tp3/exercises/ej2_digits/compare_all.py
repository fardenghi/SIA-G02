"""Evalúa todos los modelos entrenados y genera gráficos de comparación.

Ejemplos:
    uv run python -m exercises.ej2_digits.compare_all
    uv run python -m exercises.ej2_digits.compare_all --only-comparison
    uv run python -m exercises.ej2_digits.compare_all --dataset data/ej2_digits/digits_test.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exercises.ej2_digits.evaluate import run as evaluate_model

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej2_digits" / "models"
_EVAL_DIR = _ROOT / "outputs" / "ej2_digits" / "evaluation"
_METRICS_DIR = _ROOT / "outputs" / "ej2_digits" / "metrics"
_COMPARISON_DIR = _EVAL_DIR / "comparison"
_DEFAULT_DATASET = _ROOT / "data" / "ej2_digits" / "digits_test.csv"
_N_CLASSES = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa todos los modelos y compara sus resultados."
    )
    parser.add_argument("--only-comparison", action="store_true",
                        help="Omite las evaluaciones individuales y solo genera los gráficos de comparación.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET,
                        help="Path al dataset para la evaluación.")
    return parser.parse_args()


def _load_all_metrics(model_names):
    """Carga los metrics.json de cada modelo. Retorna dict model_name → results."""
    all_results = {}
    for name in model_names:
        json_path = _EVAL_DIR / name / "metrics.json"
        if not json_path.exists():
            print(f"  [WARN] No se encontró {json_path} — omitiendo {name}")
            continue
        with open(json_path) as f:
            all_results[name] = json.load(f)
    return all_results


def _load_all_loss_curves(model_names):
    """Carga los CSV de métricas de entrenamiento. Retorna dict model_name → DataFrame."""
    curves = {}
    for name in model_names:
        csv_path = _METRICS_DIR / f"{name}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        curves[name] = df
    return curves


def _plot_accuracy_global(all_results, out_path):
    names = list(all_results.keys())
    accs = [all_results[n]["accuracy"] for n in names]

    order = np.argsort(accs)[::-1]
    names = [names[i] for i in order]
    accs = [accs[i] for i in order]

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


def _plot_per_class_heatmap(all_results, out_path):
    names = sorted(all_results.keys())
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


def _plot_loss_curves(curves, out_path):
    if not curves:
        print("  [WARN] No se encontraron CSVs de métricas — omitiendo curvas de loss.")
        return

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, df) in enumerate(sorted(curves.items())):
        color = colors[i % len(colors)]
        if "epoch" in df.columns and "loss_train" in df.columns:
            axes[0].plot(df["epoch"], df["loss_train"], label=name, color=color)
        if "epoch" in df.columns and "loss_val" in df.columns:
            axes[1].plot(df["epoch"], df["loss_val"], label=name, color=color)

    for ax, title in zip(axes, ["Loss de entrenamiento", "Loss de validación"]):
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.set_title(f"Comparación — {title}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def main():
    args = parse_args()

    model_names = sorted(p.stem for p in _MODELS_DIR.glob("*.npz"))
    if not model_names:
        raise SystemExit(f"No se encontraron modelos en {_MODELS_DIR}")

    print(f"Modelos encontrados: {', '.join(model_names)}")
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

    _plot_accuracy_global(all_results, _COMPARISON_DIR / "accuracy_global.png")
    _plot_per_class_heatmap(all_results, _COMPARISON_DIR / "accuracy_per_class_heatmap.png")
    _plot_loss_curves(curves, _COMPARISON_DIR / "loss_curves.png")

    print(f"\nComparación guardada en {_COMPARISON_DIR}")


if __name__ == "__main__":
    main()
