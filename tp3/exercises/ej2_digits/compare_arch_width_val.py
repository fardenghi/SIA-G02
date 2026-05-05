"""Compara modelos de una sola capa oculta con distinto ancho usando val accuracy de los CSVs.

Ejemplos:
    uv run python -m exercises.ej2_digits.compare_arch_width_val --models arch_1hidden_16:16 arch_1hidden_32:32 arch_1hidden_64:64 arch_1hidden_128:128 arch_1hidden_256:256 arch_1hidden_512:512
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from exercises.ej2_digits.compare_all import _load_all_loss_curves, _plot_loss_curves
from exercises.ej2_digits.compare_all_val import _get_val_accuracy, _plot_val_accuracy_curves

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIGS_DIR = _ROOT / "configs" / "ej2_digits"
_DEFAULT_OUT_DIR = _ROOT / "outputs" / "ej2_digits" / "evaluation" / "comparison_arch_width_val"

_DEFAULT_MODELS = {
    "arch_narrow":   32,
    "arch_medium":   64,
    "arch_wide":     128,
    "arch_wider":    256,
    "arch_widest":   512,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara modelos de 1 capa oculta con distinto ancho usando val accuracy."
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Modelos en formato nombre:ancho, ej: arch_1hidden_128:128 arch_1hidden_256:256")
    parser.add_argument("--configs-dir", type=Path, default=_DEFAULT_CONFIGS_DIR,
                        help="Directorio de configs JSON.")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help="Directorio de salida para los gráficos.")
    return parser.parse_args()


def _parse_models(models_arg):
    result = {}
    for item in models_arg:
        if ":" not in item:
            raise SystemExit(f"Formato inválido '{item}'. Usar nombre:ancho, ej: arch_1hidden_128:128")
        name, width = item.rsplit(":", 1)
        result[name] = int(width)
    return result


def _plot_val_accuracy_global(all_results, models, out_path):
    names = [n for n in models if n in all_results]
    accs = [all_results[n]["val_accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.6)))
    bars = ax.barh(names, accs, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Val Accuracy")
    ax.set_title("Comparación — Val Accuracy global por modelo")
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


def _plot_train_val_loss_vs_width(curves, models, out_path):
    widths, train_finals, val_finals = [], [], []

    for name, width in models.items():
        if name not in curves:
            continue
        df = curves[name]
        if "loss_train" not in df.columns or "loss_val" not in df.columns:
            continue
        widths.append(width)
        train_finals.append(float(df["loss_train"].iloc[-1]))
        val_finals.append(float(df["loss_val"].iloc[-1]))

    if not widths:
        print("  [WARN] No hay datos de curvas para graficar train vs val loss.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(widths, train_finals, marker="o", color="steelblue", linewidth=2,
            markersize=8, label="Train loss final")
    ax.plot(widths, val_finals, marker="s", color="tomato", linewidth=2,
            markersize=8, linestyle="--", label="Val loss final")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("Neuronas en la capa oculta")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val loss final por ancho")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def _plot_val_accuracy_vs_width(all_results, models, out_path):
    widths, accs = [], []
    for name, width in models.items():
        if name not in all_results:
            continue
        widths.append(width)
        accs.append(all_results[name]["val_accuracy"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(widths, accs, marker="o", color="steelblue", linewidth=2, markersize=8)

    for x, y in zip(widths, accs):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)

    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.margins(x=0.15)
    ax.set_xlabel("Neuronas en la capa oculta")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Val Accuracy vs ancho de la capa oculta (1 capa)")
    ax.set_ylim(min(accs) - 0.002, max(accs) + 0.005)
    ax.grid(alpha=0.3)

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

    all_results = {}
    for name in model_names:
        val_acc, meta = _get_val_accuracy(name, args.configs_dir)
        if val_acc is None:
            print(f"  [WARN] No se pudo calcular val accuracy para {name} — omitiendo.")
            continue
        es_str = f"early stopping (época {meta['epoch']})" if meta["early_stopping"] else f"última época ({meta['epoch']})"
        print(f"  {name}: val_acc={val_acc:.4f}  [{es_str}]")
        all_results[name] = {"val_accuracy": val_acc, **meta}

    if not all_results:
        raise SystemExit("No hay resultados disponibles.")

    print("\n" + "=" * 55)
    print("Generando gráficos...")

    curves = _load_all_loss_curves(model_names)

    _plot_val_accuracy_global(all_results, models, out_dir / "val_accuracy_global.png")
    _plot_val_accuracy_vs_width(all_results, models, out_dir / "val_accuracy_vs_width.png")
    _plot_loss_curves(curves, out_dir / "loss_curves.png")
    _plot_val_accuracy_curves(curves, out_dir / "val_accuracy_curves.png")
    _plot_train_val_loss_vs_width(curves, models, out_dir / "train_val_loss_vs_width.png")

    print(f"\nComparación guardada en {out_dir}")


if __name__ == "__main__":
    main()
