"""Compara modelos usando val accuracy de los CSVs de métricas (sin tocar digits_test.csv).

La val accuracy reportada depende de si hubo early stopping:
- Si hubo early stopping: acc_val en la época con mínimo loss_val.
- Si no hubo: acc_val de la última época.

Ejemplos:
    uv run python -m exercises.ej2_digits.compare_all_val --models opt_sgd_1hidden_128_lr0.01 opt_adam_1hidden_128_lr0.01
    uv run python -m exercises.ej2_digits.compare_all_val --models m1 m2 m3 --out-dir outputs/ej2_digits/evaluation/mi_comparacion
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exercises.ej2_digits.compare_all import _load_all_loss_curves, _plot_loss_curves

_ROOT = Path(__file__).resolve().parents[2]
_METRICS_DIR = _ROOT / "outputs" / "ej2_digits" / "metrics"
_DEFAULT_CONFIGS_DIR = _ROOT / "configs" / "ej2_digits"
_DEFAULT_OUT_DIR = _ROOT / "outputs" / "ej2_digits" / "evaluation" / "comparison_val"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara modelos usando val accuracy de los CSVs de métricas."
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Nombres de modelos (sin .npz). Default: todos los CSVs disponibles.")
    parser.add_argument("--configs-dir", type=Path, default=_DEFAULT_CONFIGS_DIR,
                        help="Directorio de configs JSON.")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                        help="Directorio de salida para los gráficos.")
    return parser.parse_args()


def _get_val_accuracy(model_name, configs_dir):
    csv_path = _METRICS_DIR / f"{model_name}.csv"
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    if "acc_val" not in df.columns or "loss_val" not in df.columns:
        return None, None

    config_path = configs_dir / f"{model_name}.json"
    early_stopping = False

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        max_epochs = cfg.get("epochs", None)
        if max_epochs is not None and df["epoch"].max() < max_epochs:
            early_stopping = True

    if early_stopping:
        best_idx = df["loss_val"].idxmin()
        val_acc = float(df.loc[best_idx, "acc_val"])
        best_epoch = int(df.loc[best_idx, "epoch"])
    else:
        val_acc = float(df["acc_val"].iloc[-1])
        best_epoch = int(df["epoch"].iloc[-1])

    return val_acc, {"early_stopping": early_stopping, "epoch": best_epoch}


def _plot_val_accuracy_global(all_results, out_path):
    names = list(all_results.keys())
    accs = [all_results[n]["val_accuracy"] for n in names]

    order = np.argsort(accs)[::-1]
    names = [names[i] for i in order]
    accs = [accs[i] for i in order]

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


def _plot_val_accuracy_curves(curves, out_path):
    if not curves:
        print("  [WARN] No hay CSVs de métricas — omitiendo curvas de val accuracy.")
        return

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, df) in enumerate(sorted(curves.items())):
        if "epoch" in df.columns and "acc_val" in df.columns:
            ax.plot(df["epoch"], df["acc_val"], label=name, color=colors[i % len(colors)])

    ax.set_xlabel("Época")
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Comparación — Val Accuracy por época")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def main():
    args = parse_args()

    if args.models:
        model_names = args.models
    else:
        model_names = sorted(p.stem for p in _METRICS_DIR.glob("*.csv"))

    if not model_names:
        raise SystemExit(f"No se encontraron modelos en {_METRICS_DIR}")

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

    _plot_val_accuracy_global(all_results, args.out_dir / "val_accuracy_global.png")
    _plot_loss_curves(curves, args.out_dir / "loss_curves.png")
    _plot_val_accuracy_curves(curves, args.out_dir / "val_accuracy_curves.png")

    print(f"\nComparación guardada en {args.out_dir}")


if __name__ == "__main__":
    main()
