"""Evaluación de un modelo entrenado sobre un dataset de dígitos.

Ejemplos:
    uv run python -m exercises.ej2_digits.evaluate
    uv run python -m exercises.ej2_digits.evaluate --model baseline
    uv run python -m exercises.ej2_digits.evaluate --model adam --dataset data/ej2_digits/digits_test.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from common.datasets import load_digit_frame, to_one_hot
from common.mlp import MLP

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "outputs" / "ej2_digits" / "models"
_DEFAULT_MODEL = "baseline"
_DEFAULT_DATASET = _ROOT / "data" / "ej2_digits" / "digits_test.csv"
_N_CLASSES = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Evalúa un modelo MLP sobre un dataset de dígitos.")
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                        help="Nombre del modelo (sin .npz). Se busca en outputs/ej2_digits/models/.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET,
                        help="Path al archivo .csv del dataset.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directorio de salida. Default: outputs/ej2_digits/evaluation/<model>/")
    return parser.parse_args()


def _load_dataset(dataset_path):
    df = load_digit_frame(dataset_path)
    X = np.stack(df["image"].values)
    y = df["label"].values.astype(int)
    return X, y


def _confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _get_val_accuracy(model_name, configs_dir):
    metrics_dir = _ROOT / "outputs" / "ej2_digits" / "metrics"
    csv_path = metrics_dir / f"{model_name}.csv"
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    if "acc_val" not in df.columns or "loss_val" not in df.columns:
        return None, None
    config_path = Path(configs_dir) / f"{model_name}.json"
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


def _plot_accuracy_per_class(per_class_acc, model_name, out_path):
    classes = list(range(_N_CLASSES))
    matrix = np.array([[per_class_acc.get(str(c), 0.0) for c in classes]])

    fig, ax = plt.subplots(figsize=(12, 1.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xticks(range(_N_CLASSES))
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_yticks([0])
    ax.set_yticklabels([model_name])
    ax.set_xlabel("Dígito")
    ax.set_title("Accuracy por clase")

    for j, c in enumerate(classes):
        val = matrix[0, j]
        ax.text(j, 0, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color="black" if 0.3 < val < 0.8 else "white")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def _plot_val_vs_test_accuracy(val_acc, test_acc, model_name, out_path):
    labels = ["Val Accuracy", "Test Accuracy"]
    values = [val_acc, test_acc]
    colors = ["steelblue", "tomato"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(min(values) - 0.05, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Val vs Test Accuracy — {model_name}")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def _plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(_N_CLASSES))
    ax.set_yticks(range(_N_CLASSES))
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión")

    thresh = cm.max() / 2
    for i in range(_N_CLASSES):
        for j in range(_N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  guardado: {out_path}")
    plt.close(fig)


def run(model_name, dataset_path=None, out_dir=None):
    """Evalúa un modelo y guarda métricas + gráficos. Retorna el dict de resultados."""
    if dataset_path is None:
        dataset_path = _DEFAULT_DATASET

    model_path = _MODELS_DIR / f"{model_name}.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

    if out_dir is None:
        out_dir = _ROOT / "outputs" / "ej2_digits" / "evaluation" / model_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelo:  {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {out_dir}")
    print("-" * 45)

    mlp = MLP.load(model_path)
    X, y = _load_dataset(dataset_path)

    encoding = "signed" if mlp.loss == "mse" else "zero_one"
    Y = to_one_hot(y, _N_CLASSES, encoding=encoding)

    metrics = mlp.evaluate(X, Y)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Loss:     {metrics['loss']:.4f}")

    pred_cls = np.argmax(mlp.forward(X), axis=1)

    print("\nAccuracy por clase:")
    per_class = {}
    for c in range(_N_CLASSES):
        mask = y == c
        if mask.sum() > 0:
            acc = float(np.mean(pred_cls[mask] == c))
            per_class[str(c)] = acc
            print(f"  Dígito {c}: {acc:.4f}  ({mask.sum()} muestras)")

    results = {
        "model": model_path.name,
        "dataset": Path(dataset_path).name,
        "accuracy": float(metrics["accuracy"]),
        "loss": float(metrics["loss"]),
        "per_class_accuracy": per_class,
    }
    json_path = out_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  guardado: {json_path}")

    _plot_accuracy_per_class(per_class, model_name, out_dir / "accuracy_per_class.png")

    val_acc, _ = _get_val_accuracy(model_name, _ROOT / "configs" / "ej2_digits")
    if val_acc is not None:
        _plot_val_vs_test_accuracy(val_acc, float(metrics["accuracy"]), model_name,
                                   out_dir / "val_vs_test_accuracy.png")
    else:
        print("  [WARN] No se encontró val accuracy en métricas — omitiendo val vs test plot.")

    cm = _confusion_matrix(y, pred_cls, _N_CLASSES)
    _plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")

    return results


def main():
    args = parse_args()
    run(args.model, dataset_path=args.dataset, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
