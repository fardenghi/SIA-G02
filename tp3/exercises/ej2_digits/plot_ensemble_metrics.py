"""Plot training curves of an ensemble (mean across constituent models).

Same layout as plot_metrics.py: two subplots (loss vs epoch, acc vs epoch),
each with train and val curves. The thick lines show the per-epoch mean
across all input CSVs; thin lines show each individual model.

Models can have different lengths (early stopping); shorter runs are padded
with NaN and ignored in the mean.

Usage:
    uv run python exercises/ej2_digits/plot_ensemble_metrics.py \\
        --csvs path1.csv path2.csv ... \\
        --out output.png \\
        [--label "ensemble name"]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _stack_metric(dfs, col):
    """Stack a column from multiple dfs into a (N, max_epochs) array, NaN-padded."""
    max_len = max(len(df) for df in dfs)
    out = np.full((len(dfs), max_len), np.nan)
    for i, df in enumerate(dfs):
        if col in df.columns:
            out[i, :len(df)] = df[col].to_numpy()
    return out


def plot_ensemble(csv_paths, output_path, label=None):
    dfs = [pd.read_csv(p) for p in csv_paths]
    if not dfs:
        raise ValueError("Need at least 1 CSV.")

    epochs = np.arange(1, max(len(df) for df in dfs) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    title = label or f"ensemble ({len(dfs)} models)"

    # Loss subplot
    train_loss = _stack_metric(dfs, "loss_train")
    val_loss = _stack_metric(dfs, "loss_val")
    for i in range(len(dfs)):
        ax1.plot(epochs, train_loss[i], color="steelblue", alpha=0.25, lw=0.8)
        ax1.plot(epochs, val_loss[i], color="tomato", alpha=0.25, lw=0.8)
    ax1.plot(epochs, np.nanmean(train_loss, axis=0),
             label="Train Loss (mean)", color="steelblue", lw=2)
    ax1.plot(epochs, np.nanmean(val_loss, axis=0),
             label="Val Loss (mean)", color="tomato", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss vs Epochs ({title})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy subplot
    train_acc = _stack_metric(dfs, "acc_train")
    val_acc = _stack_metric(dfs, "acc_val")
    for i in range(len(dfs)):
        ax2.plot(epochs, train_acc[i], color="steelblue", alpha=0.25, lw=0.8)
        ax2.plot(epochs, val_acc[i], color="tomato", alpha=0.25, lw=0.8)
    ax2.plot(epochs, np.nanmean(train_acc, axis=0),
             label="Train Acc (mean)", color="steelblue", lw=2)
    ax2.plot(epochs, np.nanmean(val_acc, axis=0),
             label="Val Acc (mean)", color="tomato", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy vs Epochs ({title})")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="Paths to constituent metric CSVs")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--label", default=None, help="Label for the title")
    args = parser.parse_args()
    plot_ensemble(args.csvs, args.out, args.label)
