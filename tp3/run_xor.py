import json
import sys

import numpy as np

from src.datasets import xor_dataset
from src.mlp import MLP
from src.optimizers import SGD, Momentum, Adam

_OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}


def load_config(path="configs/xor.json"):
    with open(path) as f:
        return json.load(f)


def make_optimizer(cfg):
    cls = _OPTIMIZERS[cfg["optimizer"]]
    return cls(lr=cfg["lr"])


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/xor.json"
    cfg = load_config(config_path)

    X, y_flat = xor_dataset()
    y = y_flat.reshape(-1, 1)

    mlp = MLP(
        layer_sizes=cfg["architecture"],
        activation=cfg["activation"],
        output_activation=cfg["output_activation"],
        loss=cfg["loss"],
        weight_init=cfg.get("weight_init", "xavier"),
        seed=cfg.get("seed", 42),
    )

    optimizer = make_optimizer(cfg)

    mlp.fit(
        X, y,
        epochs=cfg["epochs"],
        batch_size=cfg.get("batch_size", 4),
        optimizer=optimizer,
        patience=cfg.get("patience"),
        verbose=True,
    )

    print("\n--- XOR Results ---")
    print(f"{'Input':>12}  {'Target':>7}  {'Output':>8}  {'Sign':>5}  OK")
    for xi, yi in zip(X, y_flat):
        out = float(mlp.forward(xi.reshape(1, -1)).ravel()[0])
        sign = int(np.sign(out))
        ok = "✓" if sign == int(yi) else "✗"
        print(f"{str(xi):>12}  {int(yi):>+7}  {out:>8.4f}  {sign:>+5}  {ok}")

    if cfg.get("plot", False):
        _plot(mlp, X, y_flat, mlp.history)


def _plot(mlp, X, y_flat, history):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Decision boundary
    ax = axes[0]
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.7)
    colors = ["red" if yi < 0 else "blue" for yi in y_flat]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=120, zorder=3, edgecolors="k")
    ax.set_title("Decision boundary")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    # Loss history
    axes[1].plot([e["loss_train"] for e in history])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE loss")
    axes[1].set_title("Training loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
