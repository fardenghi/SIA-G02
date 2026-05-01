import json
import sys
from pathlib import Path

import numpy as np

from common.datasets import load_digits, load_digits_test, to_one_hot
from common.metrics import MetricsTracker
from common.mlp import MLP
from common.optimizers import (
    AdaptiveLR, Adam, ExponentialDecay, Momentum, RMSProp, SGD, StepDecay,
)

_OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "rmsprop": RMSProp, "adam": Adam}
_SCHEDULERS = {
    "step_decay": StepDecay,
    "exponential_decay": ExponentialDecay,
    "adaptive": AdaptiveLR,
}


def load_config(path):
    with open(path) as f:
        return json.load(f)


def make_optimizer(cfg):
    cls = _OPTIMIZERS[cfg["optimizer"]]
    return cls(lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))


def make_scheduler(cfg):
    sched_cfg = cfg.get("lr_scheduler")
    if not sched_cfg:
        return None
    cls = _SCHEDULERS[sched_cfg["type"]]
    return cls(**{k: v for k, v in sched_cfg.items() if k != "type"})


def encoding_for(loss):
    return "signed" if loss == "mse" else "zero_one"


def run(config_path):
    cfg = load_config(config_path)

    X_all, y_all = load_digits()
    N = X_all.shape[0]
    val_split = cfg.get("val_split", 0.2)
    rng = np.random.default_rng(cfg.get("seed", 42))
    idx = rng.permutation(N)
    n_val = int(N * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train_raw = X_all[train_idx], y_all[train_idx]
    X_val, y_val_raw = X_all[val_idx], y_all[val_idx]
    X_test, y_test_raw = load_digits_test()

    enc = encoding_for(cfg["loss"])
    n_classes = cfg["architecture"][-1]
    Y_train = to_one_hot(y_train_raw, n_classes, encoding=enc)
    Y_val = to_one_hot(y_val_raw, n_classes, encoding=enc)
    Y_test = to_one_hot(y_test_raw, n_classes, encoding=enc)

    tracker = MetricsTracker(
        run_id=config_path,
        config_meta={
            "arch": str(cfg["architecture"]),
            "optimizer": cfg["optimizer"],
            "lr": cfg["lr"],
            "batch_size": cfg.get("batch_size", 32),
        },
    )

    mlp = MLP(
        layer_sizes=cfg["architecture"],
        activation=cfg["activation"],
        output_activation=cfg["output_activation"],
        loss=cfg["loss"],
        weight_init=cfg.get("weight_init", "xavier"),
        seed=cfg.get("seed", 42),
    )

    mlp.fit(
        X_train, Y_train,
        X_val=X_val, y_val=Y_val,
        epochs=cfg["epochs"],
        batch_size=cfg.get("batch_size", 32),
        optimizer=make_optimizer(cfg),
        patience=cfg.get("patience"),
        min_delta=cfg.get("min_delta", 0.0),
        verbose=True,
        tracker=tracker,
        data_augmentation=cfg.get("data_augmentation", False),
        lr_scheduler=make_scheduler(cfg),
    )

    test_m = mlp.evaluate(X_test, Y_test)
    print(f"\nTest accuracy (all classes): {test_m['accuracy']:.4f}  "
          f"Test loss (all classes): {test_m['loss']:.4f}")

    # Metrics on seen classes only
    present_classes = np.unique(y_train_raw)
    mask_seen = np.isin(y_test_raw, present_classes)
    if mask_seen.sum() > 0:
        seen_m = mlp.evaluate(X_test[mask_seen], Y_test[mask_seen])
        print(f"Test accuracy (only seen): {seen_m['accuracy']:.4f}  "
              f"Test loss (only seen): {seen_m['loss']:.4f}")

    # Per-class accuracy
    pred_cls = np.argmax(mlp.forward(X_test), axis=1)
    print("\nPer-class accuracy:")
    for c in range(n_classes):
        mask = y_test_raw == c
        if mask.sum() > 0:
            acc = np.mean(pred_cls[mask] == c)
            print(f"  Class {c}: {acc:.4f}  ({mask.sum()} samples)")

    if cfg.get("save_model"):
        path = cfg["save_model"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        mlp.save(path)
        print(f"\nModel saved to {path}")

    if cfg.get("export_metrics"):
        path = cfg["export_metrics"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tracker.export_csv(path)
        print(f"Metrics saved to {path}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ej2_digits/baseline.json"
    run(config_path)


if __name__ == "__main__":
    main()
