import json
import sys
from pathlib import Path

import numpy as np

from common.datasets import load_digits, load_digits_test, load_more_digits, to_one_hot
from common.metrics import MetricsTracker
from common.mlp import MLP
from common.optimizers import Adam, Momentum, SGD

_OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}


def load_config(path):
    with open(path) as f:
        return json.load(f)


def make_optimizer(cfg):
    cls = _OPTIMIZERS[cfg["optimizer"]]
    return cls(lr=cfg["lr"])


def encoding_for(loss):
    return "signed" if loss == "mse" else "zero_one"


def main():
    config_path = (sys.argv[1] if len(sys.argv) > 1
                   else "configs/ej3_more_digits/softmax.json")
    cfg = load_config(config_path)

    X_more, y_more = load_more_digits()
    if cfg.get("combine_datasets", False):
        X_dig, y_dig = load_digits()
        X_all = np.concatenate([X_more, X_dig], axis=0)
        y_all = np.concatenate([y_more, y_dig], axis=0)
        print(f"Combined dataset: {X_all.shape[0]} muestras")
    else:
        X_all, y_all = X_more, y_more
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
            "batch_size": cfg.get("batch_size", 64),
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
        batch_size=cfg.get("batch_size", 64),
        optimizer=make_optimizer(cfg),
        patience=cfg.get("patience"),
        verbose=True,
        tracker=tracker,
        data_augmentation=cfg.get("data_augmentation", False)
    )

    test_m = mlp.evaluate(X_test, Y_test)
    accuracy = test_m["accuracy"]
    print(f"\nTest accuracy: {accuracy:.4f}  Test loss: {test_m['loss']:.4f}")

    pred_cls = np.argmax(mlp.forward(X_test), axis=1)
    print("\nPer-class accuracy:")
    for c in range(n_classes):
        mask = y_test_raw == c
        if mask.sum() > 0:
            acc = np.mean(pred_cls[mask] == c)
            print(f"  Class {c}: {acc:.4f}  ({mask.sum()} samples)")

    goal = cfg.get("goal_accuracy", 0.98)
    if accuracy >= goal:
        print(f"\n✓ goal reached ({accuracy:.4f} >= {goal})")
    else:
        print(f"\n✗ goal not reached ({accuracy:.4f} < {goal})")

    if cfg.get("save_model"):
        path = cfg["save_model"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        mlp.save(path)
        print(f"Model saved to {path}")

    if cfg.get("export_metrics"):
        path = cfg["export_metrics"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tracker.export_csv(path)
        print(f"Metrics saved to {path}")


if __name__ == "__main__":
    main()
