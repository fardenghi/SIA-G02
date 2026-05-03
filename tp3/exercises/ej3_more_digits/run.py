import json
import sys
from pathlib import Path

import numpy as np

from common.datasets import load_digits, load_digits_test, load_more_digits, to_one_hot
from common.ensemble import Ensemble
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


def _suffix_path(path, suffix):
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def train_one(cfg, seed, X_train, Y_train, X_val, Y_val, config_path):
    tracker = MetricsTracker(
        run_id=f"{config_path}#seed={seed}",
        config_meta={
            "arch": str(cfg["architecture"]),
            "optimizer": cfg["optimizer"],
            "lr": cfg["lr"],
            "batch_size": cfg.get("batch_size", 64),
            "seed": seed,
        },
    )

    mlp = MLP(
        layer_sizes=cfg["architecture"],
        activation=cfg["activation"],
        output_activation=cfg["output_activation"],
        loss=cfg["loss"],
        weight_init=cfg.get("weight_init", "xavier"),
        seed=seed,
    )

    aug_scale_range = cfg.get("aug_scale_range")
    if aug_scale_range is not None:
        aug_scale_range = tuple(aug_scale_range)

    mlp.fit(
        X_train, Y_train,
        X_val=X_val, y_val=Y_val,
        epochs=cfg["epochs"],
        batch_size=cfg.get("batch_size", 64),
        optimizer=make_optimizer(cfg),
        patience=cfg.get("patience"),
        min_delta=cfg.get("min_delta", 0.0),
        verbose=True,
        tracker=tracker,
        data_augmentation=cfg.get("data_augmentation", False),
        aug_rotation_deg=cfg.get("aug_rotation_deg", 0.0),
        aug_scale_range=aug_scale_range,
        lr_scheduler=make_scheduler(cfg),
    )
    return mlp, tracker


def report(mlp_or_ens, X_test, y_test_raw, Y_test, n_classes, label):
    test_m = mlp_or_ens.evaluate(X_test, Y_test)
    print(f"\n{label}: test_acc={test_m['accuracy']:.4f}  test_loss={test_m['loss']:.4f}")

    pred_cls = np.argmax(mlp_or_ens.forward(X_test), axis=1)
    print("Per-class accuracy:")
    for c in range(n_classes):
        mask = y_test_raw == c
        if mask.sum() > 0:
            acc = np.mean(pred_cls[mask] == c)
            print(f"  Class {c}: {acc:.4f}  ({mask.sum()} samples)")
    return test_m["accuracy"]


def run(config_path):
    cfg = load_config(config_path)

    X_more, y_more = load_more_digits()
    if cfg.get("combine_datasets", False):
        X_dig, y_dig = load_digits()
        X_all = np.concatenate([X_more, X_dig], axis=0)
        y_all = np.concatenate([y_more, y_dig], axis=0)
        print(f"Combined dataset: {X_all.shape[0]} muestras")
    else:
        X_all, y_all = X_more, y_more

    seeds = cfg.get("seeds")
    if seeds is None:
        seeds = [cfg.get("seed", 42)]

    # Same train/val split for all seeds, derived from the first seed
    split_seed = seeds[0]
    N = X_all.shape[0]
    val_split = cfg.get("val_split", 0.2)
    rng = np.random.default_rng(split_seed)
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

    models = []
    for s in seeds:
        print(f"\n========== Training seed={s} ==========")
        mlp, tracker = train_one(cfg, s, X_train, Y_train, X_val, Y_val, config_path)
        models.append(mlp)

        if cfg.get("save_model"):
            base = cfg["save_model"]
            path = _suffix_path(base, f"seed{s}") if len(seeds) > 1 else base
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            mlp.save(path)
            print(f"Model saved to {path}")

        if cfg.get("export_metrics"):
            base = cfg["export_metrics"]
            path = _suffix_path(base, f"seed{s}") if len(seeds) > 1 else base
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            tracker.export_csv(path)
            print(f"Metrics saved to {path}")

    # Per-seed test report
    accs = []
    for s, mlp in zip(seeds, models):
        accs.append(report(mlp, X_test, y_test_raw, Y_test, n_classes, f"Seed {s}"))

    if len(models) > 1:
        ens = Ensemble(models)
        acc_ens = report(ens, X_test, y_test_raw, Y_test, n_classes, "ENSEMBLE")
        accuracy = acc_ens
    else:
        accuracy = accs[0]

    goal = cfg.get("goal_accuracy", 0.98)
    if accuracy >= goal:
        print(f"\n✓ goal reached ({accuracy:.4f} >= {goal})")
    else:
        print(f"\n✗ goal not reached ({accuracy:.4f} < {goal})")


def main():
    config_path = (sys.argv[1] if len(sys.argv) > 1
                   else "configs/ej3_more_digits/historical/best_l2(+pat_weight).json")
    run(config_path)


if __name__ == "__main__":
    main()
