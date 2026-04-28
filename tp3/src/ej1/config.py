import json
from pathlib import Path

DEFAULT_CONFIG = {
    "learning_rate": 0.01,
    "max_epochs": 100,
    "activation": "sigmoid",   # "linear" | "sigmoid" | "tanh"
    "val_ratio": 0.2,
    "seed": 42,
}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    return {**DEFAULT_CONFIG, **cfg}


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
