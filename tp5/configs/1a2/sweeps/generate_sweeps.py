"""Genera los configs de ablación OAT (one-at-a-time) para 1a2.

Diseño: se parte de un **baseline débil** (el punto de partida ingenuo de la
progresión 1a2, ~24 px de error) y por cada hiperparámetro se genera una serie de
configs que varían **solo esa dimensión** dejando el resto fijo en el baseline.
Cada serie incluye el valor del baseline como punto de anclaje, de modo que el
gráfico de cada dimensión muestra cuánto MEJORA el error de píxeles al mover esa
variable hacia una mejor elección.

    uv run python configs/1a2/sweeps/generate_sweeps.py

Escribe:
  - configs/1a2/sweeps/<dim>/<name>.json      (un JSON por punto de cada sweep)
  - configs/1a2/sweeps/manifest.json          (qué se varió en cada uno; lo usa plot_sweeps.py)
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # .../tp5

# --------------------------------------------------------------------------- #
# Baseline débil: el punto de partida ingenuo de 1a2 (config 01, ~24 px).
# Todas las ablaciones parten de acá y mueven UNA sola variable.
# --------------------------------------------------------------------------- #
BASE = {
    "encoder_layers": [35, 20, 2],
    "activation": "relu",
    # Cuello FIJO en 'linear' (default teórico de un AE: código sin acotar, link con
    # PCA, consistente con la media de un VAE). Se mantiene constante en todos los
    # sweeps salvo el de `latent_activation`, así cada ablación cambia una sola cosa.
    "latent_activation": "linear",
    "output_activation": "sigmoid",
    "init": "normal",
    "optimizer": "adam",
    "loss": "mse",
    "epochs": 15000,
    "lr": 1e-3,
    "restarts": 12,
    "seed": 42,
    "log_every": 300,
    "stop_at": None,  # null -> corre los 12 restarts (sin corte temprano) para medir robustez
}

# --------------------------------------------------------------------------- #
# Definición de los sweeps OAT. `param` indica qué clave del baseline se varía.
# `fmt` produce un sufijo de nombre/archivo legible y único por valor.
# --------------------------------------------------------------------------- #
SWEEPS = [
    {
        "dim": "epochs", "param": "epochs", "kind": "numeric",
        "values": [10000, 15000, 20000, 25000, 30000],
        "fmt": lambda v: f"{v:05d}",
    },
    {
        "dim": "lr", "param": "lr", "kind": "numeric",
        "values": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "fmt": lambda v: f"{v:.0e}",
    },
    {
        "dim": "optimizer", "param": "optimizer", "kind": "categorical",
        "values": ["adam", "lbfgs"],
        "fmt": lambda v: v,
    },
    {
        "dim": "loss", "param": "loss", "kind": "categorical",
        "values": ["mse", "bce"],
        "fmt": lambda v: v,
    },
    {
        "dim": "activation", "param": "activation", "kind": "categorical",
        "values": ["relu", "tanh", "sigmoid"],
        "fmt": lambda v: v,
    },
    {
        "dim": "output_activation", "param": "output_activation", "kind": "categorical",
        "values": ["sigmoid", "tanh", "linear"],
        "fmt": lambda v: v,
    },
    {
        # Activación del cuello (latente). El baseline es 'linear' (default teórico);
        # este sweep muestra que desviarse a relu/sigmoid/tanh degrada el latente.
        "dim": "latent_activation", "param": "latent_activation", "kind": "categorical",
        "values": ["linear", "tanh", "sigmoid", "relu"],
        "fmt": lambda v: v,
    },
    {
        "dim": "init", "param": "init", "kind": "categorical",
        "values": ["normal", "uniform", "xavier_uniform", "xavier_normal",
                   "he_uniform", "he_normal"],
        "fmt": lambda v: v,
    },
    {
        "dim": "architecture", "param": "encoder_layers", "kind": "categorical",
        "values": [
            [35, 2],
            [35, 20, 2],
            [35, 16, 8, 2],
            [35, 25, 15, 8, 2],
            [35, 30, 20, 12, 6, 2],
            [35, 30, 2],
        ],
        "fmt": lambda v: "-".join(str(n) for n in v),
    },
]


def build_config(name: str, params: dict, metrics_csv: str, plots_dir: str) -> dict:
    """Arma el dict del config.json a partir del baseline + overrides en `params`."""
    p = {**BASE, **params}
    architecture = {
        "encoder_layers": p["encoder_layers"],
        "activation": p["activation"],
        "output_activation": p["output_activation"],
        "init": p["init"],
    }
    # Solo emitimos latent_activation cuando el sweep la varía; el resto de los
    # configs quedan idénticos (sin la clave -> el cuello sigue a `activation`).
    if "latent_activation" in p:
        architecture["latent_activation"] = p["latent_activation"]
    return {
        "name": name,
        "data": {"font_path": "font/font.h", "subset": None},
        "architecture": architecture,
        "training": {
            "optimizer": p["optimizer"],
            "loss": p["loss"],
            "epochs": p["epochs"],
            "lr": p["lr"],
            "restarts": p["restarts"],
            "seed": p["seed"],
            "log_every": p["log_every"],
            "stop_at": p["stop_at"],
        },
        "denoising": {"enabled": False},
        "output": {"metrics_csv": metrics_csv, "plots_dir": plots_dir},
    }


def main() -> None:
    manifest = {"baseline": BASE, "sweeps": []}
    total = 0

    for sw in SWEEPS:
        dim = sw["dim"]
        dim_dir = HERE / dim
        dim_dir.mkdir(parents=True, exist_ok=True)
        # Idempotencia: borra configs de corridas previas (p. ej. valores que sacaste
        # del sweep) para no dejar JSON huérfanos que run_all.sh correría igual.
        for old in dim_dir.glob("1a2_sweep_*.json"):
            old.unlink()
        plots_dir = f"out/1a2/sweeps/{dim}/plots"
        # Valor del baseline para esta dimensión (override explícito si no está en BASE).
        base_val = sw.get("baseline", BASE.get(sw["param"]))

        entries = []
        for value in sw["values"]:
            suffix = sw["fmt"](value)
            name = f"1a2_sweep_{dim}_{suffix}"
            metrics_csv = f"out/1a2/sweeps/{dim}/{name}/metrics.csv"
            cfg = build_config(name, {sw["param"]: value}, metrics_csv, plots_dir)

            cfg_path = dim_dir / f"{name}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
            total += 1

            restarts_csv = metrics_csv.replace(".csv", "_restarts.csv")
            entries.append({
                "name": name,
                "value": value,
                "label": suffix,
                "is_baseline": base_val == value,
                "config": str(cfg_path.relative_to(REPO)),
                "metrics_restarts": restarts_csv,
            })

        manifest["sweeps"].append({
            "dim": dim,
            "param": sw["param"],
            "kind": sw["kind"],
            "baseline_value": base_val,
            "configs": entries,
        })
        print(f"  {dim:18s} -> {len(entries)} configs")

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nTotal: {total} configs en {len(SWEEPS)} dimensiones.")
    print(f"Manifest: {HERE / 'manifest.json'}")


if __name__ == "__main__":
    main()
