import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from kohonen.som import SOM


def load_data(path: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    countries = df["Country"].tolist()
    X = df.drop(columns=["Country"]).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    return countries, X_scaled


def run(cfg: dict, X: np.ndarray) -> tuple[SOM, np.ndarray]:
    som = SOM(
        grid_rows=cfg["grid_rows"],
        grid_cols=cfg["grid_cols"],
        input_dim=X.shape[1],
        lr=cfg["lr"],
        lr_decay=cfg["lr_decay"],
        radius=cfg["radius"],
        radius_decay=cfg["radius_decay"],
        neighborhood_fn=cfg["neighborhood_fn"],
        epochs=cfg["epochs"],
        seed=cfg["seed"],
    )
    som.train(X)
    coords = som.predict(X)
    return som, coords


def build_assignments(countries: list[str], coords: np.ndarray) -> dict[tuple, list[str]]:
    assignments: dict[tuple, list[str]] = {}
    for country, (r, c) in zip(countries, coords):
        key = (int(r), int(c))
        assignments.setdefault(key, []).append(country)
    return assignments


def print_assignments(assignments: dict[tuple, list[str]], grid_rows: int, grid_cols: int) -> None:
    print("\nAsignación de países por neurona:")
    print(f"{'Neurona':<12} {'Países'}")
    print("-" * 50)
    for r in range(grid_rows):
        for c in range(grid_cols):
            key = (r, c)
            countries = assignments.get(key, [])
            if countries:
                print(f"  ({r},{c})      {', '.join(countries)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Red de Kohonen — Ejercicio Europa")
    parser.add_argument("--config", default="configs/kohonen_europe.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    countries, X = load_data(cfg["data"])
    som, coords = run(cfg, X)
    assignments = build_assignments(countries, coords)
    print_assignments(assignments, cfg["grid_rows"], cfg["grid_cols"])

    print("Conteo de países por neurona:")
    for r in range(cfg["grid_rows"]):
        row_str = ""
        for c in range(cfg["grid_cols"]):
            n = len(assignments.get((r, c), []))
            row_str += f"{n:3}"
        print(row_str)

    print(f"\nModelo entrenado. Resultados en {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
