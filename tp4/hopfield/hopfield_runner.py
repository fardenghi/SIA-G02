"""Ejercicio 2.1 — Hopfield: parte (a) recuperación con ruido, parte (b) espúreo.

Lee el config (configs/hopfield.json), almacena el subconjunto indicado de letras,
y para cada letra:
  - genera una versión ruidosa
  - corre Hopfield mostrando paso a paso (consola + figura)
Luego corre el caso ruido alto para mostrar un estado espúreo.

Por defecto las 4 letras se eligen como el subconjunto más ortogonal hallado por el
módulo orthogonality (si el config no las especifica explícitamente).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hopfield.alphabet import ALPHABET, letter_vector, render_ascii
from hopfield.hopfield import HopfieldNetwork, add_noise
from hopfield.orthogonality import pairwise_dot_matrix, rank_combinations


def pick_letters(cfg: dict) -> list[str]:
    if cfg.get("letters"):
        return [c.upper() for c in cfg["letters"]]
    k = int(cfg.get("k", 4))
    dot = pairwise_dot_matrix()
    df = rank_combinations(k, dot)
    return list(df.iloc[0]["combo"])


def plot_steps(
    history: list[np.ndarray],
    energies: list[float],
    title: str,
    output: str,
    target: np.ndarray | None = None,
) -> None:
    n = len(history)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes_flat = np.atleast_2d(axes).ravel() if n > 1 else [axes]
    for ax in axes_flat:
        ax.axis("off")

    for i, (state, e) in enumerate(zip(history, energies)):
        ax = axes_flat[i]
        ax.axis("on")
        img = state.reshape(5, 5)
        ax.imshow(np.where(img == 1, 1.0, 0.0), cmap="Greys",
                  vmin=0, vmax=1, interpolation="nearest")
        match = ""
        if target is not None and np.array_equal(state, target):
            match = " ✓"
        ax.set_title(f"t={i}  E={e:.2f}{match}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def print_pattern(state: np.ndarray, header: str) -> None:
    print(f"\n{header}")
    print(render_ascii(state.reshape(5, 5)))


def run_recall(
    net: HopfieldNetwork,
    target_letter: str,
    noise: float,
    mode: str,
    max_steps: int,
    rng: np.random.Generator,
    output_dir: str,
    tag: str,
) -> dict:
    target = letter_vector(target_letter)
    noisy = add_noise(target, noise, rng)
    final, history, energies, converged = net.recall(
        noisy, mode=mode, max_steps=max_steps, rng=rng
    )

    print(f"\n\n===== Letra '{target_letter}' — ruido {noise:.0%} ({tag}) =====")
    print_pattern(target, "Patrón objetivo:")
    print_pattern(noisy, "Patrón ruidoso (entrada):")

    for t, st in enumerate(history):
        print_pattern(st, f"Paso t={t}  energía={energies[t]:.3f}")

    print(f"\nConvergió: {converged}")
    matched = net.is_stored(final)
    if np.array_equal(final, target):
        print(f"-> Recuperó la letra '{target_letter}' correctamente")
        verdict = "ok"
    elif matched != -1:
        print(f"-> Convergió al patrón almacenado #{matched} (otra letra)")
        verdict = "wrong_stored"
    else:
        print("-> Estado ESPÚREO (atractor fijo distinto a las letras almacenadas)")
        verdict = "spurious"

    plot_steps(
        history, energies,
        title=f"Letra '{target_letter}' (ruido {noise:.0%}, {mode}) — {verdict}",
        output=os.path.join(output_dir, f"recall_{tag}_{target_letter}.png"),
        target=target,
    )
    return {
        "letter": target_letter,
        "noise": noise,
        "steps": len(history) - 1,
        "converged": converged,
        "verdict": verdict,
        "final_energy": energies[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Hopfield — Ejercicio Patrones")
    parser.add_argument("--config", default="configs/hopfield.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(cfg.get("seed", 42))

    letters = pick_letters(cfg)
    print(f"Letras almacenadas: {letters}")

    patterns = np.stack([letter_vector(c) for c in letters])
    net = HopfieldNetwork(n_units=25)
    net.store(patterns)

    # Diagnóstico de la elección
    A = patterns.astype(np.int32)
    dot_sub = A @ A.T
    abs_off = np.abs(dot_sub.copy())
    np.fill_diagonal(abs_off, 0)
    print(f"max |<xi,xj>| del subconjunto: {abs_off.max()}")
    print(f"mean |<xi,xj>| del subconjunto: {abs_off[np.triu_indices_from(abs_off, k=1)].mean():.2f}")

    # ---- Parte (a) ----
    print("\n############ PARTE (a) — recuperación con ruido ############")
    results = []
    for letter in letters:
        r = run_recall(
            net, letter,
            noise=cfg.get("noise", 0.15),
            mode=cfg.get("mode", "sync"),
            max_steps=cfg.get("max_steps", 50),
            rng=rng,
            output_dir=out,
            tag="a",
        )
        results.append(r)

    # ---- Parte (b) ----
    print("\n\n############ PARTE (b) — patrón muy ruidoso / estado espúreo ############")
    high_noise = cfg.get("high_noise", 0.4)
    n_attempts = cfg.get("spurious_attempts", 10)
    spurious_results = []
    found_spurious = False
    for attempt in range(n_attempts):
        letter = letters[attempt % len(letters)]
        r = run_recall(
            net, letter,
            noise=high_noise,
            mode=cfg.get("mode", "sync"),
            max_steps=cfg.get("max_steps", 50),
            rng=rng,
            output_dir=out,
            tag=f"b_try{attempt + 1}",
        )
        spurious_results.append(r)
        if r["verdict"] == "spurious":
            print(f"\n>>> Estado espúreo encontrado en el intento #{attempt + 1} "
                  f"(letra base '{letter}', ruido {high_noise:.0%}).")
            found_spurious = True
            break
    if not found_spurious:
        print(f"\nNo se encontró estado espúreo en {n_attempts} intentos. "
              f"Probá subir 'high_noise' o cambiar 'letters' a un set menos ortogonal.")

    # Resumen
    print("\n############ RESUMEN ############")
    print(f"{'Letra':<6} {'ruido':>6} {'pasos':>6} {'conv':>6} {'verdict':<14} {'E_final':>10}")
    print("-" * 56)
    for r in results + spurious_results:
        print(f"{r['letter']:<6} {r['noise']:>6.2f} {r['steps']:>6} "
              f"{str(r['converged']):>6} {r['verdict']:<14} {r['final_energy']:>10.3f}")

    print(f"\nFiguras y resultados en {out}/")


if __name__ == "__main__":
    main()
