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

from hopfield.alphabet import (
    ALPHABET, GRID, LETTERS, letter_vector, min_scale_factor, render_ascii,
    scaled_letter_vector,
)
from hopfield.hopfield import HopfieldNetwork, add_noise
from hopfield.orthogonality import pairwise_dot_matrix, rank_combinations
from hopfield.plots import (
    plot_crosstalk, plot_crosstalk_per_neuron, plot_energy_evolution, plot_recovery_rate_vs_noise,
)


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

    fig = plt.figure(figsize=(cols * 1.8, rows * 2.0 + 2.5))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[2.0] * rows + [2.5], hspace=0.45)

    axes_flat = [fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(cols)]
    for ax in axes_flat:
        ax.axis("off")

    for i, (state, e) in enumerate(zip(history, energies)):
        ax = axes_flat[i]
        ax.axis("on")
        grid_size = int(np.sqrt(len(state)))
        img = state.reshape(grid_size, grid_size)
        ax.imshow(np.where(img == 1, 1.0, 0.0), cmap="Greys",
                  vmin=0, vmax=1, interpolation="nearest")
        match = ""
        if target is not None and np.array_equal(state, target):
            match = " ✓"
        ax.set_title(f"t={i}  E={e:.2f}{match}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_e = fig.add_subplot(gs[rows, :])
    ax_e.plot(range(len(energies)), energies, marker="o", linewidth=2, color="#1E3A5F")
    ax_e.set_xlabel("Iteración")
    ax_e.set_ylabel("Energía H")
    ax_e.set_title("Evolución de energía")
    ax_e.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_pattern(state: np.ndarray, header: str) -> None:
    print(f"\n{header}")
    grid_size = int(np.sqrt(len(state)))
    print(render_ascii(state.reshape(grid_size, grid_size)) if grid_size == 5 else f"(patrón {grid_size}×{grid_size})")


def run_recall(
    net: HopfieldNetwork,
    target_letter: str,
    noise: float,
    mode: str,
    max_steps: int,
    rng: np.random.Generator,
    output_dir: str,
    tag: str,
    scale: int = 1,
    noisy_input: np.ndarray | None = None,
) -> dict:
    target = scaled_letter_vector(target_letter, scale)
    noisy = noisy_input if noisy_input is not None else add_noise(target, noise, rng)
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


def run_alphabet_mode(cfg: dict) -> None:
    """Modo `--alphabet`: almacena las 26 letras con escalado adaptativo."""
    letters = LETTERS[:]
    k = min_scale_factor(len(letters))
    n_units = (GRID * k) ** 2
    print(f"Modo alfabeto: p={len(letters)} letras, k={k} → grilla "
          f"{GRID * k}×{GRID * k} = {n_units} neuronas (p/N={len(letters)/n_units:.4f})")

    patterns = np.stack([scaled_letter_vector(c, k) for c in letters])
    net = HopfieldNetwork(n_units=patterns.shape[1])
    net.store(patterns)

    out = os.path.join(cfg["output_dir"], "alphabet")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(cfg.get("seed", 42))

    stored = {c: p.astype(np.int32) for c, p in zip(letters, patterns)}

    from hopfield.plot_letters import plot_letters
    plot_letters(letters, os.path.join(out, f"alphabet_k{k}.png"), scale=k)

    plot_crosstalk(stored, os.path.join(out, "crosstalk_alphabet.png"))
    plot_crosstalk_per_neuron(stored, os.path.join(out, "crosstalk_AB_comparison.png"), subset=["A", "B"])
    rates = plot_recovery_rate_vs_noise(
        net, stored,
        noise_levels=cfg.get("noise_levels_analysis", [0.0, 0.05, 0.1, 0.15, 0.2]),
        n_trials=cfg.get("n_trials", 20),
        rng=rng,
        output=os.path.join(out, "recovery_rate_alphabet.png"),
        mode=cfg.get("mode", "sync"),
        max_steps=cfg.get("max_steps", 50),
    )

    # tasa de recuperación al 10% de ruido, una línea por letra
    print(f"\nTasa de recuperación al 10% de ruido (k={k}):")
    noise_levels = cfg.get("noise_levels_analysis", [0.0, 0.05, 0.1, 0.15, 0.2])
    target_noise = 0.10
    if target_noise in noise_levels:
        idx = noise_levels.index(target_noise)
        for name in letters:
            rate = rates[name][idx]
            bar = "#" * int(rate * 20)
            print(f"  {name}: {rate:.2f}  [{bar:<20}]")

    # diagnóstico sin ruido: qué pasa con cada letra
    wrong_stored = []
    spurious_list = []
    fixed_points = []
    for name in letters:
        target = stored[name].astype(np.int8)
        final, _, _, _ = net.recall(target, mode=cfg.get("mode", "sync"),
                                    max_steps=cfg.get("max_steps", 50))
        if np.array_equal(final, target):
            fixed_points.append(name)
        elif net.is_stored(final) != -1:
            wrong_stored.append(name)
        else:
            spurious_list.append(name)

    print(f"\n=== Diagnóstico sin ruido ===")
    print(f"Puntos fijos (se reconocen)      ({len(fixed_points)}): {' '.join(fixed_points) or 'ninguna'}")
    print(f"Converge a otra letra            ({len(wrong_stored)}): {' '.join(wrong_stored) or 'ninguna'}")
    print(f"Cae en estado espúreo            ({len(spurious_list)}): {' '.join(spurious_list) or 'ninguna'}")

    # recall paso a paso: 4 que se reconocen + 2 que no, sin ruido
    good = fixed_points
    bad  = (wrong_stored + spurious_list)[:2]

    rng2 = np.random.default_rng(cfg.get("seed", 42))
    print(f"\n=== Recall paso a paso (sin ruido) ===")
    for letter in good + bad:
        run_recall(
            net, letter,
            noise=0.0,
            mode=cfg.get("mode", "sync"),
            max_steps=cfg.get("max_steps", 50),
            rng=rng2,
            output_dir=out,
            tag="alphabet",
            scale=k,
        )

    print(f"\nResultados del modo alfabeto en {out}/")


def main():
    parser = argparse.ArgumentParser(description="Hopfield — Ejercicio Patrones")
    parser.add_argument("--config", default="configs/hopfield.json")
    parser.add_argument(
        "--alphabet", action="store_true",
        help="Modo alfabeto: almacena las 26 letras con escalado adaptativo "
             "y reporta recall por letra + crosstalk",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)

    if args.alphabet:
        run_alphabet_mode(cfg)
        return

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

    # ---- Plots auxiliares: crosstalk del subset y curva ruido/recall ----
    stored = {c: p.astype(np.int32) for c, p in zip(letters, patterns)}

    # imagen vertical con los patrones almacenados
    fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.8))
    for ax, (name, pat) in zip(axes.ravel(), stored.items()):
        ax.imshow(np.where(pat.reshape(5, 5) == 1, 1.0, 0.0),
                  cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stored_patterns.png"), dpi=150)
    plt.close(fig)

    plot_crosstalk(stored, os.path.join(out, "crosstalk.png"))
    plot_crosstalk_per_neuron(stored, os.path.join(out, "crosstalk_per_neuron.png"))
    plot_recovery_rate_vs_noise(
        net, stored,
        noise_levels=cfg.get("noise_levels_analysis",
                             [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]),
        n_trials=cfg.get("n_trials", 50),
        rng=np.random.default_rng(cfg.get("seed", 42) + 99),
        output=os.path.join(out, "recovery_rate.png"),
        mode=cfg.get("mode", "sync"),
        max_steps=cfg.get("max_steps", 50),
    )

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
