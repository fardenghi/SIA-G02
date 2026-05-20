import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from hopfield.network import HopfieldNetwork
from hopfield.patterns import (
    ALL_LETTERS, GRID, LETTERS, add_noise, as_vector, identify,
    min_scale_factor, scale_pattern,
)

_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hopfield", ["#F5F0E8", "#1E3A5F"]  # cream (inactive) → dark blue (active)
)


def _build_net(
    letter_names: list[str], scale: int = 1
) -> tuple[HopfieldNetwork, dict[str, np.ndarray], int]:
    """Return (net, stored_vectors, effective_grid_side)."""
    grid = GRID * scale
    stored = {
        n: scale_pattern(ALL_LETTERS[n], scale).flatten()
        for n in letter_names
    }
    net = HopfieldNetwork()
    net.train(np.array(list(stored.values())))
    return net, stored, grid


# --- plotting helpers ---

def _pattern_ax(ax, pattern: np.ndarray, title: str, grid: int = GRID) -> None:
    ax.imshow(pattern.reshape(grid, grid), cmap=_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_recovery_steps(
    query: np.ndarray,
    history: list[np.ndarray],
    original: np.ndarray,
    letter_name: str,
    stored: dict[str, np.ndarray],
    out_path: str,
    grid: int = GRID,
) -> None:
    n_steps = len(history)
    n_cols = min(n_steps + 1, 8)
    fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2.5))
    if n_cols == 1:
        axes = [axes]

    for idx in range(n_cols - 1):
        label = "consulta" if idx == 0 else f"paso {idx}"
        _pattern_ax(axes[idx], history[idx], label, grid)

    match = identify(history[-1], stored)
    final_label = f"final ({match})" if match else "final (espúreo)"
    _pattern_ax(axes[-1], history[-1], final_label, grid)

    fig.suptitle(f"Recuperación de '{letter_name}'", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_energy_evolution(
    history: list[np.ndarray],
    net: HopfieldNetwork,
    letter_name: str,
    out_path: str,
) -> None:
    energies = [net.energy(s) for s in history]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(len(energies)), energies, marker="o", linewidth=2)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Energía H")
    ax.set_title(f"Energía durante recuperación de '{letter_name}'")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_recovery_rate(
    net: HopfieldNetwork,
    stored: dict[str, np.ndarray],
    noise_levels: list[float],
    n_trials: int,
    rng: np.random.Generator,
    out_path: str,
) -> None:
    letter_names = list(stored.keys())
    rates = {name: [] for name in letter_names}

    for noise in noise_levels:
        for name in letter_names:
            v = stored[name]
            correct = sum(
                identify(net.predict(add_noise(v, noise, rng))[0], stored) == name
                for _ in range(n_trials)
            )
            rates[name].append(correct / n_trials)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in letter_names:
        ax.plot(noise_levels, rates[name], marker="o", label=name, linewidth=2)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="50% ruido")
    ax.set_xlabel("Nivel de ruido (fracción de bits invertidos)")
    ax.set_ylabel("Tasa de recuperación")
    ax.set_title("Tasa de recuperación vs nivel de ruido")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_crosstalk(stored: dict[str, np.ndarray], out_path: str) -> None:
    names = list(stored.keys())
    N = len(next(iter(stored.values())))
    mat = np.array([
        [np.dot(stored[a], stored[b]) / N for b in names]
        for a in names
    ])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Correlación normalizada")
    ax.set_title("Correlación entre patrones almacenados")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# --- spurious state analysis ---

def find_spurious(
    net: HopfieldNetwork,
    stored: dict[str, np.ndarray],
    noise_level: float,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, str | None]:
    for name in stored:
        v = stored[name]
        for _ in range(n_trials):
            noisy = add_noise(v, noise_level, rng)
            result, _ = net.predict(noisy)
            if identify(result, stored) is None:
                return result, name
    return None, None


def plot_spurious(
    spurious: np.ndarray,
    origin_name: str,
    stored: dict[str, np.ndarray],
    net: HopfieldNetwork,
    out_path: str,
    grid: int = GRID,
) -> None:
    names = list(stored.keys())
    n_stored = len(names)
    fig, axes = plt.subplots(1, n_stored + 1, figsize=(2 * (n_stored + 1), 2.8))

    for idx, name in enumerate(names):
        _pattern_ax(axes[idx], stored[name], f"'{name}'\nH={net.energy(stored[name]):.2f}", grid)

    e_spur = net.energy(spurious)
    _pattern_ax(axes[-1], spurious, f"espúreo\nH={e_spur:.2f}", grid)
    axes[-1].spines[:].set_color("red")
    for spine in axes[-1].spines.values():
        spine.set_linewidth(2)

    fig.suptitle(
        f"Estado espúreo encontrado a partir de '{origin_name}'",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# --- main ---

def _mean_recovery(
    letter_names: list[str],
    noise_levels: list[float],
    n_trials: int,
    seed: int,
    scale: int,
) -> float:
    net, stored, _ = _build_net(letter_names, scale)
    rng = np.random.default_rng(seed)
    correct = total = 0
    for noise in noise_levels:
        for name in letter_names:
            v = stored[name]
            for _ in range(n_trials):
                if identify(net.predict(add_noise(v, noise, rng))[0], stored) == name:
                    correct += 1
                total += 1
    return correct / total


def plot_capacity_experiment(
    letter_names: list[str],
    noise_levels: list[float],
    n_trials: int,
    seed: int,
    out_path: str,
) -> None:
    """Recovery rate vs p: fixed N=25 vs adaptive N (scale k = min_scale_factor(p))."""
    subset_sizes = list(range(1, len(letter_names) + 1))
    capacity_limit_fixed = 0.138 * GRID * GRID

    rates_fixed: list[float] = []
    rates_adaptive: list[float] = []

    for p in subset_sizes:
        names = letter_names[:p]
        k = min_scale_factor(p)
        rates_fixed.append(_mean_recovery(names, noise_levels, n_trials, seed, scale=1))
        rates_adaptive.append(_mean_recovery(names, noise_levels, n_trials, seed, scale=k))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(subset_sizes, rates_fixed, marker="o", linewidth=2,
            color="#1E3A5F", label="N fijo (5×5=25)")
    ax.plot(subset_sizes, rates_adaptive, marker="s", linewidth=2,
            color="#2E8B57", label="N adaptativo (escala automática)")
    ax.axvline(x=capacity_limit_fixed, color="red", linestyle="--", linewidth=1.5,
               label=f"Límite teórico N=25 (≈{capacity_limit_fixed:.1f})")
    ax.set_xlabel("Cantidad de patrones almacenados (p)")
    ax.set_ylabel("Tasa de recuperación media")
    ax.set_title("Capacidad fija vs adaptativa")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hopfield.json")
    parser.add_argument(
        "--alphabet", action="store_true",
        help="Use the full 26-letter alphabet and run capacity experiment",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_dir = "output/hopfield"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])

    if args.alphabet:
        letter_names = sorted(ALL_LETTERS.keys())
        out_dir = "output/hopfield/alphabet"
        os.makedirs(out_dir, exist_ok=True)
        p = len(letter_names)
        k = min_scale_factor(p)
        N = (GRID * k) ** 2
        print(f"Modo alfabeto completo: p={p} letras")
        print(f"Escala adaptativa: k={k} → grilla {GRID*k}×{GRID*k} = {N} neuronas")
        print(f"p/N={p/N:.4f}  (límite teórico ≈ 0.138)")

        net, stored, eff_grid = _build_net(letter_names, scale=k)

        plot_crosstalk(stored, os.path.join(out_dir, "crosstalk_alphabet.png"))
        print("Matriz de correlación del alfabeto guardada.")

        plot_capacity_experiment(
            letter_names,
            cfg["noise_levels_analysis"][:5],
            cfg["n_trials"] // 5,
            cfg["seed"],
            os.path.join(out_dir, "capacity_experiment.png"),
        )
        print("Experimento de capacidad guardado.")

        print(f"\nTasa de recuperación por letra con 10% de ruido (k={k}):")
        rng2 = np.random.default_rng(cfg["seed"] + 1)
        for name in letter_names:
            v = stored[name]
            correct = sum(
                identify(net.predict(add_noise(v, 0.1, rng2))[0], stored) == name
                for _ in range(cfg["n_trials"])
            )
            rate = correct / cfg["n_trials"]
            bar = "#" * int(rate * 20)
            print(f"  {name}: {rate:.2f}  [{bar:<20}]")

        print(f"\nTodos los plots guardados en {out_dir}/")
        return

    letter_names = cfg["letters"]
    net, stored, eff_grid = _build_net(letter_names)

    print("Patrones almacenados:")
    for name, v in stored.items():
        from hopfield.patterns import render
        print(f"\n--- {name} ---")
        print(render(v, eff_grid))

    # --- Part a: step-by-step recovery ---
    print("\n=== Recuperación de patrones ruidosos ===")
    for name in letter_names:
        v = stored[name]
        noisy = add_noise(v, cfg["noise_level"], rng)
        result, history = net.predict(noisy, max_iter=cfg["max_iter"])
        match = identify(result, stored)

        print(f"\n'{name}' + {cfg['noise_level']*100:.0f}% ruido → "
              f"{len(history)-1} iteraciones → {match or 'espúreo'}")
        for step, state in enumerate(history):
            label = "consulta" if step == 0 else f"paso {step}"
            e = net.energy(state)
            print(f"  [{label}] H={e:.3f}")

        plot_recovery_steps(
            noisy, history, v, name, stored,
            os.path.join(out_dir, f"recovery_{name}.png"),
            grid=eff_grid,
        )
        plot_energy_evolution(
            history, net, name,
            os.path.join(out_dir, f"energy_{name}.png"),
        )

    # --- Crosstalk matrix ---
    plot_crosstalk(stored, os.path.join(out_dir, "crosstalk.png"))
    print("\nMatriz de correlación entre patrones guardada.")

    # --- Recovery rate vs noise ---
    print("\n=== Análisis de tasa de recuperación ===")
    plot_recovery_rate(
        net, stored,
        cfg["noise_levels_analysis"],
        cfg["n_trials"],
        np.random.default_rng(cfg["seed"] + 1),
        os.path.join(out_dir, "recovery_rate.png"),
    )
    print("Curva de recuperación guardada.")

    # --- Part b: spurious states ---
    print("\n=== Búsqueda de estados espúreos ===")
    spurious, origin = find_spurious(
        net, stored,
        cfg["spurious_noise_level"],
        n_trials=500,
        rng=np.random.default_rng(cfg["seed"] + 2),
    )

    if spurious is not None:
        print(f"Estado espúreo encontrado (partiendo de '{origin}'):")
        from hopfield.patterns import render
        print(render(spurious, eff_grid))
        print(f"Energía del espúreo: {net.energy(spurious):.3f}")
        print("Energías de patrones almacenados:")
        for name, v in stored.items():
            print(f"  '{name}': {net.energy(v):.3f}")
        plot_spurious(
            spurious, origin, stored, net,
            os.path.join(out_dir, "spurious_state.png"),
            grid=eff_grid,
        )
        print("Plot de estado espúreo guardado.")
    else:
        print("No se encontró estado espúreo con el nivel de ruido configurado.")

    print(f"\nTodos los plots guardados en {out_dir}/")


if __name__ == "__main__":
    main()
