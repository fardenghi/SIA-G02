import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from hopfield.network import HopfieldNetwork
from hopfield.patterns import GRID, LETTERS, add_noise, as_vector, identify

_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hopfield", ["#F5F0E8", "#1E3A5F"]  # cream (inactive) → dark blue (active)
)


def _build_net(letter_names: list[str]) -> tuple[HopfieldNetwork, dict[str, np.ndarray]]:
    stored = {n: as_vector(n) for n in letter_names}
    net = HopfieldNetwork()
    net.train(np.array(list(stored.values())))
    return net, stored


# --- plotting helpers ---

def _pattern_ax(ax, pattern: np.ndarray, title: str) -> None:
    grid = pattern.reshape(GRID, GRID)
    ax.imshow(grid, cmap=_CMAP, vmin=-1, vmax=1, interpolation="nearest")
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
) -> None:
    n_steps = len(history)
    n_cols = min(n_steps + 1, 8)
    fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2.5))
    if n_cols == 1:
        axes = [axes]

    for idx in range(n_cols - 1):
        label = "consulta" if idx == 0 else f"paso {idx}"
        _pattern_ax(axes[idx], history[idx], label)

    match = identify(history[-1], stored)
    final_label = f"final ({match})" if match else "final (espúreo)"
    _pattern_ax(axes[-1], history[-1], final_label)

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
) -> None:
    names = list(stored.keys())
    n_stored = len(names)
    fig, axes = plt.subplots(1, n_stored + 1, figsize=(2 * (n_stored + 1), 2.8))

    for idx, name in enumerate(names):
        _pattern_ax(axes[idx], stored[name], f"'{name}'\nH={net.energy(stored[name]):.2f}")

    e_spur = net.energy(spurious)
    _pattern_ax(axes[-1], spurious, f"espúreo\nH={e_spur:.2f}")
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hopfield.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    out_dir = "output/hopfield"
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    letter_names: list[str] = cfg["letters"]

    net, stored = _build_net(letter_names)

    print("Patrones almacenados:")
    for name, v in stored.items():
        from hopfield.patterns import render
        print(f"\n--- {name} ---")
        print(render(v))

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
            from hopfield.patterns import render
            e = net.energy(state)
            print(f"  [{label}] H={e:.3f}")

        plot_recovery_steps(
            noisy, history, v, name, stored,
            os.path.join(out_dir, f"recovery_{name}.png"),
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
        print(render(spurious))
        print(f"Energía del espúreo: {net.energy(spurious):.3f}")
        print("Energías de patrones almacenados:")
        for name, v in stored.items():
            print(f"  '{name}': {net.energy(v):.3f}")
        plot_spurious(
            spurious, origin, stored, net,
            os.path.join(out_dir, "spurious_state.png"),
        )
        print("Plot de estado espúreo guardado.")
    else:
        print("No se encontró estado espúreo con el nivel de ruido configurado.")

    print(f"\nTodos los plots guardados en {out_dir}/")


if __name__ == "__main__":
    main()
