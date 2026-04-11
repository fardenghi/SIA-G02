"""
Crossover comparison for a single logo image.

Compares fixed crossover methods against the hybrid phased scheme
(early two-point -> late uniform) across multiple runs.

Fitness method is fixed to inverse_normalized.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genetic.engine import EvolutionConfig, create_engine
from src.genetic.mutation import create_mutation_params
from src.rendering.canvas import resize_image


@dataclass(frozen=True)
class CrossoverStrategy:
    key: str
    label: str
    method: str
    phased_enabled: bool = False
    early_method: str = "two_point"
    late_method: str = "uniform"


@dataclass(frozen=True)
class ExperimentTask:
    run_idx: int
    strategy_key: str


STRATEGIES: list[CrossoverStrategy] = [
    CrossoverStrategy("single_point", "single_point", "single_point"),
    CrossoverStrategy("two_point", "two_point", "two_point"),
    CrossoverStrategy("uniform", "uniform", "uniform"),
    CrossoverStrategy(
        "hybrid_two_to_uniform",
        "hybrid(2pt->uniform)",
        "two_point",
        phased_enabled=True,
        early_method="two_point",
        late_method="uniform",
    ),
]

STRATEGY_BY_KEY: dict[str, CrossoverStrategy] = {s.key: s for s in STRATEGIES}
_IMAGE_CACHE: dict[tuple[str, int], Image.Image] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crossover comparison on a single logo image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, default="input/logo.jpg")
    parser.add_argument("--runs", type=int, default=10, help="Runs per strategy")
    parser.add_argument("--generations", "-g", type=int, default=600)
    parser.add_argument("--population", "-p", type=int, default=100)
    parser.add_argument("--triangles", "-t", type=int, default=100)
    parser.add_argument("--max-size", type=int, default=128)
    parser.add_argument("--crossover-probability", type=float, default=0.8)
    parser.add_argument("--switch-ratio", type=float, default=0.7)
    parser.add_argument(
        "--mutation-method",
        choices=["single_gene", "limited_multigen", "uniform_multigen", "complete"],
        default="uniform_multigen",
        help="Mutation operator",
    )
    parser.add_argument("--mutation-probability", type=float, default=0.3)
    parser.add_argument("--gene-probability", type=float, default=0.1)
    parser.add_argument("--max-genes", type=int, default=3)
    parser.add_argument("--position-delta", type=float, default=0.1)
    parser.add_argument("--color-delta", type=int, default=30)
    parser.add_argument("--alpha-delta", type=float, default=0.1)
    parser.add_argument("--field-probability", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--renderer", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multiprocessing execution",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/cross_comparation_logo",
        help="Directory for csv and plots",
    )
    return parser.parse_args()


def build_engine(
    target_image: Image.Image,
    strategy: CrossoverStrategy,
    args: argparse.Namespace,
):
    evo_config = EvolutionConfig(
        population_size=args.population,
        num_triangles=args.triangles,
        max_generations=args.generations,
        phased_crossover_enabled=strategy.phased_enabled,
        phased_crossover_early_method=strategy.early_method,
        phased_crossover_late_method=strategy.late_method,
        phased_crossover_switch_ratio=args.switch_ratio,
    )

    mutation_params = create_mutation_params(
        mutation_method=args.mutation_method,
        probability=args.mutation_probability,
        gene_probability=args.gene_probability,
        max_genes=args.max_genes,
        position_delta=args.position_delta,
        color_delta=args.color_delta,
        alpha_delta=args.alpha_delta,
        field_probability=args.field_probability,
    )

    return create_engine(
        target_image=target_image,
        config=evo_config,
        selection_method="tournament",
        tournament_size=3,
        crossover_method=strategy.method,
        crossover_probability=args.crossover_probability,
        mutation_params=mutation_params,
        survival_method="exclusive",
        survival_selection_method="elite",
        offspring_ratio=1.0,
        fitness_method="inverse_normalized",
        renderer=args.renderer,
    )


def _load_image_cached(image_path: str, max_size: int) -> Image.Image:
    key = (image_path, max_size)
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached

    img = Image.open(image_path).convert("RGB")
    resized = resize_image(img, max_size=max_size)
    _IMAGE_CACHE[key] = resized
    return resized


def _run_task(task: ExperimentTask, run_cfg: dict) -> dict:
    strategy = STRATEGY_BY_KEY[task.strategy_key]
    run_seed = run_cfg["seed"] + task.run_idx

    random.seed(run_seed)
    np.random.seed(run_seed)

    target_image = _load_image_cached(run_cfg["image_path"], run_cfg["max_size"])
    worker_args = argparse.Namespace(**run_cfg)

    engine = build_engine(target_image, strategy, worker_args)
    result = engine.run()

    curve = np.array([h["best_fitness"] for h in result.history], dtype=np.float64)
    target_len = run_cfg["generations"] + 1
    if len(curve) < target_len:
        curve = np.pad(curve, (0, target_len - len(curve)), mode="edge")

    return {
        "run": task.run_idx,
        "seed": run_seed,
        "strategy": strategy.key,
        "best_fitness": result.best_fitness,
        "error": 1.0 - result.best_fitness,
        "elapsed_time_s": result.elapsed_time,
        "generations_executed": result.generations,
        "fitness_method": "inverse_normalized",
        "curve": curve,
    }


def _resolve_workers(args: argparse.Namespace) -> int:
    if args.renderer == "gpu":
        return 1
    if args.workers is not None and args.workers > 0:
        return args.workers
    cpu = os.cpu_count() or 2
    return max(1, min(6, cpu - 1))


def run_all_experiments(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}. Place logo at input/logo.png or use --image."
        )

    tasks = [
        ExperimentTask(run_idx=run_idx, strategy_key=strategy.key)
        for run_idx in range(args.runs)
        for strategy in STRATEGIES
    ]

    run_cfg = {
        "image_path": str(image_path),
        "population": args.population,
        "triangles": args.triangles,
        "generations": args.generations,
        "switch_ratio": args.switch_ratio,
        "crossover_probability": args.crossover_probability,
        "mutation_method": args.mutation_method,
        "mutation_probability": args.mutation_probability,
        "gene_probability": args.gene_probability,
        "max_genes": args.max_genes,
        "position_delta": args.position_delta,
        "color_delta": args.color_delta,
        "alpha_delta": args.alpha_delta,
        "field_probability": args.field_probability,
        "seed": args.seed,
        "max_size": args.max_size,
        "renderer": args.renderer,
        "runs": args.runs,
    }

    histories: dict[str, list[np.ndarray]] = {}
    records: list[dict] = []
    total = len(tasks)

    if args.parallel:
        workers = _resolve_workers(args)
        if args.renderer == "gpu" and workers == 1:
            print("GPU renderer detected: forcing workers=1 to avoid GPU contention")
        print(f"Parallel mode enabled with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_task = {
                pool.submit(_run_task, task, run_cfg): task for task in tasks
            }
            done = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                res = future.result()
                done += 1
                print(
                    f"[{done:03d}/{total}] run={task.run_idx + 1:02d}/{args.runs} strategy={task.strategy_key}"
                )
                histories.setdefault(res["strategy"], []).append(res["curve"])
                records.append({k: v for k, v in res.items() if k != "curve"})
    else:
        print("Parallel mode disabled; running sequentially")
        for idx, task in enumerate(tasks, start=1):
            print(
                f"[{idx:03d}/{total}] run={task.run_idx + 1:02d}/{args.runs} strategy={task.strategy_key}"
            )
            res = _run_task(task, run_cfg)
            histories.setdefault(res["strategy"], []).append(res["curve"])
            records.append({k: v for k, v in res.items() if k != "curve"})

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by=["run", "strategy"]).reset_index(drop=True)
    return df, histories


def plot_final_fitness_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    strategy_order = [s.key for s in STRATEGIES]
    labels = [s.label for s in STRATEGIES]
    data = [
        np.asarray(df.loc[df["strategy"] == key, "best_fitness"], dtype=np.float64)
        for key in strategy_order
    ]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("final best fitness")
    ax.set_xlabel("crossover strategy")
    ax.set_title("Final fitness distribution on logo.png")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "final_fitness_boxplot.png", dpi=150)
    plt.close(fig)


def plot_mean_curves(
    histories: dict[str, list[np.ndarray]], args: argparse.Namespace, out_dir: Path
) -> None:
    x = np.arange(args.generations + 1)
    fig, ax = plt.subplots(figsize=(10, 5.2))

    for strategy in STRATEGIES:
        runs = np.stack(histories[strategy.key], axis=0)
        mean_curve = runs.mean(axis=0)
        std_curve = runs.std(axis=0)
        ax.plot(x, mean_curve, label=strategy.label, linewidth=1.8)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)

    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness (mean +/- std)")
    ax.set_title("Convergence curves on logo.png")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fitness_curves_mean_std.png", dpi=150)
    plt.close(fig)


def plot_win_rate_bar(df: pd.DataFrame, out_dir: Path) -> None:
    best_idx = df.groupby("run")["best_fitness"].idxmax()
    winners = df.loc[best_idx, "strategy"]

    counts = winners.value_counts(normalize=True)
    strategy_order = [s.key for s in STRATEGIES]
    labels = [s.label for s in STRATEGIES]
    rates = [float(counts.get(k, 0.0)) for k in strategy_order]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"][: len(labels)]
    bars = ax.bar(labels, rates, color=colors)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{100.0 * rate:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("win rate")
    ax.set_xlabel("crossover strategy")
    ax.set_title("Win rate on logo.png (best strategy per run)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate_bar.png", dpi=150)
    plt.close(fig)


def save_tables(df: pd.DataFrame, out_dir: Path) -> None:
    raw_path = out_dir / "cross_logo_raw.csv"
    summary_path = out_dir / "cross_logo_summary.csv"

    summary = df.groupby(["strategy"], as_index=False).agg(
        runs=("best_fitness", "size"),
        best_fitness_mean=("best_fitness", "mean"),
        best_fitness_std=("best_fitness", "std"),
        error_mean=("error", "mean"),
        error_std=("error", "std"),
        elapsed_time_mean_s=("elapsed_time_s", "mean"),
        elapsed_time_std_s=("elapsed_time_s", "std"),
    )

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Raw metrics saved: {raw_path}")
    print(f"Summary saved:     {summary_path}")

    fitness_values = np.asarray(summary["best_fitness_mean"], dtype=float)
    best_idx = int(np.argmax(fitness_values))
    best_row = summary.iloc[best_idx]
    print(
        "Best strategy (mean final fitness): "
        f"{best_row['strategy']} -> {best_row['best_fitness_mean']:.6f} "
        f"+/- {best_row['best_fitness_std']:.6f}"
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running single-image crossover comparation")
    print(f"Image: {args.image}")
    print(
        "Settings: "
        f"runs={args.runs}, generations={args.generations}, population={args.population}, "
        f"triangles={args.triangles}, max_size={args.max_size}, renderer={args.renderer}"
    )
    print(
        "Mutation: "
        f"method={args.mutation_method}, p={args.mutation_probability}, gp={args.gene_probability}, "
        f"max_genes={args.max_genes}"
    )
    print("Fitness method fixed to: inverse_normalized")
    print()

    df, histories = run_all_experiments(args)

    save_tables(df, out_dir)
    plot_final_fitness_boxplot(df, out_dir)
    plot_mean_curves(histories, args, out_dir)
    plot_win_rate_bar(df, out_dir)

    print("Plots saved:")
    print(f"- {out_dir / 'final_fitness_boxplot.png'}")
    print(f"- {out_dir / 'fitness_curves_mean_std.png'}")
    print(f"- {out_dir / 'win_rate_bar.png'}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
