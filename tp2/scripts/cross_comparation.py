"""
Cross comparation script for crossover schemes.

Runs repeated experiments on low/medium/high images and compares fixed crossover
methods against a phased hybrid scheme (early two-point -> late uniform).

The fitness method is fixed to inverse_normalized.
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
    image_key: str
    image_path: str
    run_idx: int
    strategy_key: str


IMAGE_PRESETS: list[tuple[str, str]] = [
    ("low", "input/image-low-px.jpg"),
    ("medium", "input/image-medium-px.png"),
    ("high", "input/image-high-px.jpg"),
]

STRATEGIES: list[CrossoverStrategy] = [
    CrossoverStrategy(
        key="single_point",
        label="single_point",
        method="single_point",
    ),
    CrossoverStrategy(
        key="two_point",
        label="two_point",
        method="two_point",
    ),
    CrossoverStrategy(
        key="uniform",
        label="uniform",
        method="uniform",
    ),
    CrossoverStrategy(
        key="hybrid_two_to_uniform",
        label="hybrid(2pt->uniform)",
        method="two_point",
        phased_enabled=True,
        early_method="two_point",
        late_method="uniform",
    ),
]

STRATEGY_BY_KEY: dict[str, CrossoverStrategy] = {s.key: s for s in STRATEGIES}
_IMAGE_CACHE: dict[tuple[str, int], Image.Image] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross comparation for fixed vs hybrid crossover schemes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per image")
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
        default="output/cross_comparation",
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

    target_image = _load_image_cached(task.image_path, run_cfg["max_size"])

    worker_args = argparse.Namespace(**run_cfg)
    engine = build_engine(target_image, strategy, worker_args)
    result = engine.run()

    curve = np.array([h["best_fitness"] for h in result.history], dtype=np.float64)
    target_len = run_cfg["generations"] + 1
    if len(curve) < target_len:
        curve = np.pad(curve, (0, target_len - len(curve)), mode="edge")

    return {
        "image": task.image_key,
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


def _build_tasks(args: argparse.Namespace) -> list[ExperimentTask]:
    tasks: list[ExperimentTask] = []
    for image_key, image_path in IMAGE_PRESETS:
        for run_idx in range(args.runs):
            for strategy in STRATEGIES:
                tasks.append(
                    ExperimentTask(
                        image_key=image_key,
                        image_path=image_path,
                        run_idx=run_idx,
                        strategy_key=strategy.key,
                    )
                )
    return tasks


def run_all_experiments(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[tuple[str, str], list[np.ndarray]]]:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories: dict[tuple[str, str], list[np.ndarray]] = {}
    records: list[dict] = []

    tasks = _build_tasks(args)
    total_runs = len(tasks)
    run_cfg = {
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
                    f"[{done:03d}/{total_runs}] image={task.image_key:<6} run={task.run_idx + 1:02d}/{args.runs} "
                    f"strategy={task.strategy_key}"
                )
                histories.setdefault((res["image"], res["strategy"]), []).append(
                    res["curve"]
                )
                records.append({k: v for k, v in res.items() if k != "curve"})
    else:
        print("Parallel mode disabled; running sequentially")
        for idx, task in enumerate(tasks, start=1):
            print(
                f"[{idx:03d}/{total_runs}] image={task.image_key:<6} run={task.run_idx + 1:02d}/{args.runs} "
                f"strategy={task.strategy_key}"
            )
            res = _run_task(task, run_cfg)
            histories.setdefault((res["image"], res["strategy"]), []).append(
                res["curve"]
            )
            records.append({k: v for k, v in res.items() if k != "curve"})

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by=["image", "run", "strategy"]).reset_index(drop=True)
    return df, histories


def plot_final_fitness_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(IMAGE_PRESETS), figsize=(17, 5), sharey=True)
    strategy_order = [s.key for s in STRATEGIES]
    label_map = {s.key: s.label for s in STRATEGIES}

    for ax, (image_key, _) in zip(axes, IMAGE_PRESETS):
        subset = df[df["image"] == image_key]
        data = [
            np.asarray(
                subset.loc[subset["strategy"] == key, "best_fitness"],
                dtype=np.float64,
            )
            for key in strategy_order
        ]
        ax.boxplot(
            data, tick_labels=[label_map[k] for k in strategy_order], showmeans=True
        )
        ax.set_title(f"{image_key} image")
        ax.set_xlabel("crossover strategy")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    axes[0].set_ylabel("final best fitness")
    fig.suptitle(
        "Final fitness distribution (10 runs per image)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "final_fitness_boxplots.png", dpi=150)
    plt.close(fig)


def plot_mean_curves(
    histories: dict[tuple[str, str], list[np.ndarray]],
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    x = np.arange(args.generations + 1)
    fig, axes = plt.subplots(1, len(IMAGE_PRESETS), figsize=(17, 5), sharey=True)

    for ax, (image_key, _) in zip(axes, IMAGE_PRESETS):
        for strategy in STRATEGIES:
            runs = np.stack(histories[(image_key, strategy.key)], axis=0)
            mean_curve = runs.mean(axis=0)
            std_curve = runs.std(axis=0)

            ax.plot(x, mean_curve, label=strategy.label, linewidth=1.8)
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.15,
            )

        ax.set_title(f"{image_key} image")
        ax.set_xlabel("generation")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("best fitness (mean +/- std)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Convergence curves across runs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fitness_curves_mean_std.png", dpi=150)
    plt.close(fig)


def plot_win_rate_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    best_idx = df.groupby(["image", "run"])["best_fitness"].idxmax()
    winners_df = df.loc[best_idx, ["image", "run", "strategy"]]

    win_rate = winners_df.groupby(["image", "strategy"]).size().reset_index(name="wins")
    totals = winners_df.groupby("image").size().reset_index(name="total_runs")
    win_rate = win_rate.merge(totals, on="image", how="left")
    win_rate["win_rate"] = win_rate["wins"] / win_rate["total_runs"]

    matrix = (
        win_rate.pivot(index="image", columns="strategy", values="win_rate")
        .reindex(
            index=[i for i, _ in IMAGE_PRESETS], columns=[s.key for s in STRATEGIES]
        )
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(matrix.values, vmin=0.0, vmax=1.0, cmap="YlGn")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([s.label for s in STRATEGIES], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title("Win rate by image (fraction of runs where strategy is best)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(j, i, f"{100.0 * value:.0f}%", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="win rate")
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate_heatmap.png", dpi=150)
    plt.close(fig)


def save_tables(df: pd.DataFrame, out_dir: Path) -> None:
    raw_path = out_dir / "cross_comparation_raw.csv"
    summary_path = out_dir / "cross_comparation_summary.csv"

    summary = df.groupby(["image", "strategy"], as_index=False).agg(
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

    print("\nTop strategy by image (mean final fitness):")
    best_idx = summary.groupby("image")["best_fitness_mean"].idxmax()
    best_by_image = summary.loc[best_idx].set_index("image")
    for image_key, _ in IMAGE_PRESETS:
        top = best_by_image.loc[image_key]
        print(
            f"- {image_key:>6}: {top['strategy']:<24} "
            f"fitness={top['best_fitness_mean']:.6f} +/- {top['best_fitness_std']:.6f}"
        )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running crossover comparation")
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
    plot_final_fitness_boxplots(df, out_dir)
    plot_mean_curves(histories, args, out_dir)
    plot_win_rate_heatmap(df, out_dir)

    print("Plots saved:")
    print(f"- {out_dir / 'final_fitness_boxplots.png'}")
    print(f"- {out_dir / 'fitness_curves_mean_std.png'}")
    print(f"- {out_dir / 'win_rate_heatmap.png'}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
