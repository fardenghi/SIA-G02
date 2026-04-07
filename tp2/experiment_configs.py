"""
Experiment script: tests all key parameter combinations and reports best config per image type.
Short runs (N_GENS generations) for fast comparison.
Results saved to output/experiments/
"""

import subprocess
import csv
import os
import time
import json
from pathlib import Path
from datetime import datetime

# ─── Experiment settings ──────────────────────────────────────────────────────
N_GENS = 300
POPULATION = 50
TRIANGLES = 30
BASE_DIR = Path("output/experiments")
INPUT_DIR = Path("input")

# Representative images (simple, medium, complex)
IMAGES = {
    "bangladesh": "input/bangladesh.png",   # Very simple: green bg + red circle
    "argentina": "input/argentina.png",      # Medium: stripes + sun
    "chile": "input/chile.png",              # Medium-complex: flag with star
}

# Base (default) config — we vary one dimension at a time
BASE = {
    "selection": "tournament",
    "crossover": "uniform",
    "mutation": "uniform_multigen",
    "survival": "additive",
    "fitness": "linear",
}

# ─── Experiment matrix ────────────────────────────────────────────────────────
# Each list is a phase; we test each value while holding others at BASE

EXPERIMENTS = [
    # Phase 1: Selection
    [{"selection": m} for m in ["elite", "tournament", "probabilistic_tournament",
                                  "roulette", "universal", "boltzmann", "rank"]],
    # Phase 2: Crossover
    [{"crossover": m} for m in ["single_point", "two_point", "uniform", "annular"]],
    # Phase 3: Mutation
    [{"mutation": m} for m in ["single_gene", "limited_multigen", "uniform_multigen", "complete"]],
    # Phase 4: Survival
    [{"survival": m} for m in ["additive", "exclusive"]],
    # Phase 5: Fitness function
    [{"fitness": m} for m in ["linear", "rmse", "inverse_normalized", "exponential"]],
]


def run_single(image_path: str, config: dict, run_label: str) -> dict:
    """Run a single experiment and return result metrics."""
    image_name = Path(image_path).stem
    out_dir = BASE_DIR / run_label / image_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python3", "main.py",
        "--image", image_path,
        "--generations", str(N_GENS),
        "--population", str(POPULATION),
        "--triangles", str(TRIANGLES),
        "--output", str(out_dir),
        "--selection", config["selection"],
        "--crossover", config["crossover"],
        "--mutation", config["mutation"],
        "--survival", config["survival"],
        "--fitness", config["fitness"],
        "--save-interval", "0",
        "--max-size", "128",
        "--quiet",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0

    # Parse best fitness from output summary or metrics CSV
    best_fitness = None
    metrics_csv = out_dir / "metrics.csv"
    if metrics_csv.exists():
        try:
            with open(metrics_csv) as f:
                rows = list(csv.DictReader(f))
                if rows:
                    best_fitness = max(float(r["best_fitness"]) for r in rows)
        except Exception:
            pass

    if best_fitness is None:
        # Try parsing stdout
        for line in result.stdout.splitlines():
            if "Best fitness:" in line:
                try:
                    best_fitness = float(line.split(":")[-1].strip())
                except ValueError:
                    pass

    return {
        "image": image_name,
        "run_label": run_label,
        **config,
        "best_fitness": best_fitness,
        "elapsed_s": round(elapsed, 1),
        "returncode": result.returncode,
    }


def run_experiments():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    phase_names = ["selection", "crossover", "mutation", "survival", "fitness"]

    for phase_idx, phase_variants in enumerate(EXPERIMENTS):
        phase_name = phase_names[phase_idx]
        print(f"\n{'='*60}")
        print(f"PHASE {phase_idx+1}: {phase_name.upper()}")
        print(f"{'='*60}")

        for variant in phase_variants:
            # Build full config: BASE + override
            config = {**BASE, **variant}
            variant_val = list(variant.values())[0]
            run_label = f"p{phase_idx+1}_{phase_name}_{variant_val}"

            for img_name, img_path in IMAGES.items():
                print(f"  [{img_name}] {variant_val} ...", end=" ", flush=True)
                try:
                    result = run_single(img_path, config, run_label)
                    all_results.append(result)
                    bf = result["best_fitness"]
                    print(f"fitness={bf:.5f} ({result['elapsed_s']}s)")
                except subprocess.TimeoutExpired:
                    print("TIMEOUT")
                    all_results.append({
                        "image": img_name, "run_label": run_label, **config,
                        "best_fitness": None, "elapsed_s": 300, "returncode": -1
                    })
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({
                        "image": img_name, "run_label": run_label, **config,
                        "best_fitness": None, "elapsed_s": 0, "returncode": -2
                    })

    # Save all results
    results_path = BASE_DIR / "results.csv"
    if all_results:
        keys = list(all_results[0].keys())
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {results_path}")

    return all_results


def analyze_results(results: list):
    """Print a summary ranking for each phase and image type."""
    from collections import defaultdict

    phase_names = ["selection", "crossover", "mutation", "survival", "fitness"]
    dim_to_phase = {
        "selection": "selection",
        "crossover": "crossover",
        "mutation": "mutation",
        "survival": "survival",
        "fitness": "fitness",
    }

    print("\n" + "="*60)
    print("ANALYSIS: Best config per dimension and image type")
    print("="*60)

    for dim in dim_to_phase:
        print(f"\n--- {dim.upper()} ---")
        # Group by image and method value
        by_image_method = defaultdict(dict)
        for r in results:
            if r.get("best_fitness") is not None:
                by_image_method[r["image"]][r[dim]] = r["best_fitness"]

        for img_name in IMAGES:
            scores = by_image_method.get(img_name, {})
            if not scores:
                continue
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print(f"  {img_name}:")
            for rank, (method, score) in enumerate(ranked, 1):
                print(f"    {rank}. {method:30s} fitness={score:.5f}")

    # Best overall combo suggestion
    print("\n--- BEST OVERALL SUGGESTION ---")
    by_dim = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("best_fitness") is not None:
            for dim in dim_to_phase:
                by_dim[dim][r[dim]].append(r["best_fitness"])

    for dim in dim_to_phase:
        avg_scores = {m: sum(v)/len(v) for m, v in by_dim[dim].items()}
        best = max(avg_scores, key=avg_scores.get)
        print(f"  {dim:12s}: {best}  (avg={avg_scores[best]:.5f})")


if __name__ == "__main__":
    print(f"Starting experiments: {N_GENS} gens, pop={POPULATION}, triangles={TRIANGLES}")
    print(f"Images: {list(IMAGES.keys())}")
    print(f"Total runs: ~{sum(len(p) for p in EXPERIMENTS) * len(IMAGES)}")
    t_start = time.time()

    results = run_experiments()
    analyze_results(results)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")
