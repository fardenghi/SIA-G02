"""Train multiple ej3 configs concurrently using multiprocessing.

Each worker is a separate Python process (spawn start method) that trains
one config. Each worker's stdout/stderr is captured into a per-config log
under outputs/ej3_more_digits/parallel_logs/. After all complete, prints
a summary table.

Usage:
    uv run python -m exercises.ej3_more_digits.run_parallel \\
        configs/ej3_more_digits/ensembles/diverse_architectures/*.json \\
        --workers 4
"""

import argparse
import contextlib
import multiprocessing as mp
import os
import time
from pathlib import Path

# Critical for numpy multiprocessing on Mac/Linux: force each worker to use
# a single BLAS thread. Without this, every worker spawns threads for matmul
# and the 4 workers fight over the same cores → 5-10× slowdown vs sequential.
# Must be set BEFORE numpy is imported anywhere.
_THREAD_LIMIT_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate
    "NUMEXPR_NUM_THREADS",
)

_LOG_DIR = Path("outputs/ej3_more_digits/parallel_logs")


def _train_one(config_path):
    # Pin BLAS to a single thread BEFORE importing numpy in this worker
    for var in _THREAD_LIMIT_VARS:
        os.environ[var] = "1"

    cfg_name = Path(config_path).stem
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / f"{cfg_name}.log"

    # Status messages go to the parent's stdout (live progress in terminal),
    # while training output gets captured into the per-config log file below.
    print(f"  [start] {cfg_name}", flush=True)

    t0 = time.time()
    try:
        # buffering=1 → line-buffered, so progress is visible in the log
        # file as each epoch prints, instead of waiting for the run to end.
        with open(log_path, "w", buffering=1) as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                # Lazy import: each worker (spawn) loads run.py fresh
                from exercises.ej3_more_digits.run import run
                run(config_path)
        status = "ok"
        err = ""
    except Exception as e:
        status = "error"
        err = repr(e)

    dt = time.time() - t0
    marker = "✓" if status == "ok" else "✗"
    print(f"  [done ] {cfg_name}  {marker}  ({dt / 60:.1f}m)", flush=True)

    return {
        "config": str(config_path),
        "log": str(log_path),
        "time_s": dt,
        "status": status,
        "error": err,
    }


def main():
    # Pin BLAS to 1 thread per process, also for the parent (inherited by children)
    for var in _THREAD_LIMIT_VARS:
        os.environ[var] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Config JSON paths")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max concurrent processes (default 4)")
    args = parser.parse_args()

    cpu_count = mp.cpu_count()
    workers = min(args.workers, cpu_count, len(args.configs))
    print(f"Training {len(args.configs)} configs with {workers} workers "
          f"(CPU count={cpu_count})")
    for c in args.configs:
        print(f"  - {c}")
    print()

    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_train_one, args.configs)
    total = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"Wallclock total: {total / 60:.1f} min  "
          f"(sum of individual times: {sum(r['time_s'] for r in results) / 60:.1f} min)")
    print(f"Speedup vs sequential: {sum(r['time_s'] for r in results) / total:.2f}x")
    print(f"{'=' * 70}\n")

    print(f"{'config':<60s} {'status':>8s} {'time':>8s}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["config"]):
        cfg_short = "/".join(Path(r["config"]).parts[-3:])
        print(f"{cfg_short:<60s} {r['status']:>8s} {r['time_s'] / 60:>6.1f}m")
        if r["status"] == "error":
            print(f"    error: {r['error']}")

    print(f"\nLogs in {_LOG_DIR}/")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
