"""Compara samples generados sin two-stage vs con two-stage a partir de un modelo guardado.

    uv run python scripts/vae_compare_stages.py --model out/vae_run/fashion/model.npz
    uv run python scripts/vae_compare_stages.py --model out/vae_run/fashion/model.npz --n 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoencoder.checkpoint import load_vae, make_sampler
from autoencoder.vae_metrics_viz import show_image


def main(argv=None):
    p = argparse.ArgumentParser(description="Comparativa prior directo vs two-stage")
    p.add_argument("--model", required=True, help="ruta al .npz guardado con --save-model")
    p.add_argument("--n", type=int, default=12, help="cantidad de muestras a generar")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="ruta de salida (default: junto al modelo)")
    args = p.parse_args(argv)

    ck = load_vae(args.model)
    if "vae2" not in ck:
        raise SystemExit(
            "El checkpoint no tiene etapa 2. "
            "Reentrenar con --save-model sin --no-stage2."
        )

    vae1 = ck["vae1"]
    size = ck["size"]
    rng = np.random.default_rng(args.seed)
    sampler_2stage = make_sampler(ck)

    prior_samples = vae1.decode(rng.standard_normal((args.n, vae1.latent_dim)))
    twostage_samples = sampler_2stage(args.n, rng)

    fig, axes = plt.subplots(2, args.n, figsize=(args.n * 1.1, 2 * 1.5))
    labels = ["prior directo", "two-stage"]
    for r, block in enumerate([prior_samples, twostage_samples]):
        for j in range(args.n):
            show_image(axes[r, j], block[j], size)
        axes[r, 0].set_ylabel(labels[r], rotation=0, ha="right", va="center", fontsize=9)

    dataset = ck.get("run_args", {}).get("dataset", Path(args.model).parent.name)
    fig.suptitle(f"Prior directo vs Two-stage — {dataset}", fontsize=10)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.model).parent / "compare_stages.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
