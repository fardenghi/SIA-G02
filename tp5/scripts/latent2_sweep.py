"""Barrido de `latent2` (cuello de la etapa 2 del two-stage) sobre una etapa 1 FIJA.

Aísla el efecto del cuello de la etapa 2: reusa el encoder/decoder de la etapa 1 cargado de un
checkpoint (con su `run_args`, para reconstruir el MISMO split) y solo reentrena la etapa 2 con
cada `latent2`. Así la única variable es la capacidad de la etapa 2 para modelar la nube de
códigos `q(z)`.

Emite dos figuras en la carpeta del checkpoint:
  - `latent2_sweep_samples.png`: una fila de muestras generadas por cada `latent2` (calidad
    cualitativa — deberían afinarse al subir latent2 y luego saturar).
  - `latent2_sweep_curve.png`: MSE de reconstrucción de los códigos (¿capta la nube?) y nº de
    unidades activas de la etapa 2 vs `latent2`. Ambas saturan en la dimensión intrínseca de
    los códigos: esa saturación es la "optimización del two-stage" medida.

    uv run python scripts/latent2_sweep.py --model out/vae_run/fashion/model.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # para importar vae_run

from autoencoder.checkpoint import load_vae  # noqa: E402
from autoencoder.vae_metrics_viz import active_units, show_image  # noqa: E402
from vae_run import fit_stage2_sampler, split_data  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="Barrido de latent2 con etapa 1 fija (checkpoint)")
    p.add_argument("--model", required=True, help="model.npz de la etapa 1 (debe traer run_args)")
    p.add_argument("--latent2", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--epochs2", type=int, default=800)
    p.add_argument("--n-samples", type=int, default=12)
    args = p.parse_args(argv)

    ck = load_vae(args.model)
    if "run_args" not in ck:
        sys.exit("el checkpoint no trae run_args: no puedo reconstruir el split de la etapa 1")
    vae1, size = ck["vae1"], ck["size"]
    sargs = argparse.Namespace(**ck["run_args"])               # config de la etapa 1
    rng = np.random.default_rng(sargs.seed)
    Xtr, _Xval, _ = split_data(sargs, rng)
    print(f"etapa 1 fija: latente {vae1.latent_dim}  train={Xtr.shape[0]}  {size}x{size}")

    # códigos (medias) de la etapa 1, estandarizados — el "dataset" de la etapa 2.
    mu1, _ = vae1.encode(Xtr)
    cm, cs = mu1.mean(axis=0), mu1.std(axis=0) + 1e-8
    codes_std = (mu1 - cm) / cs
    batch = getattr(sargs, "batch", None)

    rows = []          # (latent2, code_mse, active_units)
    sample_grids = []  # (latent2, samples)
    for L2 in args.latent2:
        sample_fn, vae2, _, _ = fit_stage2_sampler(vae1, Xtr, L2, args.epochs2, sargs.seed,
                                                   batch_size=batch)
        # MSE de reconstrucción de los códigos: qué tan fiel modela la etapa 2 la nube q(z).
        codes_hat = vae2.decode(vae2.encode(codes_std)[0])
        mse = float(((codes_hat - codes_std) ** 2).mean())
        au = active_units(vae2, codes_std)                     # dims usadas por la etapa 2
        rows.append((L2, mse, au))
        sample_grids.append((L2, sample_fn(args.n_samples, np.random.default_rng(0))))
        print(f"  latent2={L2:2d}  code-MSE={mse:.4f}  active_units={au}/{L2}")

    out = Path(args.model).parent

    # ---- figura 1: grilla de samples (una fila por latent2) ----------------------------- #
    n = args.n_samples
    fig, axes = plt.subplots(len(args.latent2), n, figsize=(n * 1.0, len(args.latent2) * 1.15))
    axes = np.atleast_2d(axes)
    for r, (L2, samples) in enumerate(sample_grids):
        for j in range(n):
            show_image(axes[r, j], samples[j], size)           # color-aware (gris/RGB)
        axes[r, 0].set_ylabel(f"L2={L2}", rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Muestras generadas vs latent2 (etapa 1 fija) — afinan y saturan al subir L2")
    fig.tight_layout()
    fig.savefig(out / "latent2_sweep_samples.png", dpi=120); plt.close(fig)
    print(f"-> {out}/latent2_sweep_samples.png")

    # ---- figura 2: code-MSE y unidades activas vs latent2 ------------------------------- #
    from matplotlib.ticker import ScalarFormatter

    L2s = np.array([r[0] for r in rows], dtype=float)
    mses = np.array([r[1] for r in rows], dtype=float)
    aus = np.array([r[2] for r in rows], dtype=float)
    fig, ax_mse = plt.subplots(figsize=(7, 4.5))
    ax_mse.plot(L2s, mses, "o-", color="tab:green", label="MSE recon de códigos")
    ax_mse.set_xscale("log", base=2); ax_mse.set_xticks(L2s)
    ax_mse.xaxis.set_major_formatter(ScalarFormatter())
    ax_mse.set_xlabel("latent2 (escala log₂)")
    ax_mse.set_ylabel("MSE recon de códigos (estandarizados)", color="tab:green")
    ax_mse.grid(True, alpha=0.3)
    ax_au = ax_mse.twinx()
    ax_au.plot(L2s, aus, "s--", color="tab:blue", label="unidades activas (etapa 2)")
    ax_au.plot(L2s, L2s, ":", color="tab:gray", alpha=0.6, label="y=latent2 (todas activas)")
    ax_au.set_ylabel("nº unidades activas (etapa 2)", color="tab:blue")
    ax_au.set_ylim(bottom=0)
    lines = ax_mse.get_lines() + ax_au.get_lines()
    ax_mse.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=9)
    ax_mse.set_title("Etapa 2: fidelidad a la nube de códigos vs latent2")
    fig.tight_layout()
    fig.savefig(out / "latent2_sweep_curve.png", dpi=120); plt.close(fig)
    print(f"-> {out}/latent2_sweep_curve.png")


if __name__ == "__main__":
    main()
