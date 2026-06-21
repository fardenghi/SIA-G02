"""Runner canónico del VAE base: entrena con split train/val y emite SIEMPRE dos plots.

Por convención del proyecto, toda corrida emite:
  - `in_recon_samples.png`: entrada / reconstrucción / muestras (muestreo del prior N(0,I)).
  - `loss_curves.png`: train vs val (recon BCE/px, determinista z=μ) con la época óptima.

Split train/val SIN fuga de datos (clave en este dataset de pocas plantillas + augment):
  - `--val-mode disjoint` (default): conjunto DISJUNTO de glifos para val (clases nunca vistas).
    Único régimen donde entrenar de más degrada la val → curva de épocas óptimas honesta.
    Pensado para el set grande (`--dataset many`, ~220 glifos caras+animales).
  - `--val-mode poses`: poses nuevas (augment con otra semilla) de las mismas plantillas.
    Mide robustez a pose; con pocos emojis la val se aplana (memorización).

    uv run python scripts/vae_run.py --dataset many --val-mode disjoint --latent 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder.emoji_data import (  # noqa: E402
    FACE_ANIMAL_RANGES,
    augment_dataset,
    load_emojis,
    load_many_emojis,
)
from autoencoder.celeba_data import load_celeba  # noqa: E402
from autoencoder.checkpoint import load_vae, make_sampler, save_vae  # noqa: E402
from autoencoder.conv_vae import ConvVAE  # noqa: E402
from autoencoder.emoji_multi_data import load_multi_emojis  # noqa: E402
from autoencoder.mnist_data import load_mnist  # noqa: E402
from autoencoder.vae import VAE  # noqa: E402
from autoencoder.vae_train import EarlyStopping, train_vae  # noqa: E402

# Caras puras (sin animales/objetos): set grande PERO homogéneo. 110 glifos distintos —
# suficientes para no memorizar, homogéneos para que el latente bajo abarque la variación.
FACES_RANGES = [(0x1F600, 0x1F64A), (0x1F910, 0x1F917), (0x1F920, 0x1F92F),
                (0x1F970, 0x1F97A), (0x1F60E, 0x1F60E)]


def recon_px(vae: VAE, data: np.ndarray) -> float:
    """Recon determinista (z=μ) por píxel: BCE/px, comparable a `recon_det/px`."""
    mu, _ = vae.encode(data)
    return float(vae._loss_value(vae.decode(mu), data))


def split_data(args, rng):
    """Devuelve (Xtr, Xval, n_distinct). Sin fuga según `--val-mode` (ver docstring)."""
    if args.dataset in ("mnist", "fashion", "celeba", "emoji_multi"):
        # Datasets densos de muestras genuinamente distintas (no copias aumentadas) -> split
        # aleatorio simple, sin riesgo de fuga. n_aug se ignora (el dataset ya es denso).
        if args.dataset == "celeba":
            X_all, _ = load_celeba(n=args.max_n or 3000, size=args.size, seed=args.seed)
        elif args.dataset == "emoji_multi":
            X_all, _ = load_multi_emojis(size=args.size, seed=args.seed)
        else:
            digits = [int(d) for d in args.digits.split(",")] if args.digits else None
            X_all, _ = load_mnist(n=args.max_n, digits=digits, seed=args.seed, kind=args.dataset)
        n_distinct = X_all.shape[0]
        perm = rng.permutation(n_distinct)
        n_val = max(1, int(round(args.val_frac * n_distinct)))
        Xval, Xtr = X_all[perm[:n_val]], X_all[perm[n_val:]]
        return Xtr, Xval, n_distinct
    if args.dataset == "faces":
        X_all, labels = load_many_emojis(size=args.size, ranges=FACES_RANGES, max_n=args.max_n)
    elif args.dataset == "many":
        X_all, labels = load_many_emojis(size=args.size, ranges=FACE_ANIMAL_RANGES,
                                         max_n=args.max_n)
    else:
        X_all, labels = load_emojis(size=args.size)
    n_distinct = X_all.shape[0]

    if args.val_mode == "disjoint":
        # Conjunto disjunto de GLIFOS: val = clases nunca vistas en train.
        perm = rng.permutation(n_distinct)
        n_val = max(1, int(round(args.val_frac * n_distinct)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        tr_base = X_all[tr_idx]
        tr_labels = [labels[i] for i in tr_idx]
        Xval = X_all[val_idx]                                  # glifos disjuntos, sin augment
        Xtr, _ = augment_dataset(tr_base, tr_labels, size=args.size, n_aug=args.n_aug,
                                 rng=np.random.default_rng(args.seed))
    else:
        # Poses nuevas de las mismas plantillas (otra semilla); val sin los originales.
        Xtr, _ = augment_dataset(X_all, labels, size=args.size, n_aug=args.n_aug,
                                 rng=np.random.default_rng(args.seed))
        val_aug, _ = augment_dataset(X_all, labels, size=args.size, n_aug=4,
                                     rng=np.random.default_rng(999))
        Xval = val_aug[n_distinct:]
    return Xtr, Xval, n_distinct


def plot_loss_curves(ep, tr, va, best_epoch, stop_epoch, best_val, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, tr, label="train", color="tab:blue")
    ax.plot(ep, va, label="val", color="tab:orange")
    ax.axvline(best_epoch, color="tab:green", ls="--", lw=1.2,
               label=f"óptimo (val min, restaurado) = {best_epoch} ep")
    if stop_epoch != best_epoch:
        ax.axvline(stop_epoch, color="tab:red", ls=":", lw=1.2,
                   label=f"early stop = {stop_epoch} ep")
    ax.scatter([best_epoch], [best_val], color="tab:green", zorder=5)
    ax.set_xlabel("época"); ax.set_ylabel("recon BCE / px (determinista, z=μ)")
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fit_stage2_sampler(vae1, X, latent2, epochs, seed, batch_size=None):
    """Entrena la etapa 2 (VAE sobre los códigos de la etapa 1) y devuelve un sampler two-stage.

    Estándar de generación del proyecto: la etapa 2 modela el posterior agregado de la etapa 1
    (reemplaza al GMM). Generar = z₂~N(0,I) → decode etapa2 → des-estandarizar → decode etapa1.
    Los códigos se estandarizan (z-score) para que el MSE de la etapa 2 trate las dims parejo.
    """
    mu1, _ = vae1.encode(X)                                  # (N, L1)
    cm, cs = mu1.mean(axis=0), mu1.std(axis=0) + 1e-8
    codes = (mu1 - cm) / cs
    vae2 = VAE([vae1.latent_dim, 64, 32], latent_dim=latent2, activation="relu",
               output_activation="linear", init="he_normal", loss="mse", seed=seed)
    train_vae(vae2, codes, epochs=epochs, lr=1e-3, beta=1.0, beta_warmup=epochs // 5,
              seed=seed, lr_schedule="cosine", batch_size=batch_size)

    def sample(n, rng):
        z2 = rng.standard_normal((n, latent2))
        codes_hat = vae2.decode(z2) * cs + cm                # des-estandarizar
        return vae1.decode(codes_hat)                        # decode etapa 1
    return sample, vae2, cm, cs                              # cm/cs: para persistir el checkpoint


def plot_in_recon_samples(vae, X_show, size, rng, sample_fn, title, path, n=12):
    n = min(n, X_show.shape[0])
    ins = X_show[:n]                                         # entradas: glifos de TRAIN
    recon = vae.decode(vae.encode(ins)[0])                   # recon determinista (z=μ)
    samples = sample_fn(n, rng)                              # generación (two-stage)
    rows = [(ins, "in"), (recon, "recon"), (samples, "samples")]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.1, 3 * 1.25))
    for r, (block, name) in enumerate(rows):
        for j in range(n):
            axes[r, j].imshow(block[j].reshape(size, size), cmap="Greys", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_interpolation(vae, X, size, rng, n_pairs, n_steps, title, path):
    """Interpolación lineal del latente de la etapa 1 entre dos imágenes reales.

    Demuestra que el latente es un espacio continuo (sin "huecos"): codifica dos imágenes a
    μ_a, μ_b, recorre z_t = (1-t)·μ_a + t·μ_b y decodifica cada paso. Si la transición es
    suave (sin saltos ni basura intermedia) el decoder mapeó un manifold denso y conexo.
    Para que la transición sea visible, la pareja de cada fila es el ANCLA y su código más
    lejano (clases distintas garantizadas). Extremos t=0/t=1 = recon de las dos reales.
    """
    mu, _ = vae.encode(X)                                   # (N, L) códigos deterministas
    n = mu.shape[0]
    n_pairs = min(n_pairs, n // 2)
    anchors = rng.choice(n, size=n_pairs, replace=False)
    ts = np.linspace(0.0, 1.0, n_steps)
    fig, axes = plt.subplots(n_pairs, n_steps, figsize=(n_steps * 1.0, n_pairs * 1.05))
    if n_pairs == 1:
        axes = axes[None, :]
    for r, a in enumerate(anchors):
        b = int(np.argmax(np.linalg.norm(mu - mu[a], axis=1)))   # código más lejano al ancla
        za, zb = mu[a], mu[b]
        zs = np.stack([(1.0 - t) * za + t * zb for t in ts])     # interpolación lineal
        imgs = vae.decode(zs)                                    # (n_steps, D)
        for c in range(n_steps):
            axes[r, c].imshow(imgs[c].reshape(size, size), cmap="Greys", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"{a}→{b}", rotation=0, ha="right", va="center", fontsize=8)
    for c, t in enumerate(ts):
        axes[0, c].set_title(f"{t:.1f}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="VAE base con train/val + plots canónicos")
    p.add_argument("--dataset",
                   choices=["curated", "many", "faces", "mnist", "fashion", "celeba",
                            "emoji_multi"],
                   default="many")
    p.add_argument("--digits", default=None,
                   help="MNIST/Fashion: subconjunto de clases, ej '0,1,8' (None = las 10)")
    p.add_argument("--val-mode", choices=["disjoint", "poses"], default="disjoint")
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--latent2", type=int, default=2, help="latente de la etapa 2 (two-stage)")
    p.add_argument("--stage1", choices=["mlp", "conv"], default="mlp",
                   help="modelo de la etapa 1: MLP-VAE o ConvVAE (sesgo espacial)")
    p.add_argument("--hidden", default="256,64",
                   help="capas ocultas del encoder (coma-sep); el decoder es el espejo")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--epochs2", type=int, default=3000, help="épocas de la etapa 2")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--n-aug", type=int, default=4)
    p.add_argument("--batch", type=int, default=None,
                   help="mini-batch (None=full-batch). Usar p.ej. 128 con datasets grandes")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta-warmup", type=int, default=200,
                   help="épocas de rampa lineal de β (FIJO, no atado a --epochs). El early stop "
                        "no selecciona antes de terminarlo, para no restaurar un modelo sub-β")
    p.add_argument("--patience", type=int, default=10,
                   help="early stop: evaluaciones sin mejora antes de cortar (0=off)")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="mejora mínima de val para resetear la paciencia")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interpolate", type=int, default=0,
                   help="emite interpolation.png con N parejas (0=off): interpola linealmente "
                        "el latente de la etapa 1 entre dos imágenes reales (demo de 'sin huecos')")
    p.add_argument("--interp-steps", type=int, default=11,
                   help="pasos t∈[0,1] de la interpolación (columnas de la figura)")
    p.add_argument("--save-model", action="store_true",
                   help="guarda model.npz (etapa1 + etapa2 + stats) en la carpeta del run")
    p.add_argument("--load-model", default=None,
                   help="carga un model.npz y NO reentrena: sólo emite plots (interpolar/decodificar "
                        "puntos del latente al instante). Omite loss_curves (no hay historial)")
    args = p.parse_args(argv)

    D = args.size * args.size
    hidden = [int(h) for h in args.hidden.split(",")]
    rng = np.random.default_rng(args.seed)
    tag = (f"{args.dataset}_{args.val_mode}_{args.stage1}_L{args.latent}_{args.size}px"
           f"_aug{args.n_aug}_h{'-'.join(map(str, hidden))}_b{args.beta}")
    out = Path(f"out/vae_run/{tag}"); out.mkdir(parents=True, exist_ok=True)

    Xtr, Xval, n_distinct = split_data(args, rng)
    print(f"dataset={args.dataset} ({n_distinct} glifos)  val-mode={args.val_mode}  "
          f"train={Xtr.shape[0]}  val={Xval.shape[0]}  {args.size}x{args.size}  "
          f"latente {args.latent}")

    # ---- modo carga: reconstruir el modelo guardado y saltar el entrenamiento ----------- #
    if args.load_model:
        ck = load_vae(args.load_model)
        vae = ck["vae1"]
        sampler = make_sampler(ck)                          # prior -> imagen (two-stage)
        es = None                                           # sin historial -> sin loss_curves
        print(f"  modelo cargado de {args.load_model} (sin reentrenar)  "
              f"recon train {recon_px(vae, Xtr):.4f} /px   val {recon_px(vae, Xval):.4f} /px")
        _emit_plots(vae, es, Xtr, args, rng, sampler, out)
        return

    # ---- entrenar con early stopping (independiente del tamaño del dataset) -------------- #
    if args.stage1 == "conv":
        vae = ConvVAE(size=args.size, latent_dim=args.latent, conv_channels=[16, 32],
                      dense_hidden=hidden[-1], activation="relu",
                      output_activation="sigmoid", init="he_normal", loss="bce",
                      seed=args.seed)
        print(f"  arquitectura: ConvVAE canales=[16,32] dense={hidden[-1]} -> {args.latent}  "
              f"({vae.n_params} params)")
    else:
        vae = VAE([D, *hidden], latent_dim=args.latent, activation="relu",
                  output_activation="sigmoid", init="he_normal", loss="bce", seed=args.seed)
        print(f"  arquitectura: encoder [{D}, {', '.join(map(str, hidden))}] -> {args.latent}  "
              f"({vae.n_params} params)")
    # patience=0 desactiva el corte (paciencia infinita): corre las `epochs` completas.
    patience = args.patience if args.patience > 0 else args.epochs
    # warmup de β FIJO (no atado al cap): así β llega al target temprano y el modelo entrena la
    # mayor parte a β completo. El early stop no selecciona antes de terminarlo (start_epoch).
    es = EarlyStopping(val_fn=lambda m: recon_px(m, Xval),
                       train_fn=lambda m: recon_px(m, Xtr),
                       patience=patience, min_delta=args.min_delta,
                       start_epoch=args.beta_warmup)
    train_vae(vae, Xtr, epochs=args.epochs, lr=1e-3, beta=args.beta,
              beta_warmup=args.beta_warmup, seed=args.seed, batch_size=args.batch,
              callback=es, callback_every=args.eval_every)
    # Con early stopping (patience>0) restauramos el óptimo de val. Con --patience 0
    # conservamos los pesos FINALES (diagnóstico de capacidad: cuánto puede memorizar).
    if args.patience > 0:
        es.restore(vae)

    ep = np.array(es.epochs); tr = np.array(es.train); va = np.array(es.val)
    best_epoch, best_val = es.best_epoch, es.best
    stopped = es.stopped_epoch if es.stopped_epoch >= 0 else int(ep[-1])
    print(f"\n== EARLY STOPPING ==")
    print(f"  mejor val:      {best_val:.4f} /px  @ epoca {best_epoch}  (pesos restaurados)")
    if es.stopped_epoch >= 0:
        print(f"  cortado en:     epoca {stopped}  (paciencia {patience} x {args.eval_every} ep "
              f"sin mejora)")
    else:
        print(f"  sin corte:      llego al tope de {args.epochs} epocas")
    train_at_best = float(tr[ep == best_epoch][0]) if (ep == best_epoch).any() else float("nan")
    print(f"  train @ optimo: {train_at_best:.4f} /px   gap {best_val-train_at_best:+.4f}")
    # Recon honesta del modelo restaurado: train (lo que el decoder SI puede reproducir) vs
    # val (generalizacion a glifos no vistos). La brecha es la generalizacion, no un bug.
    print(f"  recon (modelo restaurado): train {recon_px(vae, Xtr):.4f} /px   "
          f"val {recon_px(vae, Xval):.4f} /px")

    # ---- etapa 2 (two-stage): modela los códigos de la etapa 1 para generar ------------- #
    print(f"\n== ETAPA 2 (two-stage): VAE [{args.latent}, 64, 32] -> {args.latent2}  "
          f"({args.epochs2} ep) ==")
    sampler, vae2, cm, cs = fit_stage2_sampler(vae, Xtr, args.latent2, args.epochs2, args.seed,
                                               batch_size=args.batch)

    # ---- persistir el modelo entrenado (etapa1 + etapa2 + stats) ------------------------ #
    if args.save_model:
        path = save_vae(out / "model.npz", vae, args.size, vae2=vae2, code_mean=cm, code_std=cs)
        print(f"-> {path}  (modelo guardado: recargá con --load-model)")

    _emit_plots(vae, es, Xtr, args, rng, sampler, out)


def _emit_plots(vae, es, Xtr, args, rng, sampler, out):
    """Emite los plots canónicos + interpolación. `es=None` (modo carga) omite loss_curves."""
    if es is not None:                                      # sólo si hubo entrenamiento
        ep = np.array(es.epochs); tr = np.array(es.train); va = np.array(es.val)
        stopped = es.stopped_epoch if es.stopped_epoch >= 0 else int(ep[-1])
        plot_loss_curves(ep, tr, va, es.best_epoch, stopped, es.best,
                         f"Train vs Val — {args.dataset} {args.val_mode} latente {args.latent} "
                         f"({args.size}×{args.size}) — early stop @ {stopped}",
                         out / "loss_curves.png")
        print(f"-> {out}/loss_curves.png")
    plot_in_recon_samples(
        vae, Xtr, args.size, rng, sampler,
        f"VAE — {args.dataset} latente {args.latent} h[{args.hidden}] "
        f"({args.size}×{args.size}), recon TRAIN + samples two-stage (L2={args.latent2})",
        out / "in_recon_samples.png")
    print(f"-> {out}/in_recon_samples.png")

    # ---- interpolación lineal del latente (demo de continuidad / 'sin huecos') ---------- #
    if args.interpolate > 0:
        plot_interpolation(
            vae, Xtr, args.size, rng, args.interpolate, args.interp_steps,
            f"Interpolación lineal del latente (etapa 1) — {args.dataset} latente {args.latent} "
            f"({args.size}×{args.size}) — extremos = recon de 2 reales",
            out / "interpolation.png")
        print(f"-> {out}/interpolation.png")


if __name__ == "__main__":
    main()
