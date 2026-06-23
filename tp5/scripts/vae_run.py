"""Runner canónico del VAE base: entrena con split train/val y emite SIEMPRE los plots.

Por convención del proyecto, toda corrida emite:
  - `in_recon_samples.png`: entrada / reconstrucción / muestras (muestreo del prior N(0,I)).
  - `loss_curves.png`: train vs val (recon BCE/px, determinista z=μ) con la época óptima.
  - `kl_per_dim.png`: KL por dimensión latente (uso del cuello, dims muertas vs activas).
  - `posterior_stats.png`: histogramas de μ y σ del posterior (colapso / ignorar la entrada).
  - `nn_samples.png`: cada muestra del prior y su vecino real más cercano (memorización).
Además imprime el SSIM medio de reconstrucción (train/val), complemento del BCE/px.

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
from autoencoder.minecraft_data import load_minecraft  # noqa: E402
from autoencoder.mnist_data import load_mnist  # noqa: E402
from autoencoder.vae import VAE  # noqa: E402
from autoencoder import vae_metrics_viz  # noqa: E402
from autoencoder.vae_train import EarlyStopping, train_vae  # noqa: E402

# Caras puras (sin animales/objetos): set grande PERO homogéneo. 110 glifos distintos —
# suficientes para no memorizar, homogéneos para que el latente bajo abarque la variación.
FACES_RANGES = [(0x1F600, 0x1F64A), (0x1F910, 0x1F917), (0x1F920, 0x1F92F),
                (0x1F970, 0x1F97A), (0x1F60E, 0x1F60E)]


def recon_px(vae: VAE, data: np.ndarray) -> float:
    """Recon determinista (z=μ) por píxel: BCE/px, comparable a `recon_det/px`."""
    mu, _ = vae.encode(data)
    return float(vae._loss_value(vae.decode(mu), data))


def recon_ssim(vae: VAE, data: np.ndarray, size: int) -> float:
    """SSIM medio de la reconstrucción determinista (z=μ): calidad estructural, ~1 = idéntica.

    Complementa al BCE/px: el SSIM no arrastra el piso de entropía del gris, así que es la
    métrica de recon más interpretable para el informe ("reconstruye bien" = SSIM alto)."""
    recon = vae.decode(vae.encode(data)[0])
    return float(vae_metrics_viz.ssim(recon, data, size).mean())


# Familia de salida por dataset: UNA carpeta legible por dataset (no una por hiperparámetro).
# Las variantes de emojis (curated/many/faces/emoji_multi) van todas a `emojis`.
_DATASET_DIR = {"celeba": "celebA", "fashion": "fashion", "mnist": "mnist",
                "minecraft": "minecraft", "minecraft-old": "minecraft"}


def output_dir(dataset: str, tag: str | None = None) -> Path:
    """Carpeta `out/vae_run/<familia>` del dataset (celebA/emojis/fashion/mnist).

    Una sola carpeta por dataset: corridas nuevas del mismo dataset SOBRESCRIBEN los plots, en
    vez de generar una carpeta distinta por cada combinación de hiperparámetros. Pasá `tag` para
    comparar configs sin pisarte: la salida va a `<familia>_<tag>` (p. ej. `fashion_L8`).
    """
    family = _DATASET_DIR.get(dataset, "emojis")
    if tag:
        family = f"{family}_{tag}"
    return Path("out/vae_run") / family


def split_data(args, rng):
    """Devuelve (Xtr, Xval, n_distinct). Sin fuga según `--val-mode` (ver docstring)."""
    if args.dataset in ("mnist", "fashion", "celeba", "emoji_multi", "minecraft", "minecraft-old"):
        # Datasets densos de muestras genuinamente distintas (no copias aumentadas) -> split
        # aleatorio simple, sin riesgo de fuga. n_aug se ignora (el dataset ya es denso).
        if args.dataset == "celeba":
            X_all, _ = load_celeba(n=args.max_n or 3000, size=args.size, seed=args.seed)
        elif args.dataset == "emoji_multi":
            X_all, _ = load_multi_emojis(size=args.size, seed=args.seed)
        elif args.dataset in ("minecraft", "minecraft-old"):
            X_all, _ = load_minecraft(size=args.size, color=args.color,
                                      n=args.max_n, seed=args.seed,
                                      blocks_only=args.blocks_only,
                                      classic=(args.dataset == "minecraft-old"))
        else:
            digits = [int(d) for d in args.digits.split(",")] if args.digits else None
            X_all, _ = load_mnist(n=args.max_n, digits=digits, seed=args.seed, kind=args.dataset,
                                  size=args.size)
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
    # Eje Y anclado en 0 (menos zoom): la brecha train/val se ve en su escala real, sin exagerar.
    ax.set_ylim(bottom=0)
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
    ins = X_show[:n]
    recon = vae.decode(vae.encode(ins)[0])
    samples = sample_fn(n, rng)
    rows = [(ins, "in"), (recon, "recon"), (samples, "samples")]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.1, 3 * 1.25))
    for r, (block, name) in enumerate(rows):
        for j in range(n):
            vae_metrics_viz.show_image(axes[r, j], block[j], size)  # color-aware (gris/RGB)
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
            vae_metrics_viz.show_image(axes[r, c], imgs[c], size)  # color-aware (gris/RGB)
        axes[r, 0].set_ylabel(f"{a}→{b}", rotation=0, ha="right", va="center", fontsize=8)
    for c, t in enumerate(ts):
        axes[0, c].set_title(f"{t:.1f}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="VAE base con train/val + plots canónicos")
    p.add_argument("--dataset",
                   choices=["curated", "many", "faces", "mnist", "fashion", "celeba",
                            "emoji_multi", "minecraft", "minecraft-old"],
                   default="many")
    p.add_argument("--digits", default=None,
                   help="MNIST/Fashion: subconjunto de clases, ej '0,1,8' (None = las 10)")
    p.add_argument("--color", action="store_true",
                   help="usa texturas RGB en vez de grises (solo para minecraft)")
    p.add_argument("--blocks-only", action="store_true",
                   help="filtra escaleras, vallas, plantas, etc. (solo minecraft)")
    p.add_argument("--val-mode", choices=["disjoint", "poses"], default="disjoint")
    p.add_argument("--size", type=int, default=28)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--latent2", type=int, default=2, help="latente de la etapa 2 (two-stage)")
    p.add_argument("--no-stage2", action="store_true", help="saltea el two-stage (solo etapa 1)")
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
    p.add_argument("--tag", default=None,
                   help="sufijo opcional de la carpeta de salida (<familia>_<tag>): úsalo para "
                        "comparar configs del mismo dataset sin sobrescribirte (p. ej. --tag L8)")
    p.add_argument("--interpolate", type=int, default=0,
                   help="emite interpolation.png con N parejas (0=off): interpola linealmente "
                        "el latente de la etapa 1 entre dos imágenes reales (demo de 'sin huecos')")
    p.add_argument("--interp-steps", type=int, default=11,
                   help="pasos t∈[0,1] de la interpolación (columnas de la figura)")
    p.add_argument("--save-model", action="store_true",
                   help="guarda model.npz (etapa1 + etapa2 + stats + config de la corrida) en la "
                        "carpeta del run")
    p.add_argument("--load-model", default=None,
                   help="carga un model.npz y NO reentrena: sólo emite plots. La config guardada "
                        "(dataset/seed/n_aug/...) MANDA, así que basta el path — el resto de los "
                        "flags se ignoran. Omite loss_curves (no hay historial)")
    args = p.parse_args(argv)

    # ---- modo carga: la config guardada en el checkpoint MANDA sobre los flags de CLI ----- #
    # Cargamos el .npz primero: si trae `run_args`, esos valores reemplazan a los de la CLI
    # (solo se preserva la ruta de carga), así el split y los títulos salen idénticos a los del
    # entrenamiento sin reescribir un solo flag. Checkpoints viejos (sin run_args) usan la CLI.
    ck = None
    if args.load_model:
        ck = load_vae(args.load_model)
        if "run_args" in ck:
            load_path = args.load_model
            args = argparse.Namespace(**{**vars(args), **ck["run_args"]})
            args.load_model, args.save_model = load_path, False
            print(f"  config tomada del checkpoint: dataset={args.dataset} size={args.size} "
                  f"latente={args.latent} seed={args.seed} n_aug={args.n_aug} "
                  f"val_mode={args.val_mode}")
        else:
            print("  checkpoint sin run_args: uso los flags de CLI para el split")

    # D con canales: minecraft a color es RGB (3 canales); el resto, escala de grises (1).
    n_channels = 3 if (args.color and args.dataset in ("minecraft", "minecraft-old")) else 1
    D = args.size * args.size * n_channels
    hidden = [int(h) for h in args.hidden.split(",")]
    rng = np.random.default_rng(args.seed)
    out = output_dir(args.dataset, args.tag); out.mkdir(parents=True, exist_ok=True)

    Xtr, Xval, n_distinct = split_data(args, rng)
    print(f"dataset={args.dataset} ({n_distinct} glifos)  val-mode={args.val_mode}  "
          f"train={Xtr.shape[0]}  val={Xval.shape[0]}  {args.size}x{args.size}  "
          f"latente {args.latent}")

    # ---- modo carga: reconstruir el modelo guardado y saltar el entrenamiento ----------- #
    if args.load_model:
        vae = ck["vae1"]
        sampler = make_sampler(ck)                          # prior -> imagen (two-stage)
        es = None                                           # sin historial -> sin loss_curves
        print(f"  modelo cargado de {args.load_model} (sin reentrenar)  "
              f"recon train {recon_px(vae, Xtr):.4f} /px   val {recon_px(vae, Xval):.4f} /px   "
              f"SSIM train {recon_ssim(vae, Xtr, args.size):.4f}")
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
    print(f"  SSIM recon (z=μ):          train {recon_ssim(vae, Xtr, args.size):.4f}      "
          f"val {recon_ssim(vae, Xval, args.size):.4f}   (1 = idéntica)")

    # ---- etapa 2 (two-stage): modela los códigos de la etapa 1 para generar ------------- #
    if args.no_stage2:
        sampler = lambda n, rng: vae.decode(rng.standard_normal((n, args.latent)))
        vae2, cm, cs = None, None, None
    else:
        print(f"\n== ETAPA 2 (two-stage): VAE [{args.latent}, 64, 32] -> {args.latent2}  "
              f"({args.epochs2} ep) ==")
        sampler, vae2, cm, cs = fit_stage2_sampler(vae, Xtr, args.latent2, args.epochs2, args.seed,
                                                   batch_size=args.batch)

    # ---- persistir el modelo entrenado (etapa1 + etapa2 + stats) ------------------------ #
    if args.save_model:
        # Guardamos la config de la corrida (sin los flags de orquestación): el modo carga la
        # usa para reconstruir el MISMO split y los títulos sin reescribir los flags.
        run_args = {k: v for k, v in vars(args).items() if k not in ("load_model", "save_model")}
        path = save_vae(out / "model.npz", vae, args.size, vae2=vae2, code_mean=cm, code_std=cs,
                        run_args=run_args)
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
    stage2_label = f"L2={args.latent2}" if not args.no_stage2 else "prior directo"
    plot_in_recon_samples(
        vae, Xtr, args.size, rng, sampler,
        f"VAE — {args.dataset} latente {args.latent} h[{args.hidden}] "
        f"({args.size}×{args.size}), recon TRAIN + samples ({stage2_label})",
        out / "in_recon_samples.png")
    print(f"-> {out}/in_recon_samples.png")

    # ---- diagnóstico del posterior: uso del cuello y salud de q(z|x) -------------------- #
    vae_metrics_viz.plot_kl_per_dim(vae, Xtr, path=out / "kl_per_dim.png")
    print(f"-> {out}/kl_per_dim.png")
    vae_metrics_viz.plot_posterior_stats(vae, Xtr, path=out / "posterior_stats.png")
    print(f"-> {out}/posterior_stats.png")

    # ---- memorización vs generación: cada sample del prior y su vecino real más cercano -- #
    n_nn = min(12, Xtr.shape[0])
    vae_metrics_viz.plot_sample_neighbors(
        sampler(n_nn, rng), Xtr, args.size, path=out / "nn_samples.png")
    print(f"-> {out}/nn_samples.png")

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
