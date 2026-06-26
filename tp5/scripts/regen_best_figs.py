#!/usr/bin/env python3
"""Regenera las figuras del encoder ELEGIDO (configs/1a2/06_best.json) para la ppt:
scatter latente 2D + interpolaciones multi-paso (d->y cercanos, p->z lejanos), todo
desde el MISMO modelo entrenado, para que las escalas/posiciones del latente sean
consistentes entre las slides 12, 14 y 15.

Uso:
    uv run python scripts/regen_best_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autoencoder import viz  # noqa: E402
from autoencoder.data import labels_for_subset, load_font, select_subset, to_grid  # noqa: E402
from autoencoder.train import train_multi_restart  # noqa: E402

OUT = Path("out/1a2/plots/1a2_06_best")


def plot_interpolation(net, X, labels, from_lbl, to_lbl, steps, path):
    """Interpolación lineal en el latente entre dos labels -> decode -> binariza."""
    label_list = [str(l) for l in labels]
    i, j = label_list.index(from_lbl), label_list.index(to_lbl)
    Z = net.encode(X)
    ts = np.linspace(0.0, 1.0, steps)
    zs = np.stack([(1 - t) * Z[i] + t * Z[j] for t in ts])
    imgs = (net.decode(zs) >= 0.5).astype(float)

    step_labels = [f"'{from_lbl}'"] + ["nueva"] * (steps - 2) + [f"'{to_lbl}'"]
    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.4, 2.2))
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(to_grid(imgs[k]), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(step_labels[k], fontsize=8)
    fig.suptitle(f"Interpolación en el latente: '{from_lbl}' -> '{to_lbl}'")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"-> {path}")


def main():
    # Datos (set completo de 32 glifos)
    X = select_subset(load_font("font/font.h"), None)
    labels = labels_for_subset(None)

    # Encoder ELEGIDO = configs/1a2/06_best.json
    print("== Entrenando encoder elegido (06_best): [35,25,15,8,2], cosine, 30 restarts ==")
    best_net, _, _ = train_multi_restart(
        X,
        encoder_layers=[35, 25, 15, 8, 2],
        activation="tanh",
        output_activation="sigmoid",
        init="xavier_normal",
        latent_activation="linear",
        loss="bce",
        optimizer="adam",
        epochs=15000,
        lr=0.003,
        restarts=30,
        seed=42,
        lr_schedule="cosine",
        lr_min=0.0,
        log_every=300,
        stop_at=None,
    )

    recon = best_net.forward(X)
    exact = int(((recon >= 0.5).astype(int) == (X >= 0.5).astype(int)).all(axis=1).sum())
    print(f"  -> patrones exactos: {exact}/{X.shape[0]}")

    Z = best_net.encode(X)
    viz.plot_latent_scatter(Z, labels, path=OUT / "latent_scatter.png")
    print(f"-> {OUT / 'latent_scatter.png'}")
    viz.plot_reconstruction(X, recon, labels, path=OUT / "reconstruction.png")
    print(f"-> {OUT / 'reconstruction.png'}")

    # Interpolaciones de la ppt, desde el MISMO modelo
    # Slide 14: par CERCANO -> transición coherente
    plot_interpolation(best_net, X, labels, "d", "y", 7, OUT / "interp_d_y.png")
    # Slide 15 (original): p->z, que en ESTE latente quedan cerca (no ilustra "lejanos")
    plot_interpolation(best_net, X, labels, "p", "z", 10, OUT / "interp_p_z.png")

    # Slide 15 (corregida): par REALMENTE distante en este latente -> "basura" intermedia.
    # Se elige automáticamente el par de glifos más separado en el espacio latente.
    n = X.shape[0]
    dmax, pair = -1.0, (0, 1)
    for a in range(n):
        for b in range(a + 1, n):
            d = float(np.linalg.norm(Z[a] - Z[b]))
            if d > dmax:
                dmax, pair = d, (a, b)
    a, b = pair
    fa, fb = str(labels[a]), str(labels[b])
    print(f"  -> par más distante en el latente: '{fa}'<->'{fb}'  (dist={dmax:.2f})")
    plot_interpolation(best_net, X, labels, fa, fb, 10, OUT / "interp_far.png")


if __name__ == "__main__":
    main()
