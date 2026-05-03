"""Plot de líneas mostrando el efecto combinado de la regularización (L2+Aug+ES)
sobre el gap train-val durante el entrenamiento.

El área sombreada entre las curvas de train y val es el "gap" — overfitting
visualizado. Un área grande = el modelo está memorizando train mientras val
queda atrás. Un área chica = train y val van juntos = generaliza bien.

Compara baseline_pure (sin regularización) vs best_l2_aug (con todas las técnicas).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_METRICS = _ROOT / "outputs" / "ej3_more_digits" / "metrics"
_OUT = _METRICS / "regularizacion_lines.png"


def main():
    df_sin = pd.read_csv(_METRICS / "baseline_pure.csv")
    df_con = pd.read_csv(_METRICS / "best_l2_aug.csv")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 6))

    # ===== Panel izquierdo: loss =====
    # SIN regularización
    ax_loss.plot(df_sin["epoch"], df_sin["loss_train"], color="#a83232", lw=1.8,
                 linestyle="-", label="Train SIN regularización")
    ax_loss.plot(df_sin["epoch"], df_sin["loss_val"], color="#a83232", lw=1.8,
                 linestyle="--", label="Val   SIN regularización")
    ax_loss.fill_between(df_sin["epoch"], df_sin["loss_train"], df_sin["loss_val"],
                         color="#a83232", alpha=0.18, label="Gap SIN")

    # CON regularización
    ax_loss.plot(df_con["epoch"], df_con["loss_train"], color="#3262a8", lw=1.8,
                 linestyle="-", label="Train CON regularización")
    ax_loss.plot(df_con["epoch"], df_con["loss_val"], color="#3262a8", lw=1.8,
                 linestyle="--", label="Val   CON regularización")
    ax_loss.fill_between(df_con["epoch"], df_con["loss_train"], df_con["loss_val"],
                         color="#3262a8", alpha=0.18, label="Gap CON")

    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.set_yscale("log")
    ax_loss.set_title("Loss durante el entrenamiento", fontsize=12)
    ax_loss.legend(fontsize=8.5, loc="upper right", ncol=2)
    ax_loss.grid(alpha=0.3, which="both")

    # ===== Panel derecho: accuracy =====
    ax_acc.plot(df_sin["epoch"], df_sin["acc_train"], color="#a83232", lw=1.8,
                linestyle="-", label="Train SIN regularización")
    ax_acc.plot(df_sin["epoch"], df_sin["acc_val"], color="#a83232", lw=1.8,
                linestyle="--", label="Val   SIN regularización")
    ax_acc.fill_between(df_sin["epoch"], df_sin["acc_train"], df_sin["acc_val"],
                        color="#a83232", alpha=0.18, label="Gap SIN")

    ax_acc.plot(df_con["epoch"], df_con["acc_train"], color="#3262a8", lw=1.8,
                linestyle="-", label="Train CON regularización")
    ax_acc.plot(df_con["epoch"], df_con["acc_val"], color="#3262a8", lw=1.8,
                linestyle="--", label="Val   CON regularización")
    ax_acc.fill_between(df_con["epoch"], df_con["acc_train"], df_con["acc_val"],
                        color="#3262a8", alpha=0.18, label="Gap CON")

    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy", fontsize=11)
    ax_acc.set_title("Accuracy durante el entrenamiento", fontsize=12)
    ax_acc.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax_acc.grid(alpha=0.3)
    ax_acc.set_ylim(0.85, 1.005)

    fig.suptitle("Sin regularización vs Con regularización (L2 + Aug + Early Stopping)",
                 fontsize=14, y=1.02)
    fig.text(0.5, 0.005,
             "Área sombreada entre train y val = gap (overfitting). "
             "Más grande = más overfitting.",
             ha="center", fontsize=10, style="italic", color="dimgray")

    plt.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(_OUT, dpi=130, bbox_inches="tight")
    print(f"Plot saved to {_OUT}")


if __name__ == "__main__":
    main()
