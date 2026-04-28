"""Fase 1 – Exploración y análisis descriptivo del dataset de fraude.

Ejecutar desde la raíz del proyecto:
    uv run python -m exercises.ej1_fraud.explore
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exercises.ej1_fraud.data import FEATURE_COLS, LABEL_COL, TARGET_COL, _DATA_DIR

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "outputs" / "ej1_fraud" / "plots"
_OUT.mkdir(parents=True, exist_ok=True)


def _save(name):
    path = _OUT / name
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  guardado: {path}")


def run_eda():
    df = pd.read_csv(_DATA_DIR / "fraud_dataset.csv")

    # ── 1. Inspección general ────────────────────────────────────────────────
    print("=" * 60)
    print("1. INSPECCIÓN GENERAL")
    print("=" * 60)
    print(f"Filas: {df.shape[0]:,}   Columnas: {df.shape[1]}")
    print(f"Valores faltantes:\n{df.isnull().sum().to_string()}")

    # ── 2. Estadísticas descriptivas ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. ESTADÍSTICAS DESCRIPTIVAS")
    print("=" * 60)
    print(df[FEATURE_COLS].describe().round(3).to_string())

    # ── 3. Target ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. TARGET: big_model_fraud_probability")
    print("=" * 60)
    print(df[TARGET_COL].describe().round(4).to_string())
    print(f"\nflagged_fraud (binario):\n{df[LABEL_COL].value_counts().to_string()}")
    fraud_rate = df[LABEL_COL].mean() * 100
    print(f"Tasa de fraude: {fraud_rate:.1f}%")

    _plot_target_distribution(df)

    # ── 4. Correlaciones ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. CORRELACIONES CON EL TARGET")
    print("=" * 60)
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    print(corr.sort_values(key=abs, ascending=False).round(4).to_string())

    _plot_correlation_matrix(df)

    # ── 5. Outliers ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. OUTLIERS (|z| > 3)")
    print("=" * 60)
    for col in FEATURE_COLS:
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        n = (z > 3).sum()
        pct = n / len(df) * 100
        print(f"  {col:<35} {n:>4} ({pct:.1f}%)")

    _plot_feature_boxplots(df)

    # ── 6. Resumen de decisiones ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. DECISIONES DE PREPROCESAMIENTO")
    print("=" * 60)
    print("""
  - Sin valores faltantes → no hay filas a eliminar (D1 N/A)
  - timestamp descartado  → correlación ~0, no es predictivo
  - Normalización: z-score sobre las 8 features restantes
  - Outliers mantenidos   → representan transacciones reales
  - Features de baja correlación (device_screen_resolution,
    time_since_last_login_s) conservadas; el modelo decidirá
    su peso final vía los pesos aprendidos
""")


def _plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(df[TARGET_COL], bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribución de big_model_fraud_probability")
    axes[0].set_xlabel("Probabilidad de fraude")
    axes[0].set_ylabel("Frecuencia")

    counts = df[LABEL_COL].value_counts().sort_index()
    axes[1].bar(["No fraude (0)", "Fraude (1)"], counts.values,
                color=["steelblue", "tomato"])
    axes[1].set_title("Distribución de flagged_fraud")
    axes[1].set_ylabel("Cantidad de transacciones")
    
    # Expand y-axis limit to make room for text labels
    max_count = counts.max()
    axes[1].set_ylim(0, max_count * 1.15)
    
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + (max_count * 0.02), f"{v}\n({v/len(df)*100:.1f}%)", ha="center")

    plt.tight_layout()
    _save("target_distribution.png")


def _plot_correlation_matrix(df):
    numeric_cols = FEATURE_COLS + [TARGET_COL]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    labels = [c.replace("_", "\n") for c in numeric_cols]
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=6)

    ax.set_title("Matriz de correlación")
    plt.tight_layout()
    _save("correlation_matrix.png")


def _plot_feature_boxplots(df):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        axes[i].boxplot(df[col], vert=True, patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.6))
        axes[i].set_title(col.replace("_", "\n"), fontsize=8)
        axes[i].tick_params(labelbottom=False)
    plt.suptitle("Distribución de features (boxplots)")
    plt.tight_layout()
    _save("feature_boxplots.png")


if __name__ == "__main__":
    run_eda()
