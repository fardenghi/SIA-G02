#!/usr/bin/env python3
"""
analyze_runs.py — Análisis OFAT completo de las corridas de run_many.sh

Genera por cada eje de configuración (fitness, selection, crossover, mutation, survival):
  · bar chart   — fitness final promedio por valor, agrupado por imagen
  · convergence — curvas de best_fitness por generación, una línea por valor
  · heatmap     — fitness final de todos los ejes para cada imagen
  · summary.csv / best_per_axis.csv — tablas exportables

Uso:
  python analyze_runs.py                        # usa output/run_many
  python analyze_runs.py output/run_many        # explícito
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Configuración ─────────────────────────────────────────────────────────────
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/run_many")
OUT  = BASE / "_analysis"
OUT.mkdir(parents=True, exist_ok=True)

AXES = ["fitness", "selection", "crossover", "mutation", "survival"]

PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ── 1. Cargar todos los CSVs ──────────────────────────────────────────────────
print(f"Buscando métricas en: {BASE}")
dfs = []
for f in sorted(BASE.rglob("metrics.csv")):
    parts = f.parts
    try:
        idx   = list(parts).index(BASE.name)
        image = parts[idx + 1]
        axis  = parts[idx + 2]
        value = parts[idx + 3]
    except (ValueError, IndexError):
        continue
    if axis not in AXES:
        continue
    try:
        tmp = pd.read_csv(f)
    except Exception as e:
        print(f"  [WARN] No se pudo leer {f}: {e}")
        continue
    tmp["image"] = image
    tmp["axis"]  = axis
    tmp["value"] = value
    dfs.append(tmp)

if not dfs:
    print(f"No se encontraron métricas en {BASE}")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)
images = sorted(df["image"].unique())
print(f"Cargados {len(dfs)} runs | Imágenes: {images}\n")

# ── 2. DataFrame de resultados finales (última generación de cada run) ────────
last_gen = df.groupby(["image", "axis", "value"])["generation"].transform("max")
final = (
    df[df["generation"] == last_gen]
    .groupby(["image", "axis", "value"])
    .agg(
        best_fitness   = ("best_fitness",  "max"),
        avg_fitness    = ("avg_fitness",   "mean"),
        worst_fitness  = ("worst_fitness", "mean"),
        total_time_s   = ("time_s",        "max"),
        generations    = ("generation",    "max"),
    )
    .reset_index()
)

# Exportar tabla resumen
summary_path = OUT / "summary.csv"
final.to_csv(summary_path, index=False)
print(f"Resumen exportado → {summary_path}")

# ── 3. Mejor valor por imagen y eje ──────────────────────────────────────────
best_idx = final.groupby(["image", "axis"])["best_fitness"].idxmax()
best = final.loc[best_idx, ["image", "axis", "value", "best_fitness", "total_time_s"]].sort_values(
    ["image", "axis"]
)
print("=== Mejor valor por imagen y eje ===")
print(best.to_string(index=False))
best.to_csv(OUT / "best_per_axis.csv", index=False)
print()

# ── 4. Bar charts: fitness final por valor, agrupado por imagen ───────────────
for axis in AXES:
    sub = final[final["axis"] == axis]
    if sub.empty:
        continue

    values = sorted(sub["value"].unique())
    n_vals = len(values)
    x = np.arange(len(images))
    width = 0.8 / n_vals

    fig, ax = plt.subplots(figsize=(max(9, len(images) * 2.5), 5))
    for i, val in enumerate(values):
        heights = []
        for img in images:
            row = sub[(sub["image"] == img) & (sub["value"] == val)]
            heights.append(float(row["best_fitness"].values[0]) if not row.empty else 0.0)
        offset = i * width - 0.4 + width / 2
        bars = ax.bar(x + offset, heights, width, label=val, color=PALETTE[i % len(PALETTE)])
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(images, rotation=15, ha="right")
    ax.set_ylabel("Best Fitness (última generación)")
    ax.set_title(f"Impacto de '{axis}' sobre fitness final")
    ax.legend(title=axis, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT / f"bar_{axis}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Guardado: {path.name}")

# ── 5. Curvas de convergencia por eje ─────────────────────────────────────────
for axis in AXES:
    sub = df[df["axis"] == axis]
    if sub.empty:
        continue

    values = sorted(sub["value"].unique())
    ncols = min(len(images), 2)
    nrows = (len(images) + ncols - 1) // ncols

    fig, axes_grid = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False,
    )

    for idx, image in enumerate(images):
        ax = axes_grid[idx // ncols][idx % ncols]
        for i, val in enumerate(values):
            mask = (sub["image"] == image) & (sub["value"] == val)
            data = sub[mask].sort_values("generation")
            if not data.empty:
                ax.plot(
                    data["generation"], data["best_fitness"],
                    label=val, linewidth=1.4,
                    color=PALETTE[i % len(PALETTE)],
                )
        ax.set_title(image, fontsize=11)
        ax.set_xlabel("Generación")
        ax.set_ylabel("Best Fitness")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    for idx in range(len(images), nrows * ncols):
        axes_grid[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Convergencia por '{axis}'", fontsize=13)
    fig.tight_layout()
    path = OUT / f"convergence_{axis}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {path.name}")

# ── 6. Heatmap por imagen: todos los ejes juntos ──────────────────────────────
for image in images:
    rows = []
    for axis in AXES:
        sub = final[(final["image"] == image) & (final["axis"] == axis)].sort_values("value")
        for _, r in sub.iterrows():
            rows.append({
                "label":        f"{axis} / {r['value']}",
                "best_fitness": r["best_fitness"],
            })
    if not rows:
        continue

    heat_df = pd.DataFrame(rows)
    n = len(heat_df)

    fig, ax = plt.subplots(figsize=(3.5, max(5, n * 0.38)))
    data_vals = heat_df["best_fitness"].values.reshape(-1, 1)
    im = ax.imshow(data_vals, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks(range(n))
    ax.set_yticklabels(heat_df["label"], fontsize=8)
    for i, v in enumerate(heat_df["best_fitness"]):
        ax.text(0, i, f"{v:.4f}", ha="center", va="center", fontsize=8.5,
                color="black" if v > 0.4 else "white")
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02)
    ax.set_title(f"Fitness final — {image}", fontsize=11)
    fig.tight_layout()
    path = OUT / f"heatmap_{image}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Guardado: {path.name}")

# ── 7. Ranking global: top-10 configs por fitness final promedio (cross-imagen)─
rank = (
    final.groupby(["axis", "value"])["best_fitness"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"best_fitness": "mean_best_fitness"})
)
rank.to_csv(OUT / "global_ranking.csv", index=False)
print(f"\n=== Top-10 valores por fitness promedio cross-imagen ===")
print(rank.head(10).to_string(index=False))

print(f"\nAnálisis completo en: {OUT}/")
