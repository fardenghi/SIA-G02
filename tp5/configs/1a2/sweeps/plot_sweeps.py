"""Agrega los resultados de los sweeps OAT y grafica el impacto de cada variable.

Lee `manifest.json` (qué se varió en cada config) y, por cada dimensión, junta los
`metrics_restarts.csv` para extraer el error de píxeles a lo largo de los restarts:
  - best  = min(max_pixel_error)   (mejor caso entre los restarts)
  - mean  = mean(max_pixel_error)  (robustez; con barra de ±std)

Produce un PNG por dimensión en out/1a2/sweeps/<dim>/impact_<dim>.png y un CSV
resumen out/1a2/sweeps/summary.csv. Salta las dimensiones aún no corridas.

    # primero correr los configs (ver README), luego:
    uv run python configs/1a2/sweeps/plot_sweeps.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "out" / "1a2" / "sweeps"


def collect() -> tuple[pd.DataFrame, list[dict]]:
    manifest = json.loads((HERE / "manifest.json").read_text())
    rows = []
    for sw in manifest["sweeps"]:
        for entry in sw["configs"]:
            csv = REPO / entry["metrics_restarts"]
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            rows.append({
                "dim": sw["dim"],
                "param": sw["param"],
                "kind": sw["kind"],
                "label": entry["label"],
                "value": entry["value"] if sw["kind"] == "numeric" else entry["label"],
                "is_baseline": entry["is_baseline"],
                "best_max_pix": int(df["max_pixel_error"].min()),
                "mean_max_pix": float(df["max_pixel_error"].mean()),
                "std_max_pix": float(df["max_pixel_error"].std(ddof=0)),
                "success_le1": int((df["max_pixel_error"] <= 1).sum()),
                "n_restarts": int(len(df)),
            })
    return pd.DataFrame(rows), manifest["sweeps"]


def plot_dim(sw: dict, df: pd.DataFrame) -> Path | None:
    sub = df[df["dim"] == sw["dim"]].copy()
    if sub.empty:
        return None

    numeric = sw["kind"] == "numeric"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    if numeric:
        # Graficamos sobre el valor REAL (no por índice) para no distorsionar puntos
        # intermedios como lr=5e-4. Escala log cuando el sweep abarca muchos órdenes
        # de magnitud (lr 1e-9..5e-2) y lineal si no (epochs 10000..30000).
        sub = sub.sort_values("value")
        xvals = sub["value"].to_numpy(dtype=float)
        labels = sub["label"].tolist()
        use_log = bool(xvals.min() > 0 and xvals.max() / xvals.min() >= 100)

        ax.plot(xvals, sub["mean_max_pix"], "-o", color="#1f77b4", label="media (restarts)")
        ax.fill_between(
            xvals,
            sub["mean_max_pix"] - sub["std_max_pix"],
            sub["mean_max_pix"] + sub["std_max_pix"],
            color="#1f77b4", alpha=0.15,
        )
        ax.plot(xvals, sub["best_max_pix"], "--s", color="#2ca02c", label="mejor (min)")
        if use_log:
            ax.set_xscale("log")
        ax.set_xticks(xvals)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.minorticks_off()
        base_x = sw["baseline_value"]
    else:
        xpos = list(range(len(sub)))
        labels = sub["label"].tolist()
        ax.bar([i - 0.18 for i in xpos], sub["mean_max_pix"], width=0.36,
               yerr=sub["std_max_pix"], color="#1f77b4", label="media (restarts)",
               capsize=3)
        ax.bar([i + 0.18 for i in xpos], sub["best_max_pix"], width=0.36,
               color="#2ca02c", label="mejor (min)")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        base_x = next((i for i, b in enumerate(sub["is_baseline"]) if b), None)

    # Resalta dónde cae el baseline (sobre el valor real si es numérico, o el índice).
    if base_x is not None:
        ax.axvline(base_x, color="#d62728", ls=":", alpha=0.6, lw=1)
        ax.annotate("baseline", (base_x, ax.get_ylim()[1]), color="#d62728",
                    fontsize=8, ha="center", va="top")

    ax.set_xlabel(sw["dim"] + (" (escala log)" if numeric and use_log else ""))
    ax.set_ylabel("error de píxeles (max sobre 32 patrones)")
    ax.set_title(f"1a2 — impacto de '{sw['dim']}' sobre el error de píxeles")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    dst_dir = OUT / sw["dim"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"impact_{sw['dim']}.png"
    fig.savefig(dst, dpi=130)
    plt.close(fig)
    return dst


def main() -> None:
    df, sweeps = collect()
    if df.empty:
        print("No hay metrics_restarts.csv todavía. Corré los configs primero "
              "(ver configs/1a2/sweeps/README.md).")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "summary.csv", index=False)
    print(f"Resumen: {OUT / 'summary.csv'} ({len(df)} configs con resultados)")

    for sw in sweeps:
        dst = plot_dim(sw, df)
        if dst:
            n = (df["dim"] == sw["dim"]).sum()
            print(f"  {sw['dim']:18s} -> {dst}  ({n} puntos)")
        else:
            print(f"  {sw['dim']:18s} -> (sin resultados aún)")


if __name__ == "__main__":
    main()
