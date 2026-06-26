# 1a2 — Ablación OAT (one-at-a-time) del error de píxeles

Esta carpeta complementa la [progresión 01–06](../README.md). Mientras esa tabla
muestra *configuraciones completas* ya buenas, acá el objetivo es **aislar el impacto
de cada hiperparámetro**: se parte de un **baseline débil** (el punto de partida
ingenuo, ~24 px) y por cada variable se genera una serie de configs que cambian **solo
esa dimensión**, dejando todo lo demás fijo. Así cada gráfico cuenta una historia de
**mejora**: "moviendo *esta* perilla hacia la elección correcta, el error baja de X a Y".

## Baseline (punto de partida ingenuo)

| arquitectura | act | latente | salida | init | optim | loss | epochs | lr | restarts | stop_at |
|---|---|---|---|---|---|---|---|---|---|---|
| `35-20-2` | `relu` | `linear` | `sigmoid` | `normal` | `adam` | `mse` | `15000` | `1e-3` | `12` | `null` |

El **cuello se fija en `linear`** (default teórico) y se mantiene constante en todos los
sweeps salvo el de `latent_activation`. Así el sweep de `activation` cambia solo la
activación de **ocultas** (sin arrastrar el latente) y cada ablación toca una sola variable.

`stop_at = null` ⇒ se corren los **12 restarts completos** (sin corte temprano) para
medir tanto el **mejor caso** (min) como la **robustez** (media ± desvío sobre restarts).

## Dimensiones barridas (1 variable a la vez)

> La fuente de verdad de los valores es `generate_sweeps.py` (esta tabla la resume).

| dimensión | valores | #configs |
|---|---|---|
| `epochs`            | 10000, **15000**, 20000, 25000, 30000 | 5 |
| `lr`                | 1e-9 … 1e-4, 5e-4, **1e-3**, 5e-3, 1e-2, 5e-2 | 11 |
| `optimizer`         | **adam**, lbfgs | 2 |
| `loss`              | **mse**, bce | 2 |
| `activation`        | **relu**, tanh, sigmoid | 3 |
| `output_activation` | **sigmoid**, tanh, linear | 3 |
| `latent_activation` | **linear**, tanh, sigmoid, relu | 4 |
| `init`              | **normal**, uniform, xavier_uniform, xavier_normal, he_uniform, he_normal | 6 |
| `architecture`      | 35-2, **35-20-2**, 35-16-8-2, 35-25-15-8-2, 35-30-20-12-6-2, 35-30-2 | 6 |

(en **negrita** el valor del baseline, que aparece como punto de anclaje en cada serie). Total: **42 configs**.

`latent_activation` controla la activación del **cuello** (capa latente), desacoplada de
la de ocultas. `linear` = cuello sin no-linealidad (default teórico de un AE: código sin
acotar, conexión con PCA, consistente con la media de un VAE), y es el valor del baseline.
Este sweep muestra que **desviarse a `tanh`/`sigmoid`/`relu` degrada** (relu colapsa el
latente a ≥0, sigmoid lo descentra). Si no se especifica en un config, el cuello hereda
`activation` (retrocompatible).

Los sweeps numéricos se grafican sobre el valor real: `lr` en **escala log** (abarca muchos
órdenes de magnitud, con puntos intermedios como `5e-4`) y `epochs` en lineal.

## Cómo correr

```bash
# Regenerar los JSON (si editás generate_sweeps.py)
uv run python configs/1a2/sweeps/generate_sweeps.py

# Correr todo y graficar (puede tardar; son 42 × 12 restarts)
bash configs/1a2/sweeps/run_all.sh

# O sólo algunas dimensiones
bash configs/1a2/sweeps/run_all.sh lr activation init

# O un único config
uv run autoencoder --config configs/1a2/sweeps/lr/1a2_sweep_lr_1e-03.json
```

## Salidas y gráficos

Cada config escribe `out/1a2/sweeps/<dim>/<name>/metrics{,_restarts}.csv`.
El agregador junta los `_restarts.csv` por dimensión y produce un PNG por variable:

```bash
uv run python configs/1a2/sweeps/plot_sweeps.py
```

- `out/1a2/sweeps/<dim>/impact_<dim>.png` — error de píxeles (mejor + media±std) vs. la variable, con el baseline marcado.
- `out/1a2/sweeps/summary.csv` — tabla larga con `best_max_pix`, `mean_max_pix`, `std`, `success_le1` por config (útil para armar tus propios gráficos).

El script salta las dimensiones que todavía no corriste, así podés graficar de a poco.
