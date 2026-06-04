# 1a2 — Búsqueda de configuraciones (arquitectura × optimizador × init × loss)

Esta carpeta documenta la **progresión de configuraciones probadas** para el punto 1a2,
hasta encontrar las que reconstruyen los 32 caracteres con el **error de píxeles más
bajo**. El cuello latente se mantiene fijo en **2 dimensiones** (objetivo de 1a); lo que
se varía es activación, pérdida, inicialización, profundidad y optimizador.

Cada config se corre con `restarts = 12`, `epochs = 15000`, `seed = 42`, `stop_at = null`
(se ejecutan los 12 restarts para poder medir la **robustez**, no sólo el mejor caso).
Las métricas reportadas salen de `out/1a2/<name>/metrics_restarts.csv`.

```bash
uv run autoencoder --config configs/1a2/04_deep_xavier_bce_base.json
```

## Progresión y resultados

| # | Config | Arquitectura | act / loss / init / optim | best `max_pix` | éxito (≤1px) | `max_pix` medio |
|---|--------|--------------|---------------------------|----------------|--------------|-----------------|
| 01 | `shallow_normal_mse_relu` | `35-20-2`         | relu / mse / normal / adam        | **24** | 0/12  | 28.83 |
| 02 | `shallow_xavier_bce`      | `35-20-2`         | tanh / bce / xavier_normal / adam | **0**  | 12/12 | 0.00  |
| 03 | `deep_naive_init`         | `35-25-15-8-2`    | tanh / bce / normal / adam        | **0**  | 9/12  | 0.83  |
| 04 | `deep_xavier_bce_base`    | `35-25-15-8-2`    | tanh / bce / xavier_normal / adam | **0**  | 12/12 | 0.08  |
| 05 | `deeper_xavier_bce`       | `35-30-20-12-6-2` | tanh / bce / xavier_normal / adam | **0**  | 10/12 | 0.42  |
| 06 | `deep_xavier_bce_lbfgs`   | `35-25-15-8-2`    | tanh / bce / xavier_normal / lbfgs| **0**  | 6/12  | 4.00  |

## Lectura de la búsqueda

1. **01 — punto de partida ingenuo:** `relu` + `mse` + init `normal` en una red chata
   **no logra aprender** (mejor reconstrucción 24 px de error, 0/12 restarts cumplen el
   objetivo). Combinación de salida/pérdida/init incoherente → satura.
2. **02 — corregir activación + pérdida + init:** pasar a `tanh`/`sigmoid` + `bce` +
   `xavier_normal` hace que **incluso una red chata** `35-20-2` reconstruya 32/32 en
   **todos** los restarts (12/12). Es el salto cualitativo de la búsqueda.
3. **03 — el efecto de la init:** la misma red profunda con init `normal` llega a 0 pero
   es **menos robusta** (9/12); la inicialización escalada importa para la estabilidad.
4. **04 — config base (ganadora):** red profunda `35-25-15-8-2` con `xavier_normal` →
   **12/12** y `max_pix` medio 0.08, el más bajo. Es la configuración de referencia.
5. **05 — más profundidad no ayuda:** `35-30-20-12-6-2` también llega a 0 pero es algo
   menos robusta (10/12) y tiene más parámetros: profundidad extra sin ganancia.
6. **06 — el optimizador:** mismo diseño con **L-BFGS-B** alcanza 0 pero cae en mínimos
   locales mucho más seguido (6/12, `max_pix` medio 4.0). Adam es claramente más estable
   para este problema.

**Conclusión:** las configuraciones de **menor error** son `02_shallow_xavier_bce` y
`04_deep_xavier_bce_base` (ambas 12/12, `max_pix` medio ≈ 0) — es decir
`tanh`/`sigmoid` + `bce` + `xavier_normal` + `adam`. El driver dominante del error no es
la profundidad sino la **coherencia activación/pérdida/init** y la elección de
**optimizador**.
