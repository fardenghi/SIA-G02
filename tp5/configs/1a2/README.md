# 1a2 — Progresión de configuraciones (arquitectura × init × optimizador)

Esta carpeta documenta la **progresión de configuraciones** del punto 1a2 hasta la que
reconstruye los 32 caracteres con el **error de píxeles más bajo y de forma más robusta**.

El punto de partida **no** es un contraejemplo: la combinación pérdida/activación ya queda
fijada **por coherencia teórica** (ver más abajo), así que la búsqueda se concentra en lo que
realmente mueve el resultado con el cuello latente fijo en **2 dimensiones**: **profundidad,
inicialización y optimizador**.

## Base fija (igual en todas las configs)

| Parámetro | Valor | Motivo |
|---|---|---|
| `loss` | **`bce`** | píxeles binarios {0,1} → verosimilitud Bernoulli; gradiente que no se desvanece |
| `output_activation` | **`sigmoid`** | salida en (0,1) = probabilidad de píxel; **obligatorio con BCE** |
| `activation` (oculta) | **`tanh`** | acotada y centrada; coherente con salida sigmoide |
| `latent_activation` | **`linear`** | cuello sin acotar (default teórico del AE; conexión con PCA) |

> La elección de `bce` + `sigmoid` se justifica con el **gradiente** (no hace falta una corrida
> para mostrarlo): con salida sigmoide, `∂L/∂z = (ŷ−y)·σ'(z)` para MSE — y `σ'(z)=σ(z)(1−σ(z))`
> se anula al saturar, así que el gradiente desaparece — mientras que BCE cancela ese factor y
> queda `∂L/∂z = ŷ−y`. Por eso toda la progresión usa BCE/sigmoid de entrada.

Cada config corre con `epochs = 15000`, `seed = 42`, `stop_at = null` (se ejecutan todos los
restarts para medir **robustez**, no sólo el mejor caso). Las métricas salen de
`out/1a2/<name>/metrics_restarts.csv`.

```bash
uv run autoencoder --config configs/1a2/02_deep.json
```

## Progresión y resultados

| # | Config | Qué varía | Arquitectura | init / optim | best `max_px` | éxito (≤1px) | `max_px` medio |
|---|--------|-----------|--------------|--------------|---------------|--------------|----------------|
| 01 | `base_shallow`     | punto de partida coherente   | `35-20-2`         | xavier / adam        | **0** | 11/12 | 0.33 |
| 02 | `deep`             | + profundidad                | `35-25-15-8-2`    | xavier / adam        | **0** | 11/12 | 0.25 |
| 03 | `deeper`           | profundidad extra            | `35-30-20-12-6-2` | xavier / adam        | **0** | 11/12 | 0.58 |
| 04 | `deep_normal_init` | efecto de la init            | `35-25-15-8-2`    | **normal** / adam    | **0** | 11/12 | 0.17 |
| 05 | `deep_lbfgs`       | efecto del optimizador       | `35-25-15-8-2`    | xavier / **lbfgs**   | **0** | 9/12  | 2.08 |
| 06 | `best`             | mejor combinación (×20+cos.) | `35-25-15-8-2`    | xavier / adam        | **0** | 19/20 | 0.15 |

## Lectura de la búsqueda

1. **01 — base chata ya resuelve:** fijada la coherencia, incluso `35-20-2` reconstruye los
   32/32 caracteres (0 px) en casi todos los restarts. El problema base es *fácil* una vez
   elegidas bien pérdida/activación.
2. **02 — profundidad ayuda al promedio:** la red profunda `35-25-15-8-2` baja el `max_px`
   medio (0.25 vs 0.33) manteniendo 0 px y 11/12.
3. **03 — más profundidad no compensa:** `35-30-20-12-6-2` llega a 0 px pero con `max_px`
   medio más alto (0.58) y más parámetros: profundidad extra sin ganancia (rendimientos
   decrecientes).
4. **04 — la init pesa poco acá:** la misma red con init `normal` rinde igual o mejor
   (0.17) que con `xavier_normal`. Con el cuello **lineal** la init escalada deja de ser
   determinante para la robustez.
5. **05 — el optimizador sí importa:** con **L-BFGS** la red alcanza 0 px pero cae en mínimos
   locales más seguido (9/12, `max_px` medio 2.08). **Adam es claramente más estable.**
6. **06 — mejor configuración:** red profunda + **Adam con schedule coseno** y más restarts
   (20) da la corrida más robusta: 19/20 (95%) y el `max_px` medio más bajo (0.15).

**Conclusión:** fijadas pérdida/activación por coherencia, el problema se resuelve siempre
(0 px); el driver del resto es la **robustez**, gobernada por el **optimizador** (Adam ≫ L-BFGS)
y, en menor medida, por una profundidad moderada. La profundidad excesiva y la init escalada
aportan poco con el cuello lineal.
