# TP4 — Aprendizaje No Supervisado

Implementaciones de Red de Kohonen (SOM), Modelo de Oja y Red de Hopfield sobre datasets europeos y patrones de letras.

## Dataset

`data/europe.csv` — 28 países europeos con 7 features numéricas:

| Feature | Descripción |
|---|---|
| Area | Superficie en km² |
| GDP | PBI per cápita (USD) |
| Inflation | Tasa de inflación (%) |
| Life.expect | Esperanza de vida (años) |
| Military | Gasto militar (% del PBI) |
| Pop.growth | Crecimiento poblacional (%) |
| Unemployment | Tasa de desempleo (%) |

---

## Ejercicio 1.1 — Red de Kohonen

Implementación propia (numpy puro) de un Self-Organizing Map (SOM) para agrupar países con características geopolíticas, económicas y sociales similares.

### Uso

```bash
uv run python kohonen_europe.py --config configs/kohonen_europe.json
```

### Configuración (`configs/kohonen_europe.json`)

| Parámetro | Descripción | Default |
|---|---|---|
| `grid_rows` / `grid_cols` | Dimensiones de la grilla | `5` / `5` |
| `lr` | Tasa de aprendizaje inicial | `0.5` |
| `lr_decay` | Decaimiento del lr: `exponential` \| `linear` | `exponential` |
| `radius` | Radio de vecindad inicial | `3.0` |
| `radius_decay` | Decaimiento del radio: `exponential` \| `linear` | `exponential` |
| `neighborhood_fn` | Función de vecindad: `gaussian` \| `bubble` | `gaussian` |
| `epochs` | Épocas de entrenamiento | `1000` |
| `seed` | Semilla aleatoria | `42` |
| `output_dir` | Directorio de salida para gráficos | `output/kohonen` |

### Salida

Imprime la asignación de países a neuronas y genera tres gráficos en `output_dir/`:

| Archivo | Descripción |
|---|---|
| `country_map.png` | Grilla con los nombres de los países en su neurona BMU |
| `u_matrix.png` | Distancias promedio entre neuronas vecinas (U-Matrix) |
| `hit_map.png` | Cantidad de países asignados a cada neurona |

### Metodología

1. Los datos se estandarizan con `StandardScaler` (media 0, varianza 1).
2. Los pesos se inicializan con ruido gaussiano.
3. En cada época se presenta cada muestra en orden aleatorio; se busca la neurona ganadora (BMU) por distancia euclidiana mínima y se actualizan los pesos vecinos según la función de vecindad y la tasa de aprendizaje vigentes.
4. El radio y la tasa de aprendizaje decaen a lo largo del entrenamiento (exponencial o lineal).
5. La U-Matrix se calcula como la distancia promedio de cada neurona a sus 4 vecinos directos.

---

## Ejercicio 1.2 — PCA con scikit-learn (referencia)

Análisis PCA sobre el mismo dataset usando scikit-learn.

### Uso

```bash
uv run python pca_europe.py [--n-components N]
```

**Argumentos:**
- `--n-components` — número de componentes principales a calcular (default: `1`)

---

## Análisis cruzado (Kohonen + PCA + Oja)

Scripts en `analysis/` que se apoyan en los modelos anteriores.

```bash
make compare                              # Validación cruzada PCA vs Kohonen (Secciones 1.1, 1.2, 2.1)
make compare MIDDLE_THRESHOLD=0.8         # umbral personalizado para países "promedio"
make standardization                      # Experimento 2x2 estandarización vs datos crudos
make analysis                             # corre compare + standardization
```

Salidas en `output/compare/` y `output/standardization/`.

---

## Ejercicio 2.1 — Modelo de Hopfield

Memoria asociativa que recupera patrones de letras 5×5 a partir de versiones ruidosas.
El abecedario completo (A–Z) está definido en `hopfield/alphabet.py` como matrices de `{+1, −1}`.

### Uso

```bash
# parte (a) recuperación con ruido + (b) estado espúreo (elige el subset más ortogonal)
make hopfield

# graficar un rango cualquiera de letras en grilla 5x5
make hopfield-alphabet LET_START=C LET_END=H
uv run python -m hopfield.plot_letters --start a --end z --output output/hopfield/abecedario.png

# análisis de ortogonalidad para C(26, k) combinaciones
make hopfield-orthogonality HOPFIELD_K=4

# cómo varía la métrica de recall según la cantidad de patrones almacenados
make hopfield-capacity
make hopfield-capacity-adaptive           # idem comparando N fijo (5x5) vs N adaptativo

# almacenar las 26 letras con escalado adaptativo (k=3, 15x15 = 225 neuronas)
make hopfield-full-alphabet
```

### Configuración (`configs/hopfield.json`)

| Parámetro | Descripción | Default |
|---|---|---|
| `letters` | Letras a almacenar. Si es `null`, se elige el subconjunto de tamaño `k` más ortogonal | `null` |
| `k` | Tamaño del subconjunto a elegir cuando `letters` es `null` | `4` |
| `noise` | Fracción de bits invertidos en la parte (a) | `0.15` |
| `high_noise` | Fracción de bits invertidos en la parte (b) | `0.40` |
| `mode` | `sync` o `async` | `sync` |
| `max_steps` | Máximo de iteraciones por consulta | `50` |
| `spurious_attempts` | Intentos para encontrar un estado espúreo | `10` |
| `noise_levels_analysis` | Barrido de ruido para `recovery_rate.png` | `[0.0 … 0.5]` |
| `n_trials` | Ensayos por nivel de ruido | `50` |
| `seed` | Semilla aleatoria | `42` |
| `output_dir` | Directorio de salida | `output/hopfield` |

### Salida — modo estándar (`output/hopfield/`)

| Archivo | Descripción |
|---|---|
| `recall_a_<L>.png` | Evolución paso a paso para cada letra almacenada (parte a) |
| `recall_b_try*_<L>.png` | Intentos con ruido alto buscando estado espúreo (parte b) |
| `energy_a_<L>.png` | Energía H vs iteración durante la convergencia de L |
| `crosstalk.png` | Correlación normalizada `xi·xj / N` entre patrones almacenados |
| `recovery_rate.png` | Tasa de recuperación vs nivel de ruido, una curva por letra |
| `orthogonality/dot_heatmap.png` | Matriz 26×26 de \|⟨xi,xj⟩\| entre todas las letras |
| `orthogonality/combos_k<k>.csv` | Todas las combinaciones C(26,k) con su `max_abs_dot` y `mean_abs_dot` |
| `orthogonality/top_bottom_k<k>.png` | Mejores y peores subconjuntos según ortogonalidad |
| `capacity/accuracy_vs_n.png` | Recall accuracy vs N de patrones almacenados |
| `capacity/spurious_vs_n.png` | Tasa de estados espúreos vs N |
| `capacity/hamming_vs_n.png` | Hamming promedio al patrón original vs N |
| `capacity/fixed_vs_adaptive.png` | Recall N fijo vs N adaptativo (con `--adaptive`) |
| `capacity/capacity.csv` | Métricas para cada combinación de modo / N / ruido |
| `alphabet/crosstalk_alphabet.png` | Crosstalk de las 26 letras (modo `--alphabet`) |
| `alphabet/recovery_rate_alphabet.png` | Curvas de recall por letra (modo `--alphabet`) |

### Metodología

**Regla de Hebb:**

$$W_{ij} = \frac{1}{N} \sum_{\mu} \xi_i^\mu \xi_j^\mu, \quad W_{ii} = 0$$

**Recuperación síncrona:** `s(t+1) = sgn(W s(t))`. Se detecta tanto la convergencia (estado fijo) como el ciclo de período 2 típico del modo síncrono.

**Recuperación asíncrona:** cada neurona se actualiza una a una en orden aleatorio.

**Energía:** `E = −½ sᵀ W s`, monótonamente no creciente bajo update asíncrono.

**Estado espúreo:** punto fijo distinto a todos los patrones almacenados (y a sus complementos).

**Elección del subconjunto:** se barren las **C(26, 4) = 14 950** combinaciones y se elige la que minimiza `max |⟨xi, xj⟩|`. Para 5×5 = 25 dimensiones, el óptimo es **`GRTV`** con `max|⟨·,·⟩| = 1` (prácticamente ortogonal).

### Escalado adaptativo del alfabeto completo

Con p=26 y N=25 la red colapsa (p/N ≈ 1.04 ≫ 0.138). Vía `np.kron` se replica cada bit en un bloque k×k y la grilla pasa a (5k)×(5k), con k = ⌈√(p / (0.138·25))⌉:

| p | k | N = (5k)² | p/N |
|---|---|---|---|
| ≤ 3 | 1 | 25 | ≤ 0.12 |
| 4–13 | 2 | 100 | ≤ 0.13 |
| 14–31 | 3 | 225 | ≤ 0.116 |

Con k=3 (225 neuronas) la red queda dentro del límite teórico; las letras visualmente similares (A/H, C/G/O, E/F) siguen interfiriéndose por **crosstalk**, no por capacidad.

---

## Tests

```bash
uv run pytest
```

Cubre red de Hopfield (pesos de Hebb verificados contra el ejemplo de las diapositivas, sync + async, ciclos, complemento como atractor, escalado adaptativo, ortogonalidad), SOM, regla de Oja, PCA y config (>100 tests).
