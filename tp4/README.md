# TP4 — Aprendizaje No Supervisado

Implementaciones de Red de Kohonen (SOM) y Modelo de Oja sobre el dataset de países europeos.

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

## Ejercicio 2.1 — Modelo de Hopfield

Memoria asociativa que recupera patrones de letras 5×5 a partir de versiones ruidosas.
El abecedario completo (A–Z) está definido en `hopfield/alphabet.py` como matrices
de `{+1, -1}`.

### Uso

```bash
# parte (a) recuperación con ruido + (b) estado espúreo
make hopfield

# graficar un rango cualquiera de letras en grilla 5x5
make hopfield-alphabet LET_START=C LET_END=H
uv run python -m hopfield.plot_letters --start a --end z --output output/hopfield/abecedario.png

# análisis de ortogonalidad para C(26, k) combinaciones
make hopfield-orthogonality HOPFIELD_K=4

# cómo varía la métrica de recall según la cantidad de patrones almacenados
make hopfield-capacity
```

### Configuración (`configs/hopfield.json`)

| Parámetro | Descripción | Default |
|---|---|---|
| `letters` | Lista de letras a almacenar. Si es `null`, se elige el subconjunto de tamaño `k` más ortogonal | `null` |
| `k` | Tamaño del subconjunto a elegir cuando `letters` es `null` | `4` |
| `noise` | Fracción de bits invertidos en la parte (a) | `0.15` |
| `high_noise` | Fracción de bits invertidos en la parte (b) | `0.40` |
| `mode` | `sync` o `async` | `sync` |
| `max_steps` | Máximo de iteraciones por consulta | `50` |
| `spurious_attempts` | Intentos para encontrar un estado espúreo | `10` |
| `seed` | Semilla aleatoria | `42` |
| `output_dir` | Directorio de salida | `output/hopfield` |

### Salida

| Archivo | Descripción |
|---|---|
| `recall_a_<L>.png` | Evolución paso a paso para cada letra almacenada (parte a) |
| `recall_b_try*_<L>.png` | Intentos con ruido alto buscando estado espúreo (parte b) |
| `orthogonality/dot_heatmap.png` | Matriz 26×26 de \|⟨xi,xj⟩\| entre todas las letras |
| `orthogonality/combos_k<k>.csv` | Todas las combinaciones C(26,k) con su `max_abs_dot` y `mean_abs_dot` |
| `orthogonality/top_bottom_k<k>.png` | Mejores y peores subconjuntos según ortogonalidad |
| `capacity/accuracy_vs_n.png` | Recall accuracy vs N de patrones almacenados |
| `capacity/spurious_vs_n.png` | Tasa de estados espúreos vs N |
| `capacity/hamming_vs_n.png` | Hamming promedio al patrón original vs N |
| `capacity/capacity.csv` | Métricas para cada combinación de modo / N / ruido |

### Metodología

1. **Almacenamiento**: regla de Hebb `W = (1/N) Σ ξᵖ ξᵖᵀ` con diagonal nula.
2. **Recuperación síncrona**: `s(t+1) = sgn(W s(t))`. Si dos estados consecutivos coinciden → convergió.
   Se detecta también ciclo de período 2 (oscilación típica del modo síncrono).
3. **Recuperación asíncrona**: cada neurona se actualiza una a una en orden aleatorio.
4. **Energía**: `E = -½ sᵀ W s` (monótonamente no creciente en modo asíncrono).
5. **Estado espúreo**: punto fijo distinto a cualquier patrón almacenado (y a su complemento).
6. **Elección del subconjunto**: por defecto se barren las C(26,4) = 14 950 combinaciones y se
   elige la que minimiza `max |⟨xi,xj⟩|` (más ortogonal). Para 5×5 = 25 dimensiones, el set
   óptimo encontrado es **`GRTV`** con `max|⟨·,·⟩| = 1`.

---

## Tests

```bash
uv run pytest
```

Cubre inicialización del SOM, BMU, funciones de vecindad, decaimiento, entrenamiento, U-matrix, validación del config, generación de gráficos, regla de Oja, y abecedario/Hopfield (86 tests).
