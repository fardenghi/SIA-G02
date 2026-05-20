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

---

## Ejercicio 2 — Red de Hopfield

Implementación propia de una red de Hopfield para almacenar y recuperar patrones de letras sobre una grilla bipolar (+1/−1).

### Uso

```bash
# Análisis con 4 letras (Z, E, N, T) — configuración por defecto
uv run python hopfield_letters.py --config configs/hopfield.json

# Experimento con el alfabeto completo (A–Z) con grilla adaptativa
uv run python hopfield_letters.py --alphabet
```

### Configuración (`configs/hopfield.json`)

| Parámetro | Descripción | Default |
|---|---|---|
| `letters` | Letras a almacenar | `["Z","E","N","T"]` |
| `noise_level` | Fracción de bits invertidos para recuperación | `0.2` |
| `max_iter` | Iteraciones máximas de convergencia | `20` |
| `seed` | Semilla aleatoria | `42` |
| `noise_levels_analysis` | Barrido de ruido para análisis de tasa | `[0.0 … 0.5]` |
| `n_trials` | Ensayos por nivel de ruido | `100` |
| `spurious_noise_level` | Ruido para buscar estados espúreos | `0.5` |

### Salida — modo estándar (`output/hopfield/`)

| Archivo | Descripción |
|---|---|
| `recovery_<X>.png` | Recuperación paso a paso de la letra X desde versión ruidosa |
| `energy_<X>.png` | Evolución de la energía H durante la convergencia de X |
| `recovery_rate.png` | Tasa de recuperación vs nivel de ruido por letra |
| `crosstalk.png` | Matriz de correlación normalizada entre patrones almacenados |
| `spurious_state.png` | Estado espúreo identificado vs patrones almacenados |

### Salida — modo alfabeto (`output/hopfield/alphabet/`)

| Archivo | Descripción |
|---|---|
| `crosstalk_alphabet.png` | Matriz de correlación para las 26 letras |
| `capacity_experiment.png` | Tasa de recuperación vs p: N fijo (5×5) vs N adaptativo |

### Metodología

**Entrenamiento (regla de Hebb):**

$$W_{ij} = \frac{1}{N} \sum_{\mu} \xi_i^\mu \xi_j^\mu, \quad W_{ii} = 0$$

Los pesos se calculan solo sobre el triángulo superior y se copian por simetría.

**Actualización síncrona:**

$$S_i(t+1) = \text{sign}\!\left(\sum_j W_{ij} S_j(t)\right)$$

La red converge cuando ningún nodo cambia entre iteraciones.

**Función de energía:**

$$H = -\sum_{j>i} W_{ij} S_i S_j$$

La energía es no-creciente a lo largo de la dinámica.

### Análisis (4 letras Z, E, N, T)

- p=4, N=25 → p/N=0.16, ligeramente por encima del límite teórico (≈0.138).
- Alta correlación entre patrones: Z·E=0.44, Z·T=0.36 → crosstalk significativo.
- T se recupera solo ~46% de las veces con 10% de ruido; Z es la más robusta.
- Estado espúreo encontrado con energía −12.48 (igual a N), originado desde perturbaciones de Z.

### Escalado adaptativo del alfabeto completo

Con p=26 letras y N=25 la red colapsa (p/N=1.04 ≫ 0.138). La grilla se puede escalar automáticamente con factor k tal que (5k)² × 0.138 ≥ p:

| p | k | N = (5k)² | p/N |
|---|---|---|---|
| ≤ 3 | 1 | 25 | ≤ 0.12 |
| 4–13 | 2 | 100 | ≤ 0.13 |
| 14–31 | 3 | 225 | ≤ 0.138 |

Con k=3 (15×15=225 neuronas) la red queda dentro del límite de capacidad (p/N=0.116). H, O, B, U, N recuperan bien; las letras visualmente similares (A/H, C/G/O, E/F) siguen siendo afectadas por crosstalk, no por capacidad.

---

## Tests

```bash
uv run pytest
```

Cubre red de Hopfield (pesos, convergencia, energía, patrones, escalado adaptativo), SOM, config y gráficos.
