# TP4 — Aprendizaje No Supervisado

Implementaciones **desde cero** (NumPy puro) de tres modelos de aprendizaje no supervisado:

| Módulo | Algoritmo | Dataset |
|--------|-----------|---------|
| `kohonen/` | Self-Organizing Map (SOM) | `data/europe.csv` (28 países, 7 variables) |
| `oja/` | Regla de Oja (PCA de 1 componente) | `data/europe.csv` (28 países, 7 variables) |
| `hopfield/` | Red de Hopfield (memoria asociativa) | Patrones de letras 5×5 (A–Z) |
| `pca_test/` | PCA con scikit-learn (referencia) | `data/europe.csv` |
| `analysis/` | Análisis cruzado (Kohonen + PCA + Oja) | `data/europe.csv` |

---

## Estructura del proyecto

```
tp4/
├── configs/                    # Archivos de configuración JSON
│   ├── kohonen_europe.json
│   ├── kohonen_radius_study.json
│   ├── oja_europe.json
│   ├── hopfield.json
│   └── hopfield-worst-case.json
├── data/
│   └── europe.csv              # Dataset de países europeos
├── kohonen/
│   ├── som.py                  # Implementación del SOM (numpy puro)
│   ├── kohonen_europe.py       # Script principal: entrena y grafica
│   ├── convergence_analysis.py # Barrido de hiperparámetros (QE vs épocas)
│   └── kohonen_radius_study.py # Estudio TE vs QE para distintos radios
├── oja/
│   ├── oja.py                  # Implementación de la regla de Oja
│   └── oja_europe.py           # Script principal: compara con sklearn PCA
├── hopfield/
│   ├── hopfield.py             # Red de Hopfield (Hebb, sync/async)
│   ├── alphabet.py             # Patrones 5×5 de las 26 letras
│   ├── hopfield_runner.py      # Orquestador principal (recall, espúreo, etc.)
│   ├── orthogonality.py        # Análisis de ortogonalidad C(26, k)
│   ├── capacity.py             # Análisis de capacidad vs cantidad de patrones
│   └── plot_letters.py         # Visualización de letras en grilla
├── pca_test/
│   ├── pca.py                  # PCA con scikit-learn
│   ├── plots.py                # Biplot, varianza explicada, ranking
│   └── boxplots.py             # Boxplots pre/post estandarización
├── analysis/
│   ├── compare.py              # Validación cruzada PCA vs Kohonen vs Oja
│   └── standardization_experiment.py  # Experimento 2×2: estandarizado vs crudo
├── tests/                      # Suite de tests (>100 tests con pytest)
├── Makefile                    # Recetas de ejecución
└── pyproject.toml              # Dependencias del proyecto (uv)
```

---

## Dataset

`data/europe.csv` — 28 países europeos con 7 features numéricas:

| Feature | Descripción |
|---|---|
| `Area` | Superficie en km² |
| `GDP` | PBI per cápita (USD) |
| `Inflation` | Tasa de inflación (%) |
| `Life.expect` | Esperanza de vida (años) |
| `Military` | Gasto militar (% del PBI) |
| `Pop.growth` | Crecimiento poblacional (%) |
| `Unemployment` | Tasa de desempleo (%) |

Todos los modelos que trabajan sobre este dataset **estandarizan los datos** con `StandardScaler` (media 0, varianza 1) antes del entrenamiento.

---

## Requisitos e instalación

### Dependencias

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) (gestor de paquetes recomendado)

| Paquete | Versión mínima |
|---|---|
| `numpy` | ≥ 2.4.4 |
| `pandas` | ≥ 3.0.2 |
| `matplotlib` | ≥ 3.10.9 |
| `scikit-learn` | ≥ 1.8.0 |
| `pytest` | ≥ 9.0.3 |
| `python-pptx` | ≥ 1.0.2 |

### Instalación

```bash
# Clonar y entrar al directorio
cd tp4/

# Instalar dependencias con uv (crea el virtualenv automáticamente)
uv sync
```

> **Alternativa sin uv**: crear un virtualenv estándar e instalar con `pip install -e .`

---

## Cómo correr los experimentos

Todos los comandos se ejecutan **desde el directorio `tp4/`**.

### Vista rápida — todos los make targets

```bash
make help
```

---

## Ejercicio 1.1 — Red de Kohonen (SOM)

Implementación propia (NumPy puro) de un Self-Organizing Map para agrupar países con características geopolíticas, económicas y sociales similares.

### Correr el experimento principal

```bash
# Opción A: Makefile (recomendado)
make kohonen

# Opción B: directo con Python
uv run python -m kohonen.kohonen_europe --config configs/kohonen_europe.json

# Con un config alternativo
make kohonen KOHONEN_CONFIG=configs/mi_config.json
```

### Configuración — `configs/kohonen_europe.json`

```json
{
  "data": "data/europe.csv",
  "grid_rows": 5,
  "grid_cols": 5,
  "lr": 0.3,
  "lr_decay": "exponential",
  "radius": 3.0,
  "radius_decay": "exponential",
  "neighborhood_fn": "gaussian",
  "epochs": 1000,
  "seed": 42,
  "output_dir": "output/kohonen"
}
```

| Parámetro | Descripción | Opciones/Default |
|---|---|---|
| `data` | Path al CSV de entrada | `"data/europe.csv"` |
| `grid_rows` / `grid_cols` | Dimensiones de la grilla | `5` / `5` |
| `lr` | Tasa de aprendizaje inicial | `0.3` |
| `lr_decay` | Decaimiento del learning rate | `"exponential"` \| `"linear"` |
| `radius` | Radio de vecindad inicial | `3.0` |
| `radius_decay` | Decaimiento del radio | `"exponential"` \| `"linear"` |
| `neighborhood_fn` | Función de vecindad | `"gaussian"` \| `"bubble"` |
| `epochs` | Épocas de entrenamiento | `1000` |
| `seed` | Semilla aleatoria | `42` |
| `output_dir` | Directorio de salida para gráficos | `"output/kohonen"` |

### Salida — `output/kohonen/`

Imprime la asignación de países a neuronas en pantalla y genera:

| Archivo | Descripción |
|---|---|
| `country_map.png` | Grilla con los nombres de los países en su neurona BMU |
| `u_matrix.png` | Distancias promedio entre neuronas vecinas (U-Matrix) |
| `hit_map.png` | Cantidad de países asignados a cada neurona |
| `component_planes.png` | Planos de componentes individuales para cada una de las 7 variables |

---

### Análisis de Convergencia (variación de hiperparámetros)

Grafica el Error de Cuantización (distancia promedio a BMU) en función de las épocas, variando radio y learning rate.

```bash
make convergence
# Alternativa directa:
uv run python -m kohonen.convergence_analysis
```

#### Salida — `output/convergence/`

| Archivo | Descripción |
|---|---|
| `convergencia_radio.png` | Curvas comparativas variando Radio ($R = [0.5, 1.5, 3.0, 5.0]$) |
| `convergencia_lr.png` | Curvas comparativas variando Learning Rate ($LR = [0.3, 0.5, 0.7]$) |

---

### Estudio de Radio vs Error Topológico

Compara el Error de Cuantización (QE) y el Error Topológico (TE) para distintos radios iniciales.

```bash
make kohonen-radius-study

# Con config personalizado:
make kohonen-radius-study RADIUS_STUDY_CONFIG=configs/kohonen_radius_study.json
```

#### Configuración — `configs/kohonen_radius_study.json`

```json
{
  "data": "data/europe.csv",
  "grid_rows": 5,
  "grid_cols": 5,
  "lr": 0.3,
  "lr_decay": "exponential",
  "radius_decay": "exponential",
  "neighborhood_fn": "gaussian",
  "epochs": 500,
  "seed": 42,
  "radii": [1.0, 2.0, 3.0, 4.0, 5.0],
  "output_dir": "output/kohonen_radius_study"
}
```

#### Salida — `output/kohonen_radius_study/`

Gráficos de QE y TE por radio barrido.

---

### Metodología del SOM

1. Los datos se estandarizan con `StandardScaler` (media 0, varianza 1).
2. Los pesos se inicializan con una muestra aleatoria de los propios datos.
3. En cada época se presenta cada muestra en orden aleatorio; se busca la neurona ganadora (BMU) por distancia euclidiana mínima.
4. Se actualizan los pesos de todas las neuronas vecinas según la función de vecindad y la tasa de aprendizaje vigentes:
   - **Gaussiana**: `h(d) = exp(-d² / 2σ²)`
   - **Burbuja**: `h(d) = 1 si d ≤ R, 0 si d > R`
5. El radio y la tasa de aprendizaje decaen a lo largo del entrenamiento:
   - **Exponencial**: `v(t) = v₀ · exp(-t / τ)`
   - **Lineal**: `v(t) = v₀ · (1 - t/T)`

---

## Ejercicio 1.2 — PCA con scikit-learn (referencia)

Análisis PCA completo sobre el dataset europeo usando scikit-learn.

```bash
# PCA completo (análisis + plots + boxplots)
make pca

# Solo gráficos de varianza, biplot y ranking
make pca-plots

# Solo boxplots pre/post estandarización
make pca-boxplots

# Directo con Python (especificar número de componentes)
uv run python -m pca_test.pca [--n-components N]
```

### Salida — `pca_test/plots/`

Gráficos de varianza explicada, biplot de los dos primeros componentes, ranking de países, y boxplots comparativos.

---

## Ejercicio 2.1 — Regla de Oja

Neurona lineal con aprendizaje hebbiano normalizado que converge al primer componente principal.

### Correr el experimento

```bash
# Opción A: Makefile
make oja

# Opción B: directo
uv run python -m oja.oja_europe --config configs/oja_europe.json

# Con config personalizado:
make oja OJA_CONFIG=configs/mi_oja.json
```

### Configuración — `configs/oja_europe.json`

```json
{
  "data": "data/europe.csv",
  "lr": 0.5,
  "epochs": 1000,
  "seed": 42,
  "output_dir": "output/oja"
}
```

| Parámetro | Descripción | Default |
|---|---|---|
| `data` | Path al CSV de entrada | `"data/europe.csv"` |
| `lr` | Tasa de aprendizaje inicial (decae como `lr/step`) | `0.5` |
| `epochs` | Épocas de entrenamiento | `1000` |
| `seed` | Semilla aleatoria | `42` |
| `output_dir` | Directorio de salida | `"output/oja"` |

### Salida — `output/oja/`

Imprime en pantalla la comparación de loadings Oja vs sklearn, métricas de similitud (coseno, correlación, varianza explicada) y el ranking de países. Genera:

| Archivo | Descripción |
|---|---|
| `loadings.png` | Barras comparativas de loadings Oja vs sklearn PCA |
| `country_scores.png` | Scores de cada país sobre PC1 (Oja), ordenados |
| `convergence.png` | Distancia al autovector de sklearn y coseno en función de la época |
| `scores_comparison.png` | Scatter plot de scores Oja vs sklearn (diagonal = identidad) |
| `lr_convergence.png` | Convergencia para distintos valores de `lr` ([0.001, 0.01, 0.1, 0.5, 1.0, 5.0]) |

### Regla de actualización

```
y    = w · x                   (proyección)
w(t+1) = w(t) + η(t) · y · (x - y · w(t))   (Oja)
η(t) = lr / step               (learning rate decayente)
```

La regla de Oja garantiza `‖w‖ → 1` y converge al primer autovector de la matriz de covarianza.

---

## Ejercicio 2.2 — Red de Hopfield

Memoria asociativa que almacena y recupera patrones de letras 5×5 a partir de versiones ruidosas. El abecedario completo (A–Z) está definido en `hopfield/alphabet.py` como matrices de `{+1, −1}`.

### Comandos disponibles

```bash
# Parte (a): recuperación con ruido + parte (b): estado espúreo
# (elige automáticamente el subconjunto de k letras más ortogonales)
make hopfield

# Con config personalizado:
make hopfield HOPFIELD_CONFIG=configs/hopfield-worst-case.json

# Graficar un rango de letras en grilla 5×5
make hopfield-alphabet LET_START=C LET_END=H
# Alternativa directa:
uv run python -m hopfield.plot_letters --start c --end h --output output/hopfield/letras.png

# Graficar todas las letras
make hopfield-alphabet LET_START=A LET_END=Z

# Análisis de ortogonalidad para todas las combinaciones C(26, k)
make hopfield-orthogonality HOPFIELD_K=4
# Alternativa directa:
uv run python -m hopfield.orthogonality --k 4

# Análisis de capacidad: recall accuracy vs cantidad de patrones almacenados
make hopfield-capacity

# Idem, comparando N fijo (5×5) vs N adaptativo
make hopfield-capacity-adaptive

# Almacenar las 26 letras con escalado adaptativo (k=3, grilla 15×15 = 225 neuronas)
make hopfield-full-alphabet
```

### Configuración — `configs/hopfield.json`

```json
{
  "letters": null,
  "k": 4,
  "noise": 0.15,
  "high_noise": 0.40,
  "mode": "sync",
  "max_steps": 50,
  "spurious_attempts": 10,
  "seed": 42,
  "output_dir": "output/hopfield",
  "noise_levels_analysis": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
  "n_trials": 50
}
```

| Parámetro | Descripción | Default |
|---|---|---|
| `letters` | Lista de letras a almacenar. Si `null`, elige el subconjunto más ortogonal de tamaño `k` | `null` |
| `k` | Tamaño del subconjunto cuando `letters` es `null` | `4` |
| `noise` | Fracción de bits invertidos en la recuperación normal (parte a) | `0.15` |
| `high_noise` | Fracción de bits invertidos para buscar estados espúreos (parte b) | `0.40` |
| `mode` | Modo de actualización: `"sync"` (todas las neuronas a la vez) o `"async"` (una a una, orden aleatorio) | `"sync"` |
| `max_steps` | Máximo de iteraciones por consulta | `50` |
| `spurious_attempts` | Intentos para encontrar un estado espúreo | `10` |
| `noise_levels_analysis` | Barrido de ruido para `recovery_rate.png` | `[0.0 … 0.5]` |
| `n_trials` | Ensayos por nivel de ruido | `50` |
| `seed` | Semilla aleatoria | `42` |
| `output_dir` | Directorio de salida | `"output/hopfield"` |

> **Tip**: usa `configs/hopfield-worst-case.json` para estudiar un subconjunto de letras visualmente similares (`H, M, N, W`), que representan el peor caso de crosstalk.

### Salida — `output/hopfield/`

**Modo estándar** (`make hopfield`):

| Archivo | Descripción |
|---|---|
| `recall_a_<L>.png` | Evolución paso a paso para cada letra almacenada (parte a, ruido bajo) |
| `recall_b_try*_<L>.png` | Intentos con ruido alto buscando estado espúreo (parte b) |
| `energy_a_<L>.png` | Energía H vs iteración durante la convergencia de L |
| `crosstalk.png` | Correlación normalizada `ξᵢ·ξⱼ / N` entre patrones almacenados |
| `recovery_rate.png` | Tasa de recuperación vs nivel de ruido, una curva por letra |

**Análisis de ortogonalidad** (`make hopfield-orthogonality`):

| Archivo | Descripción |
|---|---|
| `orthogonality/dot_heatmap.png` | Matriz 26×26 de \|⟨ξᵢ, ξⱼ⟩\| entre todas las letras |
| `orthogonality/combos_k<k>.csv` | Todas las combinaciones C(26,k) con `max_abs_dot` y `mean_abs_dot` |
| `orthogonality/top_bottom_k<k>.png` | Mejores y peores subconjuntos según ortogonalidad |

**Análisis de capacidad** (`make hopfield-capacity`):

| Archivo | Descripción |
|---|---|
| `capacity/accuracy_vs_n.png` | Recall accuracy vs N patrones almacenados |
| `capacity/spurious_vs_n.png` | Tasa de estados espúreos vs N |
| `capacity/hamming_vs_n.png` | Distancia de Hamming promedio al patrón original vs N |
| `capacity/fixed_vs_adaptive.png` | Recall N fijo vs N adaptativo (con `--adaptive`) |
| `capacity/capacity.csv` | Métricas para cada combinación de modo / N / ruido |

**Modo alfabeto** (`make hopfield-full-alphabet`):

| Archivo | Descripción |
|---|---|
| `alphabet/crosstalk_alphabet.png` | Crosstalk de las 26 letras |
| `alphabet/recovery_rate_alphabet.png` | Curvas de recall por letra |

### Metodología

**Regla de Hebb (almacenamiento):**

$$W_{ij} = \frac{1}{N} \sum_{\mu} \xi_i^\mu \xi_j^\mu, \quad W_{ii} = 0$$

**Recuperación síncrona:** `s(t+1) = sgn(W s(t))` — se detecta tanto la convergencia (estado fijo) como ciclos de período 2 (oscilación clásica del modo síncrono).

**Recuperación asíncrona:** cada neurona se actualiza una a una en orden aleatorio. La energía `E = −½ sᵀ W s` es monótonamente no creciente bajo update asíncrono.

**Estado espúreo:** punto fijo distinto a todos los patrones almacenados y a sus complementos `{−ξᵘ}`.

**Elección del subconjunto más ortogonal:** se barren las **C(26, 4) = 14 950** combinaciones y se elige la que minimiza `max |⟨ξᵢ, ξⱼ⟩|`. Para grillas 5×5 (N=25), el subconjunto óptimo es **`GRTV`** con `max|⟨·,·⟩| = 1`.

### Escalado adaptativo del alfabeto completo

Con p=26 patrones y N=25 neuronas, p/N ≈ 1.04 ≫ 0.138 (límite teórico de capacidad). Para escalar, se replica cada bit en un bloque k×k via `np.kron`, llevando la grilla a (5k)×(5k):

| p | k | N = (5k)² | p/N |
|---|---|---|---|
| ≤ 3 | 1 | 25 | ≤ 0.12 |
| 4–13 | 2 | 100 | ≤ 0.13 |
| 14–31 | 3 | 225 | ≤ 0.116 |

Con k=3 (225 neuronas) la red se mantiene dentro del límite teórico. Las letras visualmente similares (A/H, C/G/O, E/F) siguen interfiriéndose por **crosstalk** (no por capacidad).

---

## Análisis cruzado — Kohonen + PCA + Oja

Scripts en `analysis/` que comparan los tres modelos entre sí.

```bash
# Validación cruzada PCA vs Kohonen (secciones 1.1, 1.2, 2.1)
make compare

# Con umbral personalizado para países "promedio"
make compare MIDDLE_THRESHOLD=0.8

# Experimento 2×2: estandarización vs datos crudos
make standardization

# Corre compare + standardization
make analysis
```

### Salidas

| Directorio | Descripción |
|---|---|
| `output/compare/` | Mapas y rankings comparativos PCA vs Kohonen vs Oja |
| `output/standardization/` | Grillas SOM entrenadas con/sin estandarización, comparación cuantitativa |

---

## Tests

```bash
# Correr todos los tests
make test

# Alternativa directa
uv run pytest -v
```

La suite cubre (>100 tests):

- **Hopfield**: pesos de Hebb verificados contra el ejemplo de las diapositivas, recuperación sync + async, detección de ciclos, complemento como atractor, escalado adaptativo, ortogonalidad.
- **SOM**: entrenamiento, cálculo de BMU, U-Matrix, error de cuantización, error topológico.
- **Oja**: convergencia al primer componente principal, comparación con sklearn.
- **PCA**: varianza explicada, loadings.
- **Configs**: validación de parámetros de configuración.

---

## Limpieza de outputs

```bash
make clean                  # Borra todo el output generado
make clean-kohonen          # Borra output/kohonen/
make clean-convergence      # Borra output/convergence/
make clean-pca              # Borra pca_test/plots/ y CSVs generados
make clean-oja              # Borra oja/output/oja/ (o output/oja/)
make clean-hopfield         # Borra output/hopfield/
make clean-analysis         # Borra output/compare/ y output/standardization/
```

---

## Correr todo de una vez

```bash
make all
```

Equivale a: `make kohonen` + `make pca` + `make oja` + `make hopfield`

---

## Grupo

**SIA-G02** — ITBA, 2025
