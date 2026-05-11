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

## Tests

```bash
uv run pytest
```

Cubre inicialización del SOM, BMU, funciones de vecindad, decaimiento, entrenamiento, U-matrix, validación del config y generación de gráficos (47 tests).
