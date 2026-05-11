# TP4 — Análisis de Componentes Principales (PCA)

Análisis PCA sobre el dataset de países europeos usando scikit-learn.

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

## Uso

```bash
uv run --with pandas --with scikit-learn python pca_europe.py [--n-components N]
```

**Argumentos:**
- `--n-components` — número de componentes principales a calcular (default: `1`)

**Ejemplo con 3 componentes:**
```bash
uv run --with pandas --with scikit-learn python pca_europe.py --n-components 3
```

## Salida

Para cada componente imprime:
- Varianza explicada (%)
- Autovalor
- Loadings de cada feature ordenados por magnitud

Seguido de una tabla de scores por país (ordenados por PC1 descendente).

## Metodología

1. Se estandarizan las features con `StandardScaler` (media 0, varianza 1) para que la diferencia de escala entre variables no distorsione el PCA.
2. Se ajusta `PCA(n_components=N)` sobre los datos estandarizados.
3. Los **loadings** (`pca.components_`) son los autovectores de la matriz de covarianza — definen la dirección de cada componente.
4. Los **autovalores** (`pca.explained_variance_`) indican la varianza capturada en cada dirección. La suma de todos los autovalores es igual al número de features (7), ya que los datos están estandarizados.
