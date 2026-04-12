# Scripts de Comparación

Esta carpeta contiene scripts para realizar comparaciones experimentales de diferentes componentes del algoritmo genético.

## Scripts Disponibles

### 1. `compare_selection.py` - Comparativa de Métodos de Selección

Compara todos los métodos de selección disponibles usando los mismos parámetros y genera gráficos de evolución del fitness.

**Uso básico:**
```bash
uv run python scripts/compare_selection.py --image input/logo.png
```

**Parámetros:**
- `--image, -i`: Imagen objetivo (requerido)
- `--generations, -g`: Número de generaciones (default: 200)
- `--population, -p`: Tamaño de población (default: 50)
- `--triangles, -t`: Genes por individuo (default: 50)
- `--shape`: Familia de formas (`triangle` o `ellipse`, default: `triangle`)
- `--max-size`: Tamaño máximo de imagen (default: 128)
- `--output, -o`: Directorio de salida (default: `output/comparativa`)
- `--methods`: Métodos específicos a comparar (default: todos)

**Métodos disponibles:**
- `elite`: Elite
- `tournament`: Torneo Determinístico
- `probabilistic_tournament`: Torneo Probabilístico
- `roulette`: Ruleta
- `universal`: Universal (SUS)
- `boltzmann`: Boltzmann
- `rank`: Ranking

**Ejemplo completo:**
```bash
uv run python scripts/compare_selection.py \
    --image input/logo.png \
    --generations 500 \
    --population 100 \
    --triangles 50 \
    --shape ellipse \
    --methods elite tournament roulette boltzmann
```

**Salidas generadas:**
- `comparativa_fitness.png`: Gráfico comparativo de evolución
- `imagenes/*.png`: Imagen final de cada método
- Tabla resumen en consola

---

### 2. `compare_crossover.py` - Comparativa de Métodos de Cruza

Compara todos los métodos de cruza (crossover) disponibles usando el mismo método de selección y genera gráficos de evolución del fitness.

**Uso básico:**
```bash
uv run python scripts/compare_crossover.py --image input/logo.png
```

**Parámetros:**
- `--image, -i`: Imagen objetivo (requerido)
- `--generations, -g`: Número de generaciones (default: 200)
- `--population, -p`: Tamaño de población (default: 50)
- `--triangles, -t`: Genes por individuo (default: 50)
- `--shape`: Familia de formas (`triangle` o `ellipse`, default: `triangle`)
- `--max-size`: Tamaño máximo de imagen (default: 128)
- `--output, -o`: Directorio de salida (default: `output/comparativa_crossover`)
- `--methods`: Métodos específicos a comparar (default: todos)
- `--selection`: Método de selección a usar (default: `tournament`)

**Métodos de cruza disponibles:**
- `single_point`: Cruce de Un Punto
- `two_point`: Cruce de Dos Puntos
- `uniform`: Cruce Uniforme
- `annular`: Cruce Anular (Circular)

**Ejemplo completo:**
```bash
uv run python scripts/compare_crossover.py \
    --image input/logo.png \
    --generations 500 \
    --population 100 \
    --triangles 50 \
    --shape ellipse \
    --selection tournament \
    --methods single_point two_point uniform annular
```

**Salidas generadas:**
- `comparativa_fitness.png`: Gráfico comparativo de evolución
- `imagenes/*.png`: Imagen final de cada método
- Tabla resumen en consola

---

## Ejemplos de Uso

### Comparación rápida (para testing)
```bash
# Selección
uv run python scripts/compare_selection.py \
    -i input/logo.png -g 50 -p 30 -t 20 --shape ellipse --max-size 64

# Crossover
uv run python scripts/compare_crossover.py \
    -i input/logo.png -g 50 -p 30 -t 20 --shape ellipse --max-size 64
```

### Comparación completa (para análisis)
```bash
# Selección
uv run python scripts/compare_selection.py \
    -i input/logo.png -g 1000 -p 100 -t 100 --max-size 256

# Crossover
uv run python scripts/compare_crossover.py \
    -i input/logo.png -g 1000 -p 100 -t 100 --max-size 256
```

### Comparar solo algunos métodos
```bash
# Solo torneos y elite
uv run python scripts/compare_selection.py \
    -i input/logo.png \
    --methods elite tournament probabilistic_tournament

# Solo cruce de puntos
uv run python scripts/compare_crossover.py \
    -i input/logo.png \
    --methods single_point two_point
```

---

## Notas

- Todos los scripts fijan una semilla aleatoria internamente para asegurar reproducibilidad
- Los tiempos de ejecución dependen del tamaño de imagen, número de generaciones y población
- Para comparaciones justas, todos los métodos se ejecutan con los mismos parámetros
- Los gráficos se guardan en formato PNG con alta resolución (150 dpi)
- Las imágenes finales permiten comparación visual de los resultados
