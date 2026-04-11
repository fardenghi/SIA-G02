# compare_method_combinations.py

Script para comparar empíricamente dos combinaciones distintas de operadores del AG sobre la misma imagen. Permite evaluar cómo afectan las decisiones de selección, supervivencia, cruza y mutación al resultado final y a la velocidad de convergencia.

---

## Uso

```bash
uv run python scripts/compare_method_combinations.py \
  --image <ruta_imagen> \
  [--triangles N] \
  [--generations N] \
  [--population N] \
  [--fitness MÉTODO] \
  [--config <ruta_yaml>] \
  [--output <directorio>] \
  [--max-size N]
```

### Argumentos

| Argumento | Alias | Default | Descripción |
|---|---|---|---|
| `--image` | `-i` | (requerido) | Ruta a la imagen objetivo |
| `--triangles` | `-t` | `50` | Cantidad de triángulos por individuo |
| `--generations` | `-g` | `200` | Número máximo de generaciones |
| `--population` | `-p` | `50` | Tamaño de la población |
| `--fitness` | `-f` | `linear` | Función de fitness a usar |
| `--config` | `-c` | `scripts/compare_method_combinations_config.yaml` | Archivo YAML con las dos combinaciones |
| `--output` | `-o` | `output/compare_combinations` | Directorio donde se guardan los resultados |
| `--max-size` | — | `128` | Tamaño máximo del lado más largo de la imagen (se redimensiona si es mayor) |

### Valores válidos para `--fitness`

`linear`, `rmse`, `inverse_normalized`, `exponential`, `inverse_mse`, `detail_weighted`, `composite`, `ssim`, `edge_loss`

---

## Archivo de configuración YAML

Por defecto el script busca `compare_method_combinations_config.yaml` en el directorio desde donde se ejecuta. Se puede cambiar con `--config`.

### Estructura

```yaml
combination_a:
  name: "Nombre descriptivo de la combinación A"
  selection:
    method: <método>
    <params_propios_del_método>
  survival:
    method: <método>
    selection_method: <método_selección_sobrevivientes>
    offspring_ratio: <float>
  crossover:
    method: <método>
    probability: <float>
  mutation:
    method: <método>
    probability: <float>
    gene_probability: <float>
    position_delta: <float>
    color_delta: <int>
    alpha_delta: <float>
    field_probability: <float>

combination_b:
  # misma estructura que combination_a
```

### Métodos y parámetros válidos

#### Selección (`selection.method`)

Cada método acepta **solo sus propios parámetros**:

| Método | Parámetros propios |
|---|---|
| `elite` | — |
| `roulette` | — |
| `universal` | — |
| `rank` | — |
| `tournament` | `tournament_size` (int, default 3) |
| `probabilistic_tournament` | `threshold` (float 0.5–1.0, default 0.75) |
| `boltzmann` | `boltzmann_t0` (float), `boltzmann_tc` (float), `boltzmann_k` (float) |

#### Supervivencia (`survival.method`)

| Valor | Descripción |
|---|---|
| `exclusive` | Los hijos tienen prioridad; padres solo completan si K < N |
| `additive` | Se seleccionan N individuos del pool padres + hijos |

Parámetros adicionales:
- `selection_method`: método para elegir sobrevivientes dentro del pool (`elite`, `tournament`, etc.)
- `offspring_ratio`: ratio de hijos a generar respecto a la población (K = N × ratio)

#### Cruza (`crossover.method`)

| Valor | Descripción |
|---|---|
| `single_point` | Corte en un punto aleatorio |
| `two_point` | Corte en dos puntos, intercambia sección central |
| `uniform` | Cada triángulo se hereda de un padre al azar (p=0.5) |
| `annular` | Segmento circular de longitud aleatoria |
| `spatial_zindex` | Cruza por posición espacial en el canvas |
| `arithmetic` | Promedia los valores de ambos triángulos |

Parámetro adicional:
- `probability`: probabilidad de aplicar cruza (default 0.8)

#### Mutación (`mutation.method`)

| Valor | Descripción |
|---|---|
| `single_gene` | Muta exactamente 1 triángulo |
| `limited_multigen` | Muta entre 1 y `max_genes` triángulos |
| `uniform_multigen` | Cada triángulo muta con probabilidad `gene_probability` |
| `complete` | Muta todos los triángulos |
| `error_map_guided` | Sesga mutaciones hacia zonas de alto error |

Parámetros adicionales:

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `probability` | float | 0.3 | Probabilidad de mutar el individuo (Pm) |
| `gene_probability` | float | 0.1 | Probabilidad por gen (para `uniform_multigen`) |
| `position_delta` | float | 0.1 | Magnitud de perturbación en coordenadas [0,1] |
| `color_delta` | int | 30 | Magnitud de perturbación en RGB [0,255] |
| `alpha_delta` | float | 0.1 | Magnitud de perturbación en alpha [0,1] |
| `field_probability` | float | 1.0 | Probabilidad de mutar cada campo individual del triángulo |

---

## Salidas generadas

```
<output>/
├── comparison_images.png   # Imágenes resultado de A y B lado a lado
├── comparison_fitness.png  # Curvas de evolución del fitness por generación
└── comparison_time.png     # Barras de tiempo de ejecución de cada combinación
```

---

## Ejemplo completo

```bash
uv run python scripts/compare_method_combinations.py \
  --image images/bandera_argentina.png \
  --triangles 50 \
  --generations 300 \
  --population 60 \
  --fitness linear \
  --config compare_method_combinations_config.yaml \
  --output output/experimento_1 \
  --max-size 128
```

Con el YAML por defecto esto ejecutaría:
- **Combinación A**: Torneo (k=3) + Supervivencia Exclusiva + Cruza Un Punto + Mutación Multigen Uniforme
- **Combinación B**: Ruleta + Supervivencia Aditiva + Cruza Uniforme + Mutación Multigen Limitado

Y generaría los 3 PNGs en `output/experimento_1/`.
