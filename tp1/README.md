# TP1 — Metodos de Busqueda

**Sistemas de Inteligencia Artificial — ITBA 2026 1C**

Motor de busqueda para resolver tableros de **Sokoban** implementado en Python.

---

## Descripcion

El motor lee un tablero de Sokoban desde un archivo de texto, ejecuta un algoritmo de busqueda y reporta la solucion encontrada junto con metricas de rendimiento (nodos expandidos, nodos en frontera, tiempo, costo de la solucion).

---

## Algoritmos implementados

Se implementaron 5 algoritmos de busqueda, divididos en **no informados** e **informados**:

### No informados (no requieren heuristica)

| Algoritmo | Descripcion | Optimo | Completo |
|-----------|-------------|--------|----------|
| **BFS** | Breadth-First Search. Usa frontera FIFO (cola). Verifica objetivo al generar el nodo. | Si (minimo pasos) | Si |
| **DFS** | Depth-First Search. Usa frontera LIFO (pila). Verifica objetivo al generar el nodo. | No | Si (con graph-search) |
| **IDDFS** | Iterative Deepening DFS. Ejecuta DFS con limite de profundidad incremental (0, 1, 2, ...) en una variante graph-search: usa un conjunto de visitados fresco por cada iteracion. | Si (minima profundidad) | Si |

### Informados (requieren heuristica)

| Algoritmo | Descripcion | Optimo | Completo |
|-----------|-------------|--------|----------|
| **Greedy** | Best-First Search. Expande el nodo con menor h(n). Usa frontera de prioridad (min-heap). Verifica objetivo al expandir. | No | Si |
| **A\*** | Expande el nodo con menor f(n) = g(n) + h(n). Usa frontera de prioridad (min-heap). Verifica objetivo al expandir. | Si (con heuristica admisible) | Si |

Todos los algoritmos usan **graph-search** con un conjunto de visitados para evitar re-explorar estados.

---

## Heuristicas disponibles

Para los algoritmos informados (Greedy y A*) se implementaron 5 heuristicas:

| Heuristica | Valor en config | Admisible | Descripcion |
|------------|-----------------|-----------|-------------|
| **Manhattan** | `manhattan` | Si | Suma de distancias Manhattan desde cada caja a su objetivo mas cercano. |
| **Euclidean** | `euclidean` | Si | Suma de distancias Euclidianas desde cada caja a su objetivo mas cercano. Siempre da valores <= Manhattan, por lo que guia menos la busqueda. |
| **Dead Square** | `dead_square` | Si | Retorna infinito si alguna caja que no esta en objetivo queda atrapada en una esquina; sino retorna 0. Funciona como mecanismo de poda mas que como estimador de distancia. |
| **Hungarian** | `hungarian` | Si | Suma de distancias Manhattan asignando de manera optima cajas a metas minimizando la distancia neta (Algoritmo Hungaro). Requiere Scipy. |
| **Weighted Hungarian** | `weighted_hungarian` | No | Variante agresiva: multiplica Hungarian por 1.5 y suma 0.5 veces la distancia del jugador a la caja no resuelta mas cercana. Puede acelerar la busqueda, pero no garantiza optimalidad. La poda por esquinas puede agregarse aparte combinandola con `dead_square`. |

### Deteccion de dead squares

En esta version no se preprocesa el tablero. La deteccion se hace al momento de consultar una celda: se considera "dead square" a una esquina no objetivo formada por dos paredes perpendiculares o por el borde del tablero. Es una aproximacion mas simple que la deteccion global por BFS, pero cumple con la idea de podar estados obviamente irresolubles sin precomputacion.

---

## Tableros disponibles

Los tableros principales estan en `boards/sokoban/` y el benchmark automatico usa todos los archivos `level_*.txt` (actualmente: **21**).

Actualmente hay:
- `level_01.txt` a `level_11.txt`
- `level_12_*.txt` (8 variantes direccionales)
- `level_13.txt` y `level_14.txt`

Ejemplos representativos:

| Tablero | Tamanio | Cajas | Notas |
|---------|---------|-------|-------|
| `level_01.txt` | 6x7 | 1 | Caso simple |
| `level_02.txt` | 9x7 | 3 | Dificultad intermedia |
| `level_03.txt` | 10x11 | 8 | Caso grande y costoso |
| `level_04.txt` | 13x13 | 2 | Caso mediano |
| `level_05.txt` | 9x5 | 2 | Irresoluble |
| `level_06.txt` | 5x6 | 1 | Caso trivial |
| `level_07.txt` | 6x8 | 2 | Caso mediano |
| `level_08.txt` | 7x8 | 2 | Caso mediano |
| `level_09.txt` | 8x11 | 16 | Caso muy grande |
| `level_10.txt` | 8x7 | 5 | Caso intermedio |
| `level_11.txt` | 10x16 | 2 | Caso intermedio/largo |
| `level_12_up_left.txt` | 20x19 | 1 | Variante direccional compacta |
| `level_14.txt` | 20x50 | 1 | Variante direccional extensa |

Tableros especiales para experimentos puntuales:

| Tablero | Tamanio | Cajas | Uso |
|---------|---------|-------|-----|
| `weighted_hungarian_counterexample.txt` | 7x35 | 2 | Contraejemplo donde `weighted_hungarian` encuentra una solucion peor, pero mucho mas rapido |
| `dead_square_corner_showcase.txt` | 8x10 | 2 | Showcase para comparar `manhattan` vs `max(manhattan, dead_square)` |
| `four_corners_showcase.txt` | 17x28 | 2 | Contraste para comparar el aporte de dead squares |
| `box_count_traps/boxes_02.txt` | 8x11 | 2 | Familia para comparar heuristicas segun cantidad de cajas |
| `box_count_traps/boxes_05.txt` | 8x24 | 5 | Familia para comparar heuristicas segun cantidad de cajas |
| `box_count_traps/boxes_11.txt` | 10x40 | 11 | Familia para comparar heuristicas segun cantidad de cajas |

---

## Requisitos

- [uv](https://docs.astral.sh/uv/) — gestor de paquetes y entornos Python
- Python 3.12 (se descarga automaticamente con `uv`)

---

## Instalacion

```bash
# Clonar el repositorio y entrar al directorio tp1
cd tp1

# Instalar dependencias y crear el entorno virtual
uv sync
```

Para correr scripts de benchmark y generacion de graficos, conviene instalar tambien las dependencias de desarrollo:

```bash
uv sync --dev
```

---

## Formato de configuracion TOML

Los archivos de configuracion viven en `configs/sokoban/`. Estructura:

```toml
[search]
algorithm = "<algoritmo>"                  # Requerido
board = "boards/sokoban/<nivel>.txt"       # Requerido
heuristic = "<heuristica>"                 # String (1 heuristica)
# o
heuristic = ["<h1>", "<h2>"]              # Array (se combina con max(h1, h2, ...))
```

### Valores validos

| Campo | Opciones |
|-------|----------|
| `algorithm` | `bfs`, `dfs`, `iddfs`, `greedy`, `astar` |
| `board` | Ruta a cualquier archivo `.txt` en `boards/sokoban/` |
| `heuristic` | String o array con `manhattan`, `euclidean`, `dead_square`, `hungarian`, `weighted_hungarian` |

### Validaciones

- El algoritmo debe ser uno de los 5 listados
- El archivo de tablero debe existir
- Los algoritmos informados (`greedy`, `astar`) **deben** tener una heuristica
- Si `heuristic` es array, se usa una heuristica compuesta: `max(h1, ..., hn)`
- Cada heuristica del array debe ser una de las listadas arriba

---

## Ejecucion

### Ejecuciones normales

| Comando | Flags (todos los disponibles)                                                                                                                                                   |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `uv run tp1-search <config.toml>` | `--save-replay` (guarda replay JSON en `results/raw/`)                                                                                                                          |
| `uv run tp1-animate <replay.json>` | `--speed FPS` (default: `2.0`), `--cell-size PX` (default: `64`), `--save-gif OUT.gif` (exporta GIF y termina), `--save-video` (exporta MP4 en `results/animations/` y termina) |
| `uv run pytest [flags_de_pytest]` | -                                                                                                                                                                               |

Configs base listas para correr:

```bash
uv run tp1-search configs/sokoban/bfs.toml
uv run tp1-search configs/sokoban/dfs.toml
uv run tp1-search configs/sokoban/iddfs.toml
uv run tp1-search configs/sokoban/greedy.toml
uv run tp1-search configs/sokoban/astar.toml
```

Controles de teclado en `tp1-animate`:

| Tecla | Que hace |
|-------|----------|
| `UP` | Aumenta la velocidad de reproduccion en `+0.5 FPS` (maximo `30.0 FPS`). |
| `DOWN` | Disminuye la velocidad de reproduccion en `-0.5 FPS` (minimo `0.5 FPS`). |
| `SPACE` | Pausa o reanuda la reproduccion. |
| `RIGHT` | Avanza 1 frame solo si esta en pausa. |
| `LEFT` | Retrocede 1 frame solo si esta en pausa. |
| `R` | Reinicia la animacion al frame inicial y la deja reproduciendo (sin pausa). |
| `Q` | Cierra la ventana del animador. |

La velocidad actual se muestra en el HUD inferior como `Velocidad: X.Y FPS | UP/DOWN`.

### Ejecuciones de analisis

Requieren dependencias de desarrollo: `uv sync --dev`.

| Script | Comando | Flags (todos los disponibles) |
|--------|---------|-------------------------------|
| Benchmark total | `uv run python scripts/run_batch.py` | `--timeout SECONDS` (default: `30.0`), `--output PATH` (default: `results/benchmark/benchmark.csv`) |
| Heuristicas vs cajas | `uv run python scripts/box_count_traps_plots.py` | `--timeout SECONDS` (default: `12.0`), `--runs N` (default: `5`), `--csv PATH`, `--nodes-plot PATH`, `--time-plot PATH`, `--cost-plot PATH`, `--frontier-plot PATH`, `--csv-only`, `--plot-only` |
| Showcase dead square | `uv run python scripts/dead_square_showcase.py` | `--board PATH` (repetible; si se omite usa dos tableros default), `--csv PATH`, `--plot PATH`, `--runs N` (default: `5`) |
| Trade-off weighted | `uv run python scripts/plot_weighted_tradeoff.py` | `--board PATH`, `--out PATH`, `--csv PATH`, `--runs N` (default: `5`) |
| Un tablero, todos los algoritmos | `uv run python scripts/single_board_all_algorithms.py` | `--board PATH` (default: `boards/sokoban/level_02.txt`), `--runs N` (default: `10`), `--timeout SECONDS` (default: `30.0`), `--csv PATH`, `--plot PATH` |
| Corridas BFS/DFS/IDDFS | `uv run python scripts/bfs_vs_dfs_vs_iddfs.py --map PATH` | `--map PATH` (requerido) |
| Plots BFS/DFS/IDDFS | `uv run python scripts/bfs_vs_dfs_vs_iddfs_plots.py --result DIR` | `--result DIR` (requerido), `--zoom-expanded-nodes`, `--zoom-frontier-nodes`, `--zoom-execution-time`, `--zoom-solution-cost` |
| Plot frontera final vs maxima | `uv run python scripts/plot_max_frontier.py --result DIR` | `--result DIR` (requerido) |

`run_batch.py` hoy corre `21` tableros `level_*.txt`: `3` algoritmos no informados + `2` informados x `5` heuristicas = **273 ejecuciones**.

---

## Formato de tableros

Los tableros se definen en archivos `.txt` dentro de `boards/sokoban/`.
Cada celda se representa con un simbolo:

| Simbolo | Significado |
|---------|-------------|
| `#` | Pared |
| `@` | Jugador |
| `+` | Jugador sobre objetivo |
| `$` | Caja |
| `*` | Caja sobre objetivo |
| `.` | Objetivo |
| ` ` | Celda vacia |

**Ejemplo — `boards/sokoban/level_01.txt`:**

```
  #####
  #   #
  # $ #
  # . #
  # @ #
  #####
```

**Restricciones del parser:**
- Debe haber exactamente 1 jugador
- La cantidad de cajas debe ser igual a la cantidad de objetivos
- No se admiten simbolos desconocidos

---

## Metricas reportadas

| Metrica | Descripcion |
|---------|-------------|
| Resultado | EXITO o FRACASO |
| Costo | Cantidad de pasos de la solucion |
| Nodos expandidos | Cuantos estados fueron procesados |
| Nodos en frontera | Cuantos nodos quedaron sin explorar al terminar |
| Camino | Secuencia de direcciones (UP, DOWN, LEFT, RIGHT) |
| Tiempo | Duracion del algoritmo en segundos |

---

## Notas para el analisis

- **level_05 es irresoluble**: todos los algoritmos deben retornar FAILURE. Sirve para verificar que el motor detecta correctamente tableros sin solucion.
- **run_batch.py** benchmarkea todos los `level_*.txt`; hoy son `21` tableros (total: `273` corridas por pasada completa).
- **Manhattan vs Euclidean**: Manhattan siempre da valores >= Euclidean para el mismo estado, por lo que guia mejor la busqueda y expande menos nodos.
- **Dead square**: no estima distancia (solo retorna 0 o infinito), pero poda estados irrecuperables de forma muy efectiva.
- **dead_square_corner_showcase.txt** esta pensado para mostrar el efecto de combinar `dead_square` con otra heuristica.
- **weighted_hungarian**: es util para guiar la busqueda mas agresivamente, pero en `boards/sokoban/weighted_hungarian_counterexample.txt` encuentra una solucion peor que `manhattan`, mostrando que no garantiza optimalidad.
- **box_count_traps/** se usa para comparar escalabilidad de heuristicas al aumentar la cantidad de cajas.
- Los algoritmos no informados (BFS, DFS, IDDFS) no usan heuristica y su rendimiento depende unicamente de la estructura del espacio de estados.

---

## Estructura del proyecto

```
tp1/
├── configs/sokoban/                    # Archivos de configuracion TOML
├── boards/sokoban/                     # Tableros (.txt)
│   ├── level_*.txt                     # 21 tableros usados por run_batch
│   ├── level_12_*.txt                  # Variantes direccionales
│   ├── dead_square_corner_showcase.txt # Showcase dead_square
│   ├── four_corners_showcase.txt       # Contraste dead_square
│   ├── weighted_hungarian_counterexample.txt
│   └── box_count_traps/                # Familia para comparar heuristicas
├── scripts/                            # Scripts de analisis
│   ├── run_batch.py
│   ├── box_count_traps_plots.py
│   ├── dead_square_showcase.py
│   ├── plot_weighted_tradeoff.py
│   ├── single_board_all_algorithms.py
│   ├── bfs_vs_dfs_vs_iddfs.py
│   ├── bfs_vs_dfs_vs_iddfs_plots.py
│   └── plot_max_frontier.py
├── results/
│   ├── raw/                            # Replays JSON (--save-replay)
│   ├── plots/                          # Graficos y HTML de analisis
│   └── benchmark/
│       ├── benchmark.csv
│       ├── heuristics_by_boxcount_traps_greedy.csv
│       ├── dead_square_showcase.csv
│       └── weighted_hungarian_counterexample_tradeoff.csv
├── tests/
│   ├── test_search.py
│   ├── test_heuristics.py
│   ├── test_animator.py
│   ├── test_config.py
│   ├── test_state.py
│   └── test_successors.py
└── src/tp1_search/
    ├── cli.py                          # Entry point (tp1-search)
    ├── config.py                       # Carga y validacion de TOML
    ├── types.py                        # Position, Direction
    ├── search/
    │   ├── node.py
    │   ├── frontier.py
    │   ├── state_key.py
    │   ├── bfs.py
    │   ├── dfs.py
    │   ├── iddfs.py
    │   ├── greedy.py
    │   └── astar.py
    ├── sokoban/
    │   ├── board.py
    │   ├── state.py
    │   ├── parser.py
    │   ├── actions.py
    │   ├── successors.py
    │   ├── goal.py
    │   └── heuristics.py
    ├── output/
    │   ├── writer.py
    │   ├── animator_cli.py             # Entry point (tp1-animate)
    │   └── pygame_anim.py
    └── metrics/
        └── result.py
```
