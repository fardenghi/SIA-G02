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
| **Hungarian** | `hungarian` | Si | Suma de distancias Manhattan asignando de manera optima cajas a metas minimizando la distancia neta (Algoritmo Hungaro), incluyendo chequeo de esquinas muertas. Requiere Scipy. |
| **Weighted Hungarian** | `weighted_hungarian` | No | Variante agresiva: multiplica Hungarian por 1.5 y suma 0.5 veces la distancia del jugador a la caja no resuelta mas cercana. Puede acelerar la busqueda, pero no garantiza optimalidad. La poda por esquinas puede agregarse aparte combinandola con `dead_square`. |

### Deteccion de dead squares

En esta version no se preprocesa el tablero. La deteccion se hace al momento de consultar una celda: se considera "dead square" a una esquina no objetivo formada por dos paredes perpendiculares o por el borde del tablero. Es una aproximacion mas simple que la deteccion global por BFS, pero cumple con la idea de podar estados obviamente irresolubles sin precomputacion.

---

## Tableros disponibles

Los tableros principales estan en `boards/sokoban/` y el benchmark automatico usa todos los archivos `level_*.txt`.

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

Tableros especiales para experimentos puntuales:

| Tablero | Tamanio | Cajas | Uso |
|---------|---------|-------|-----|
| `weighted_hungarian_counterexample.txt` | 7x35 | 2 | Contraejemplo donde `weighted_hungarian` encuentra una solucion peor, pero mucho mas rapido |
| `dead_square_corner_showcase.txt` | 8x10 | 2 | Showcase para comparar `manhattan` vs `max(manhattan, dead_square)` |
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

## Uso

### Ejecucion individual

```bash
uv run tp1-search <archivo_config.toml>
```

**Flags opcionales:**

| Flag | Descripcion |
|------|-------------|
| `--save-replay` | Guarda un archivo JSON en `results/raw/` para animar despues |
| `--show-dead-squares` | Imprime el tablero con dead squares marcados como `X` |

**Alternativa (ejecutar como modulo):**

```bash
uv run python -m tp1_search.cli <archivo_config.toml>
```

### Salida de ejemplo

```
Algoritmo: bfs
Tablero:   boards/sokoban/level_01.txt

========================================
Resultado:          EXITO
Costo (pasos):      6
Nodos expandidos:   26
Nodos en frontera:  10
Tiempo:             0.0002s
Camino (6 pasos): UP -> LEFT -> UP -> UP -> RIGHT -> DOWN
========================================
```

---

## Instructivo: ejecutar algoritmos y benchmarks

### Resumen de combinaciones

| Algoritmo | Heuristicas | Niveles | Total ejecuciones |
|-----------|-------------|---------|-------------------|
| BFS | — | 10 | 10 |
| DFS | — | 10 | 10 |
| IDDFS | — | 10 | 10 |
| Greedy | 5 | 10 | 50 |
| A* | 5 | 10 | 50 |
| **Total** | | | **130** |

---

Para ejecucion individual, los TOML base estan en `configs/sokoban/`:

```bash
uv run tp1-search configs/sokoban/bfs.toml
uv run tp1-search configs/sokoban/dfs.toml
uv run tp1-search configs/sokoban/iddfs.toml
uv run tp1-search configs/sokoban/greedy.toml
uv run tp1-search configs/sokoban/astar.toml
```

Los archivos `greedy.toml` y `astar.toml` se editan cambiando `board` y `heuristic` segun el experimento que quieras correr.

---

## Ejecucion masiva: benchmark automatico

En lugar de editar muchas configuraciones a mano, `run_batch.py` ejecuta todas las combinaciones sobre los tableros `level_*.txt`:

```bash
uv run python scripts/run_batch.py --timeout 30 --output results/benchmark/benchmark.csv
```

**Parametros:**

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `--timeout` | 30 | Tiempo maximo en segundos por ejecucion |
| `--output` | `results/benchmark/benchmark.csv` | Ruta del archivo CSV de salida |

**Columnas del CSV generado:**

| Columna | Descripcion |
|---------|-------------|
| `board` | Nombre del tablero |
| `algorithm` | Algoritmo usado |
| `heuristic` | Heuristica usada (`-` para algoritmos no informados) |
| `success` | Si encontro solucion |
| `cost` | Cantidad de pasos de la solucion |
| `expanded_nodes` | Nodos expandidos |
| `frontier_nodes` | Nodos restantes en la frontera |
| `time_elapsed` | Tiempo en segundos |
| `timed_out` | Si se excedio el timeout |

---

## Generacion de graficos

Los graficos actuales se generan con scripts especificos segun el experimento:

- `scripts/box_count_traps_plots.py`: compara heuristicas con **Greedy** en funcion de la cantidad de cajas sobre `boards/sokoban/box_count_traps/`
- `scripts/dead_square_showcase.py`: compara `manhattan` vs `max(manhattan, dead_square)` en un tablero con muchas esquinas
- `scripts/plot_weighted_tradeoff.py`: compara `manhattan` vs `weighted_hungarian` en el contraejemplo no admisible
- `scripts/single_board_all_algorithms.py`: corre un mismo tablero 10 veces para todos los algoritmos y grafica barras con error, usando combinaciones utiles de heuristicas para Greedy y A*

Ejemplos:

```bash
uv run python scripts/box_count_traps_plots.py --csv-only --timeout 12 --runs 5
uv run python scripts/box_count_traps_plots.py --plot-only

uv run python scripts/dead_square_showcase.py --runs 5
uv run python scripts/plot_weighted_tradeoff.py --runs 5
uv run python scripts/single_board_all_algorithms.py --board boards/sokoban/level_02.txt --runs 10
```

Archivos generados actualmente:

- `results/plots/heuristics_by_boxcount_nodes.png`
- `results/plots/heuristics_by_boxcount_time.png`
- `results/plots/heuristics_by_boxcount_time.html`
- `results/plots/dead_square_showcase.png`
- `results/plots/weighted_hungarian_counterexample_tradeoff.png`
- `results/plots/level_02_all_algorithms.png`

---

## Animacion de soluciones

### Paso 1: guardar el replay

```bash
uv run tp1-search configs/sokoban/bfs.toml --save-replay
```

Esto genera un archivo JSON en `results/raw/`.

### Paso 2: reproducir la animacion

```bash
uv run tp1-animate results/raw/<archivo_generado>.json
```

Para exportar la misma animacion como GIF:

```bash
uv run tp1-animate results/raw/<archivo_generado>.json --save-gif results/raw/animacion.gif
```

**Parametros opcionales:**

| Parametro | Default | Descripcion                            |
|-----------|---------|----------------------------------------|
| `--speed` | 2.0 | Frames por segundo (velocidad inicial) |
| `--cell-size` | 64 | Tamaño en pixeles de cada celda        |
| `--save-gif` | - | Ruta de salida para exportar la animacion como GIF |

Ejemplo:

```bash
uv run tp1-animate results/raw/replay.json --speed 3 --cell-size 64
```

El GIF usa la velocidad indicada por `--speed` como duracion entre frames.

### Controles de teclado durante la animacion

| Tecla | Accion |
|-------|--------|
| `UP` | Aumentar velocidad (+0.5 FPS) |
| `DOWN` | Disminuir velocidad (-0.5 FPS) |
| `SPACE` | Pausar / reanudar |
| `RIGHT` | Avanzar un frame (solo en pausa) |
| `LEFT` | Retroceder un frame (solo en pausa) |
| `R` | Reiniciar animacion desde el inicio |
| `Q` | Salir |

La velocidad actual se muestra en el HUD inferior. Rango permitido: 0.5 a 30.0 FPS.

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

## Tests

```bash
uv run pytest          # correr todos los tests
uv run pytest -v       # con detalle por test
```

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
- **run_batch.py** benchmarkea todos los `level_*.txt`, por lo que hoy incluye `level_01` a `level_10`.
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
├── configs/sokoban/            # Archivos de configuracion TOML
├── boards/sokoban/             # Tableros de Sokoban (.txt)
│   ├── level_01.txt            # 6x7, 1 caja
│   ├── level_02.txt            # 9x7, 3 cajas
│   ├── level_03.txt            # 10x11, 8 cajas
│   ├── level_04.txt            # 13x13, 2 cajas
│   ├── level_05.txt            # 9x5, 2 cajas (irresoluble)
│   ├── level_06.txt            # 5x6, 1 caja
│   ├── level_07.txt            # 6x8, 2 cajas
│   ├── level_08.txt            # 7x8, 2 cajas
│   ├── level_09.txt            # 8x11, 16 cajas
│   ├── level_10.txt            # 8x7, 5 cajas
│   ├── dead_square_corner_showcase.txt        # Showcase para dead squares
│   ├── weighted_hungarian_counterexample.txt  # Contraejemplo no admisible
│   └── box_count_traps/        # Familia de tableros para comparar heuristicas
├── scripts/
│   ├── run_batch.py             # Benchmark general sobre level_*.txt
│   ├── box_count_traps_plots.py # Benchmark + plots por cantidad de cajas
│   ├── dead_square_showcase.py  # Plot puntual para dead_square
│   └── plot_weighted_tradeoff.py # Plot puntual para weighted_hungarian
├── results/
│   ├── raw/                    # Replays JSON (generados con --save-replay)
│   ├── plots/                  # Graficos y HTML generados para el analisis
│   └── benchmark/
│       ├── benchmark.csv       # Benchmark general sobre level_*.txt
│       ├── heuristics_by_boxcount_traps_greedy.csv
│       ├── dead_square_showcase.csv
│       └── weighted_hungarian_counterexample_tradeoff.csv
├── tests/
│   ├── test_state.py           # Tests de Position y SokobanState
│   ├── test_successors.py      # Tests de apply_action, get_successors, is_goal
│   ├── test_search.py          # Tests de los 5 algoritmos
│   └── test_heuristics.py      # Tests de las 5 heuristicas
└── src/tp1_search/
    ├── cli.py                  # Entry point (tp1-search)
    ├── config.py               # Carga y validacion de configuracion TOML
    ├── types.py                # Position, Direction
    ├── search/
    │   ├── node.py             # SearchNode (arbol de busqueda)
    │   ├── frontier.py         # QueueFrontier, StackFrontier, PriorityFrontier
    │   ├── state_key.py        # Clave compacta (bytes) para el conjunto de visitados
    │   ├── bfs.py              # BFS
    │   ├── dfs.py              # DFS
    │   ├── iddfs.py            # IDDFS
    │   ├── greedy.py           # Greedy Best-First Search
    │   └── astar.py            # A*
    ├── sokoban/
    │   ├── board.py            # Board (paredes, objetivos, dead squares locales)
    │   ├── state.py            # SokobanState (jugador, cajas)
    │   ├── parser.py           # Parser de archivos .txt
    │   ├── actions.py          # apply_action (reglas de movimiento)
    │   ├── successors.py       # get_successors (generar movimientos validos)
    │   ├── goal.py             # is_goal (todas las cajas en objetivos)
    │   └── heuristics.py       # 5 heuristicas + registro HEURISTICS
    ├── output/
    │   ├── writer.py           # Serializador de replays JSON
    │   ├── animator_cli.py     # Entry point (tp1-animate)
    │   └── pygame_anim.py      # Renderizado Pygame
    └── metrics/
        └── result.py           # SearchResult (metricas de la busqueda)
```
