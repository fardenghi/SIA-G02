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
| **IDDFS** | Iterative Deepening DFS. Ejecuta DFS con limite de profundidad incremental (0, 1, 2, ...). Usa un conjunto de visitados fresco por cada iteracion. | Si (minima profundidad) | Si |

### Informados (requieren heuristica)

| Algoritmo | Descripcion | Optimo | Completo |
|-----------|-------------|--------|----------|
| **Greedy** | Best-First Search. Expande el nodo con menor h(n). Usa frontera de prioridad (min-heap). Verifica objetivo al expandir. | No | Si |
| **A\*** | Expande el nodo con menor f(n) = g(n) + h(n). Usa frontera de prioridad (min-heap). Verifica objetivo al expandir. | Si (con heuristica admisible) | Si |

Todos los algoritmos usan **graph-search** con un conjunto de visitados para evitar re-explorar estados.

---

## Heuristicas disponibles

Para los algoritmos informados (Greedy y A*) se implementaron 3 heuristicas:

| Heuristica | Valor en config | Admisible | Descripcion |
|------------|-----------------|-----------|-------------|
| **Manhattan** | `manhattan` | Si | Suma de distancias Manhattan desde cada caja a su objetivo mas cercano. Es la heuristica mas informativa de las tres. |
| **Euclidean** | `euclidean` | Si | Suma de distancias Euclidianas desde cada caja a su objetivo mas cercano. Siempre da valores <= Manhattan, por lo que guia menos la busqueda. |
| **Dead Square** | `dead_square` | Si | Retorna infinito si alguna caja esta en un "dead square" (celda desde la cual ninguna caja puede llegar a ningun objetivo), sino retorna 0. Funciona como mecanismo de poda mas que como estimador de distancia. |

### Deteccion de dead squares

La clase `Board` precomputa los dead squares al inicializarse usando un **BFS reverso desde todos los objetivos**. Una celda es "viva" si una caja colocada ahi podria eventualmente llegar a algun objetivo mediante empujes validos. Cualquier celda que no sea pared ni sea "viva" es un dead square. Esta informacion se almacena en un array booleano de NumPy para consultas O(1).

---

## Tableros disponibles

Los tableros estan en `boards/sokoban/` y van de menor a mayor dificultad:

| Tablero | Tamanio | Cajas | Dificultad | Notas |
|---------|---------|-------|------------|-------|
| `level_01.txt` | 6x6 | 1 | Trivial | Solucion optima: 6 pasos |
| `level_02.txt` | 7x9 | 3 | Media | |
| `level_03.txt` | 10x11 | 7 | Muy dificil | 2 cajas ya estan en objetivo. Probablemente solo resoluble con `dead_square` dentro del timeout |
| `level_04.txt` | 13x13 | 2 | Media | |
| `level_05.txt` | 9x5 | 2 | Irresoluble | Todos los algoritmos deben retornar FAILURE |
| `level_06.txt` | 5x6 | 1 | Trivial | Solucion optima: 4 pasos |
| `level_07.txt` | 6x8 | 2 | Media | Solucion optima: 13 pasos |

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
| `heuristic` | String o array con `manhattan`, `euclidean`, `dead_square` |

### Validaciones

- El algoritmo debe ser uno de los 5 listados
- El archivo de tablero debe existir
- Los algoritmos informados (`greedy`, `astar`) **deben** tener una heuristica
- Si `heuristic` es array, se usa una heuristica compuesta: `max(h1, ..., hn)`
- Cada heuristica del array debe ser una de las 3 listadas

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

## Instructivo: correr cada algoritmo con todas sus variantes

### Resumen de combinaciones

| Algoritmo | Heuristicas | Niveles | Total ejecuciones |
|-----------|-------------|---------|-------------------|
| BFS | — | 7 | 7 |
| DFS | — | 7 | 7 |
| IDDFS | — | 7 | 7 |
| Greedy | 3 | 7 | 21 |
| A* | 3 | 7 | 21 |
| **Total** | | | **63** |

---

### 1. BFS (sin heuristica) — 7 ejecuciones

Crear un archivo TOML por nivel. Ejemplo para level_01:

```toml
# configs/sokoban/bfs_l1.toml
[search]
algorithm = "bfs"
board = "boards/sokoban/level_01.txt"
```

Ejecutar:

```bash
uv run tp1-search configs/sokoban/bfs_l1.toml
```

Repetir cambiando `board` para cada nivel (`level_01.txt` a `level_07.txt`).

---

### 2. DFS (sin heuristica) — 7 ejecuciones

```toml
# configs/sokoban/dfs_l1.toml
[search]
algorithm = "dfs"
board = "boards/sokoban/level_01.txt"
```

```bash
uv run tp1-search configs/sokoban/dfs_l1.toml
```

Repetir para cada nivel.

---

### 3. IDDFS (sin heuristica) — 7 ejecuciones

```toml
# configs/sokoban/iddfs_l1.toml
[search]
algorithm = "iddfs"
board = "boards/sokoban/level_01.txt"
```

```bash
uv run tp1-search configs/sokoban/iddfs_l1.toml
```

Repetir para cada nivel.

---

### 4. Greedy (requiere heuristica) — 21 ejecuciones

3 heuristicas x 7 niveles. Ejemplo para manhattan + level_01:

```toml
# configs/sokoban/greedy_manhattan_l1.toml
[search]
algorithm = "greedy"
board = "boards/sokoban/level_01.txt"
heuristic = "manhattan"
```

```bash
uv run tp1-search configs/sokoban/greedy_manhattan_l1.toml
```

Variantes de heuristica para el mismo nivel:

```toml
# configs/sokoban/greedy_euclidean_l1.toml
[search]
algorithm = "greedy"
board = "boards/sokoban/level_01.txt"
heuristic = "euclidean"
```

```toml
# configs/sokoban/greedy_dead_square_l1.toml
[search]
algorithm = "greedy"
board = "boards/sokoban/level_01.txt"
heuristic = "dead_square"
```

Repetir las 3 heuristicas para cada nivel (`level_01` a `level_07`).

---

### 5. A* (requiere heuristica) — 21 ejecuciones

3 heuristicas x 7 niveles. Ejemplo para manhattan + level_01:

```toml
# configs/sokoban/astar_manhattan_l1.toml
[search]
algorithm = "astar"
board = "boards/sokoban/level_01.txt"
heuristic = "manhattan"
```

```bash
uv run tp1-search configs/sokoban/astar_manhattan_l1.toml
```

Variantes de heuristica para el mismo nivel:

```toml
# configs/sokoban/astar_euclidean_l1.toml
[search]
algorithm = "astar"
board = "boards/sokoban/level_01.txt"
heuristic = "euclidean"
```

```toml
# configs/sokoban/astar_dead_square_l1.toml
[search]
algorithm = "astar"
board = "boards/sokoban/level_01.txt"
heuristic = "dead_square"
```

Repetir las 3 heuristicas para cada nivel (`level_01` a `level_07`).

---

## Ejecucion masiva: benchmark automatico

En lugar de crear 63 archivos TOML manualmente, el script `run_batch.py` ejecuta **todas las combinaciones** automaticamente:

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
| `heuristic` | Heuristica usada (vacio para no informados) |
| `success` | Si encontro solucion |
| `cost` | Cantidad de pasos de la solucion |
| `expanded_nodes` | Nodos expandidos |
| `frontier_nodes` | Nodos restantes en la frontera |
| `time_elapsed` | Tiempo en segundos |
| `timed_out` | Si se excedio el timeout |

---

## Generacion de graficos

```bash
uv run python scripts/make_plots.py --input results/benchmark/benchmark.csv --outdir results/benchmark/plots
```

**Graficos generados:**

- Bar charts agrupados (nodos expandidos, tiempo, costo de solucion)
- Heatmap de algoritmo x nivel (nodos expandidos)
- Comparacion de heuristicas por algoritmo (A* y Greedy)
- Dashboard interactivo HTML (Plotly)

---

## Animacion de soluciones

### Paso 1: guardar el replay

```bash
uv run tp1-search configs/sokoban/bfs_l1.toml --save-replay
```

Esto genera un archivo JSON en `results/raw/`.

### Paso 2: reproducir la animacion

```bash
uv run tp1-animate results/raw/<archivo_generado>.json
```

**Parametros opcionales:**

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `--speed` | 2.0 | Frames por segundo (velocidad inicial) |
| `--cell-size` | 64 | Tamanio en pixeles de cada celda |

Ejemplo:

```bash
uv run tp1-animate results/raw/replay.json --speed 3 --cell-size 64
```

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
- **level_03 es el mas dificil** (7 cajas): probablemente solo sea resoluble dentro del timeout con la heuristica `dead_square` o combinaciones de A* con `manhattan`.
- **Manhattan vs Euclidean**: Manhattan siempre da valores >= Euclidean para el mismo estado, por lo que guia mejor la busqueda y expande menos nodos.
- **Dead square**: no estima distancia (solo retorna 0 o infinito), pero poda estados irrecuperables de forma muy efectiva.
- **A\* con manhattan** deberia ser la combinacion mas eficiente en general para encontrar soluciones optimas.
- Los algoritmos no informados (BFS, DFS, IDDFS) no usan heuristica y su rendimiento depende unicamente de la estructura del espacio de estados.

---

## Estructura del proyecto

```
tp1/
├── configs/sokoban/            # Archivos de configuracion TOML
├── boards/sokoban/             # Tableros de Sokoban (.txt)
│   ├── level_01.txt            # 6x6, 1 caja (trivial)
│   ├── level_02.txt            # 7x9, 3 cajas
│   ├── level_03.txt            # 10x11, 7 cajas (muy dificil)
│   ├── level_04.txt            # 13x13, 2 cajas
│   ├── level_05.txt            # 9x5, 2 cajas (irresoluble)
│   ├── level_06.txt            # 5x6, 1 caja (trivial)
│   └── level_07.txt            # 6x8, 2 cajas
├── scripts/
│   ├── run_batch.py            # Benchmark: todas las combinaciones -> CSV
│   └── make_plots.py           # Generacion de graficos (matplotlib + plotly)
├── results/
│   ├── raw/                    # Replays JSON (generados con --save-replay)
│   └── benchmark/
│       ├── benchmark.csv       # Resultados del benchmark
│       └── plots/              # Graficos PNG + dashboard HTML
├── tests/
│   ├── test_state.py           # Tests de Position y SokobanState
│   ├── test_successors.py      # Tests de apply_action, get_successors, is_goal
│   ├── test_search.py          # Tests de los 5 algoritmos
│   └── test_heuristics.py      # Tests de las 3 heuristicas
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
    │   ├── board.py            # Board (paredes, objetivos, dead squares)
    │   ├── state.py            # SokobanState (jugador, cajas)
    │   ├── parser.py           # Parser de archivos .txt
    │   ├── actions.py          # apply_action (reglas de movimiento)
    │   ├── successors.py       # get_successors (generar movimientos validos)
    │   ├── goal.py             # is_goal (todas las cajas en objetivos)
    │   └── heuristics.py       # 3 heuristicas + registro HEURISTICS
    ├── output/
    │   ├── writer.py           # Serializador de replays JSON
    │   ├── animator_cli.py     # Entry point (tp1-animate)
    │   └── pygame_anim.py      # Renderizado Pygame
    └── metrics/
        └── result.py           # SearchResult (metricas de la busqueda)
```
