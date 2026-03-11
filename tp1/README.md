# TP1 — Métodos de Búsqueda

**Sistemas de Inteligencia Artificial — ITBA 2026 1C**

Motor de búsqueda para resolver tableros de **Sokoban** implementado en Python.

---

## Descripción

El motor lee un tablero de Sokoban desde un archivo de texto, ejecuta un algoritmo de búsqueda y reporta la solución encontrada junto con métricas de rendimiento.

**Algoritmos implementados:**
- BFS — Breadth-First Search

**Próximos algoritmos:**
- DFS, Greedy, A*, IDDFS (opcional)

---

## Requisitos

- [uv](https://docs.astral.sh/uv/) — gestor de paquetes y entornos Python
- Python 3.12 (se descarga automáticamente con `uv`)

---

## Instalación

```bash
# Clonar el repositorio y entrar al directorio tp1
cd tp1

# Instalar dependencias y crear el entorno virtual
uv sync
```

---

## Uso

### Ejecutar con un archivo de configuración TOML

```bash
uv run tp1-search configs/sokoban/bfs_simple.toml
```

Salida de ejemplo:

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

### Alternativa: ejecutar como módulo

```bash
uv run python -m tp1_search.cli configs/sokoban/bfs_simple.toml
```

---

## Formato de configuración TOML

Los archivos de configuración viven en `configs/`. Estructura mínima:

```toml
[search]
algorithm = "bfs"
board = "boards/sokoban/level_01.txt"
```

Para algoritmos informados (cuando estén implementados):

```toml
[search]
algorithm = "astar"
board = "boards/sokoban/level_01.txt"
heuristic = "manhattan"
```

**Algoritmos válidos:** `bfs`, `dfs`, `greedy`, `astar`, `iddfs`

---

## Formato de tableros

Los tableros se definen en archivos `.txt` dentro de `boards/sokoban/`.
Cada celda se representa con un símbolo:

| Símbolo | Significado |
|---------|-------------|
| `#` | Pared |
| `@` | Jugador |
| `+` | Jugador sobre objetivo |
| `$` | Caja |
| `*` | Caja sobre objetivo |
| `.` | Objetivo |
| ` ` | Celda vacía |

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
- No se admiten símbolos desconocidos

---

## Estructura del proyecto

```
tp1/
├── configs/sokoban/        # Archivos de configuración TOML
├── boards/sokoban/         # Tableros de Sokoban (.txt)
├── results/                # Resultados generados (no versionados)
│   ├── raw/
│   ├── plots/
│   └── animations/
├── tests/                  # Tests con pytest
└── src/tp1_search/
    ├── types.py            # Position, Direction
    ├── config.py           # Carga de configuración TOML
    ├── cli.py              # Entry point (tp1-search)
    ├── search/
    │   ├── node.py         # SearchNode (árbol de búsqueda)
    │   ├── frontier.py     # Fronteras (QueueFrontier para BFS)
    │   └── bfs.py          # Algoritmo BFS
    ├── sokoban/
    │   ├── board.py        # Board (parte estática: paredes, objetivos)
    │   ├── state.py        # SokobanState (parte dinámica: jugador, cajas)
    │   ├── parser.py       # Parser de archivos .txt
    │   ├── actions.py      # apply_action (reglas de movimiento)
    │   ├── successors.py   # get_successors
    │   └── goal.py         # is_goal
    └── metrics/
        └── result.py       # SearchResult (métricas de la búsqueda)
```

---

## Tests

```bash
uv run pytest
```

```bash
uv run pytest -v   # con detalle por test
```

**Cobertura actual:** 29 tests cubriendo estado, sucesores y búsqueda BFS.

---

## Métricas reportadas

| Métrica | Descripción |
|---------|-------------|
| Resultado | EXITO o FRACASO |
| Costo | Cantidad de pasos de la solución |
| Nodos expandidos | Cuántos estados fueron procesados |
| Nodos en frontera | Cuántos nodos quedaron sin explorar al terminar |
| Camino | Secuencia de direcciones (UP, DOWN, LEFT, RIGHT) |
| Tiempo | Duración del algoritmo en segundos |
