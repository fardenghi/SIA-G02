import struct
import sys
import time

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.sokoban.successors import get_successors
from tp1_search.sokoban.goal import is_goal
from tp1_search.search.node import SearchNode
from tp1_search.search.frontier import QueueFrontier
from tp1_search.metrics.result import SearchResult

_LOG_INTERVAL = 10_000  # imprimir progreso cada N nodos expandidos


def _state_key(state: SokobanState, cols: int) -> bytes:
    """Convierte un estado a bytes para hashing eficiente en el visited set."""
    player_idx = state.player.row * cols + state.player.col
    box_idxs = sorted(b.row * cols + b.col for b in state.boxes)
    n = len(box_idxs)
    return struct.pack(f"{1 + n}H", player_idx, *box_idxs)


def _log(expanded: int, frontier: QueueFrontier, visited: set, start: float) -> None:
    elapsed = time.time() - start
    msg = (
        f"\r[BFS] expandidos: {expanded:>8,}  |  "
        f"frontera: {len(frontier):>8,}  |  "
        f"visitados: {len(visited):>8,}  |  "
        f"tiempo: {elapsed:6.1f}s"
    )
    sys.stderr.write(msg)
    sys.stderr.flush()


def bfs(board: Board, initial_state: SokobanState) -> SearchResult:
    """Breadth-First Search (graph-search).

    Explora el árbol nivel por nivel usando una cola FIFO.
    Garantiza encontrar la solución con menor cantidad de pasos.
    Usa un conjunto de visitados con bytes key para eficiencia de hashing.
    Poda estados con cajas en dead squares (via get_successors).
    """
    start_time = time.time()
    expanded = 0
    cols = board.cols

    root = SearchNode.root(initial_state)

    # Chequear si el estado inicial ya es goal
    if is_goal(board, initial_state):
        return SearchResult(
            success=True,
            path=root.reconstruct_path(),
            cost=root.path_cost,
            expanded_nodes=expanded,
            frontier_nodes=0,
            time_elapsed=time.time() - start_time,
        )

    frontier = QueueFrontier()
    frontier.push(root)
    visited: set[bytes] = {_state_key(initial_state, cols)}

    print(
        f"[BFS] inicio — tablero {board.rows}×{board.cols}, "
        f"{len(board.goals)} objetivo(s)",
        file=sys.stderr,
    )

    while not frontier.is_empty():
        node = frontier.pop()
        expanded += 1

        if expanded % _LOG_INTERVAL == 0:
            _log(expanded, frontier, visited, start_time)

        for child_state, action, step_cost in get_successors(board, node.state):
            key = _state_key(child_state, cols)
            if key in visited:
                continue

            visited.add(key)
            child_node = node.expand_child(child_state, action, step_cost)

            # Chequear goal al generar (no al expandir) — optimización de BFS
            if is_goal(board, child_state):
                elapsed = time.time() - start_time
                sys.stderr.write(
                    f"\r[BFS] ¡solución encontrada! "
                    f"expandidos: {expanded:,}  |  "
                    f"costo: {child_node.path_cost}  |  "
                    f"tiempo: {elapsed:.3f}s\n"
                )
                sys.stderr.flush()
                return SearchResult(
                    success=True,
                    path=child_node.reconstruct_path(),
                    cost=child_node.path_cost,
                    expanded_nodes=expanded,
                    frontier_nodes=len(frontier),
                    time_elapsed=elapsed,
                )

            frontier.push(child_node)

    elapsed = time.time() - start_time
    sys.stderr.write(
        f"\r[BFS] sin solución. expandidos: {expanded:,}  |  tiempo: {elapsed:.3f}s\n"
    )
    sys.stderr.flush()
    return SearchResult(
        success=False,
        path=[],
        cost=0,
        expanded_nodes=expanded,
        frontier_nodes=0,
        time_elapsed=elapsed,
    )
