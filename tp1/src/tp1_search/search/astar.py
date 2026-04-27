import sys
import time

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.sokoban.successors import get_successors
from tp1_search.sokoban.goal import is_goal
from tp1_search.sokoban.heuristics import HeuristicFn
from tp1_search.search.node import SearchNode
from tp1_search.search.frontier import PriorityFrontier
from tp1_search.search.state_key import state_key
from tp1_search.metrics.result import SearchResult

_LOG_INTERVAL = 10_000


def _log(
    expanded: int, frontier: PriorityFrontier, best_g: dict[bytes, int], start: float
) -> None:
    elapsed = time.time() - start
    msg = (
        f"\r[A*] expandidos: {expanded:>8,}  |  "
        f"frontera: {len(frontier):>8,}  |  "
        f"mejores g: {len(best_g):>8,}  |  "
        f"tiempo: {elapsed:6.1f}s"
    )
    sys.stderr.write(msg)
    sys.stderr.flush()


def astar(
    board: Board, initial_state: SokobanState, heuristic: HeuristicFn
) -> SearchResult:
    """A* Search (graph-search).

    Expande siempre el nodo con menor f(n) = g(n) + h(n).
    Garantiza optimalidad si la heurística es admisible.
    Usa PriorityFrontier con prioridad = g(n) + h(n).
    """
    start_time = time.time()
    expanded = 0
    max_frontier_size = 0
    cols = board.cols

    root = SearchNode.root(initial_state)

    if is_goal(board, initial_state):
        return SearchResult(
            success=True,
            path=root.reconstruct_path(),
            cost=root.path_cost,
            expanded_nodes=expanded,
            frontier_nodes=0,
            max_frontier_nodes=max_frontier_size,
            time_elapsed=time.time() - start_time,
        )

    frontier = PriorityFrontier()
    h_root = heuristic(board, initial_state)
    frontier.push(root, priority=root.path_cost + h_root)
    max_frontier_size = max(max_frontier_size, len(frontier))

    # g-score mínimo conocido por estado para preservar optimalidad
    # en graph-search (permite reabrir estados si aparece un mejor camino).
    best_g: dict[bytes, int] = {state_key(initial_state, cols): 0}

    print(
        f"[A*] inicio — tablero {board.rows}×{board.cols}, "
        f"{len(board.goals)} objetivo(s)",
        file=sys.stderr,
    )

    while not frontier.is_empty():
        node = frontier.pop()

        node_key = state_key(node.state, cols)
        # Entrada vieja en frontera: ya existe un camino más barato al mismo estado.
        if node.path_cost > best_g.get(node_key, float("inf")):
            continue

        expanded += 1

        if expanded % _LOG_INTERVAL == 0:
            _log(expanded, frontier, best_g, start_time)

        # Chequear goal al expandir — en A* la optimalidad se garantiza
        # al expandir, no al generar
        if is_goal(board, node.state):
            elapsed = time.time() - start_time
            sys.stderr.write(
                f"\r[A*] ¡solución encontrada! "
                f"expandidos: {expanded:,}  |  "
                f"costo: {node.path_cost}  |  "
                f"tiempo: {elapsed:.3f}s                   \n"
            )
            sys.stderr.flush()
            return SearchResult(
                success=True,
                path=node.reconstruct_path(),
                cost=node.path_cost,
                expanded_nodes=expanded,
                frontier_nodes=len(frontier),
                max_frontier_nodes=max_frontier_size,
                time_elapsed=elapsed,
            )

        for child_state, action, step_cost in get_successors(board, node.state):
            key = state_key(child_state, cols)
            child_node = node.expand_child(child_state, action, step_cost)
            new_g = child_node.path_cost

            prev_g = best_g.get(key)
            if prev_g is not None and prev_g <= new_g:
                continue

            best_g[key] = new_g
            h = heuristic(board, child_state)
            f = new_g + h
            frontier.push(child_node, priority=f)
            max_frontier_size = max(max_frontier_size, len(frontier))

    elapsed = time.time() - start_time
    sys.stderr.write(
        f"\r[A*] sin solución. expandidos: {expanded:,}  |  tiempo: {elapsed:.3f}s\n"
    )
    sys.stderr.flush()
    return SearchResult(
        success=False,
        path=[],
        cost=0,
        expanded_nodes=expanded,
        frontier_nodes=0,
        max_frontier_nodes=max_frontier_size,
        time_elapsed=elapsed,
    )
