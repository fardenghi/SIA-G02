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


def _log(expanded: int, frontier: PriorityFrontier, visited: set, start: float) -> None:
    elapsed = time.time() - start
    msg = (
        f"\r[Greedy] expandidos: {expanded:>8,}  |  "
        f"frontera: {len(frontier):>8,}  |  "
        f"visitados: {len(visited):>8,}  |  "
        f"tiempo: {elapsed:6.1f}s"
    )
    sys.stderr.write(msg)
    sys.stderr.flush()


def greedy(
    board: Board, initial_state: SokobanState, heuristic: HeuristicFn
) -> SearchResult:
    """Greedy Best-First Search (graph-search).

    Expande siempre el nodo con menor valor heurístico h(n).
    No garantiza optimalidad.
    Usa PriorityFrontier con prioridad = h(n).
    """
    start_time = time.time()
    expanded = 0
    cols = board.cols

    root = SearchNode.root(initial_state)

    if is_goal(board, initial_state):
        return SearchResult(
            success=True,
            path=root.reconstruct_path(),
            cost=root.path_cost,
            expanded_nodes=expanded,
            frontier_nodes=0,
            time_elapsed=time.time() - start_time,
        )

    frontier = PriorityFrontier()
    h_root = heuristic(board, initial_state)
    frontier.push(root, priority=h_root)
    visited: set[bytes] = {state_key(initial_state, cols)}

    print(
        f"[Greedy] inicio — tablero {board.rows}×{board.cols}, "
        f"{len(board.goals)} objetivo(s)",
        file=sys.stderr,
    )

    while not frontier.is_empty():
        node = frontier.pop()
        expanded += 1

        if expanded % _LOG_INTERVAL == 0:
            _log(expanded, frontier, visited, start_time)

        # Chequear goal al expandir (no al generar) — en Greedy el nodo
        # con menor h puede no ser el primero generado
        if is_goal(board, node.state):
            elapsed = time.time() - start_time
            sys.stderr.write(
                f"\r[Greedy] ¡solución encontrada! "
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
                time_elapsed=elapsed,
            )

        for child_state, action, step_cost in get_successors(board, node.state):
            key = state_key(child_state, cols)
            if key in visited:
                continue

            visited.add(key)
            child_node = node.expand_child(child_state, action, step_cost)
            h = heuristic(board, child_state)
            frontier.push(child_node, priority=h)

    elapsed = time.time() - start_time
    sys.stderr.write(
        f"\r[Greedy] sin solución. expandidos: {expanded:,}  |  "
        f"tiempo: {elapsed:.3f}s\n"
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
