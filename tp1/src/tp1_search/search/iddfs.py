import sys
import time

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.sokoban.successors import get_successors
from tp1_search.sokoban.goal import is_goal
from tp1_search.search.node import SearchNode
from tp1_search.search.state_key import state_key
from tp1_search.metrics.result import SearchResult

_LOG_INTERVAL = 10_000


def iddfs(board: Board, initial_state: SokobanState) -> SearchResult:
    """Iterative Deepening Depth-First Search.

    Combina la completitud de BFS con el uso de memoria de DFS.
    Incrementa el límite de profundidad en cada iteración y ejecuta
    un DFS limitado por profundidad (con visited set por iteración).
    Garantiza encontrar la solución con menor profundidad.
    """
    start_time = time.time()
    total_expanded = 0
    cols = board.cols

    root = SearchNode.root(initial_state)

    if is_goal(board, initial_state):
        return SearchResult(
            success=True,
            path=root.reconstruct_path(),
            cost=root.path_cost,
            expanded_nodes=total_expanded,
            frontier_nodes=0,
            time_elapsed=time.time() - start_time,
        )

    print(
        f"[IDDFS] inicio — tablero {board.rows}×{board.cols}, "
        f"{len(board.goals)} objetivo(s)",
        file=sys.stderr,
    )

    depth_limit = 0

    while True:
        # DFS limitado con visited set fresco para esta iteración
        result, expanded, cutoff_occurred = _depth_limited_search(
            board, root, depth_limit, cols
        )
        total_expanded += expanded

        if total_expanded % _LOG_INTERVAL < expanded:
            elapsed = time.time() - start_time
            sys.stderr.write(
                f"\r[IDDFS] profundidad: {depth_limit:>4}  |  "
                f"expandidos total: {total_expanded:>8,}  |  "
                f"tiempo: {elapsed:6.1f}s"
            )
            sys.stderr.flush()

        if result is not None:
            # Encontró solución
            elapsed = time.time() - start_time
            sys.stderr.write(
                f"\r[IDDFS] ¡solución encontrada! "
                f"profundidad: {depth_limit}  |  "
                f"expandidos: {total_expanded:,}  |  "
                f"costo: {result.path_cost}  |  "
                f"tiempo: {elapsed:.3f}s\n"
            )
            sys.stderr.flush()
            return SearchResult(
                success=True,
                path=result.reconstruct_path(),
                cost=result.path_cost,
                expanded_nodes=total_expanded,
                frontier_nodes=0,
                time_elapsed=elapsed,
            )

        if not cutoff_occurred:
            # Exploró todo el espacio sin encontrar solución ni cortar
            elapsed = time.time() - start_time
            sys.stderr.write(
                f"\r[IDDFS] sin solución. "
                f"expandidos: {total_expanded:,}  |  "
                f"tiempo: {elapsed:.3f}s\n"
            )
            sys.stderr.flush()
            return SearchResult(
                success=False,
                path=[],
                cost=0,
                expanded_nodes=total_expanded,
                frontier_nodes=0,
                time_elapsed=elapsed,
            )

        depth_limit += 1


def _depth_limited_search(
    board: Board,
    root: SearchNode,
    limit: int,
    cols: int,
) -> tuple[SearchNode | None, int, bool]:
    """DFS limitado por profundidad.

    Returns:
        (solution_node, expanded_count, cutoff_occurred)
        - solution_node: nodo goal si se encontró, None si no
        - expanded_count: nodos expandidos en esta iteración
        - cutoff_occurred: True si algún nodo fue cortado por el límite
    """
    expanded = 0
    cutoff_occurred = False

    # Stack iterativo: (nodo, ya_expandido)
    stack: list[SearchNode] = [root]
    # Visited set local a esta iteración para graph-search
    visited: set[bytes] = {state_key(root.state, cols)}

    while stack:
        node = stack.pop()

        if node.depth >= limit:
            cutoff_occurred = True
            continue

        expanded += 1

        for child_state, action, step_cost in get_successors(board, node.state):
            key = state_key(child_state, cols)
            if key in visited:
                continue

            visited.add(key)
            child_node = node.expand_child(child_state, action, step_cost)

            if is_goal(board, child_state):
                return child_node, expanded, False

            stack.append(child_node)

    return None, expanded, cutoff_occurred
