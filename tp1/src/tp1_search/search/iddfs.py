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
    """Iterative Deepening Depth-First Search con graph-search por iteración.

    Incrementa el límite de profundidad en cada iteración y ejecuta un DFS
    limitado por profundidad. En cada iteración reinicia un diccionario de
    visitados para evitar ciclos y permitir reabrir estados si reaparecen a
    menor profundidad.

    Con costos uniformes, garantiza encontrar la solución de menor
    profundidad. Como mantiene visitados por iteración, esta variante no usa
    la memoria mínima del IDDFS clásico puro: el consumo crece con los estados
    alcanzados dentro de la iteración actual.
    """
    start_time = time.time()
    total_expanded = 0
    total_max_frontier = 0
    cols = board.cols

    root = SearchNode.root(initial_state)

    if is_goal(board, initial_state):
        return SearchResult(
            success=True,
            path=root.reconstruct_path(),
            cost=root.path_cost,
            expanded_nodes=total_expanded,
            frontier_nodes=0,
            max_frontier_nodes=0,
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
        result, expanded, cutoff_occurred, frontier_nodes, max_frontier_nodes = _depth_limited_search(
            board, root, depth_limit, cols
        )
        total_expanded += expanded
        total_max_frontier = max(total_max_frontier, max_frontier_nodes)

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
                f"tiempo: {elapsed:.3f}s                   \n"
            )
            sys.stderr.flush()
            return SearchResult(
                success=True,
                path=result.reconstruct_path(),
                cost=result.path_cost,
                expanded_nodes=total_expanded,
                frontier_nodes=frontier_nodes,
                max_frontier_nodes=total_max_frontier,
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
                frontier_nodes=frontier_nodes,
                max_frontier_nodes=total_max_frontier,
                time_elapsed=elapsed,
            )

        depth_limit += 1


def _depth_limited_search(
    board: Board,
    root: SearchNode,
    limit: int,
    cols: int,
) -> tuple[SearchNode | None, int, bool, int, int]:
    """DFS limitado por profundidad.

    Returns:
        (solution_node, expanded_count, cutoff_occurred, frontier_nodes, max_frontier_nodes)
        - solution_node: nodo goal si se encontró, None si no
        - expanded_count: nodos expandidos en esta iteración
        - cutoff_occurred: True si algún nodo fue cortado por el límite
        - frontier_nodes: nodos pendientes en la pila al terminar la iteración
        - max_frontier_nodes: máximo de nodos pendientes alcanzado en la iteración
    """
    expanded = 0
    cutoff_occurred = False

    stack: list[SearchNode] = [root]
    max_stack_size = len(stack)
    # Visited dict: state_key -> menor profundidad a la que fue alcanzado.
    # Permite re-explorar un estado si se llega por un camino mas corto,
    # lo cual es necesario para garantizar optimalidad en IDDFS.
    visited: dict[bytes, int] = {state_key(root.state, cols): 0}

    while stack:
        node = stack.pop()

        if node.depth >= limit:
            cutoff_occurred = True
            continue

        expanded += 1

        for child_state, action, step_cost in get_successors(board, node.state):
            key = state_key(child_state, cols)
            child_depth = node.depth + 1

            prev_depth = visited.get(key)
            if prev_depth is not None and prev_depth <= child_depth:
                continue

            visited[key] = child_depth
            child_node = node.expand_child(child_state, action, step_cost)

            if is_goal(board, child_state):
                return child_node, expanded, False, len(stack), max_stack_size

            stack.append(child_node)
            max_stack_size = max(max_stack_size, len(stack))

    return None, expanded, cutoff_occurred, len(stack), max_stack_size
