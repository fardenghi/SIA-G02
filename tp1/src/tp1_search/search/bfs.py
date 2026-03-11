import time

from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.sokoban.successors import get_successors
from tp1_search.sokoban.goal import is_goal
from tp1_search.search.node import SearchNode
from tp1_search.search.frontier import QueueFrontier
from tp1_search.metrics.result import SearchResult


def bfs(board: Board, initial_state: SokobanState) -> SearchResult:
    """Breadth-First Search (graph-search).

    Explora el árbol nivel por nivel usando una cola FIFO.
    Garantiza encontrar la solución con menor cantidad de pasos.
    Usa un conjunto de visitados para no expandir el mismo estado dos veces.
    """
    start_time = time.time()
    expanded = 0

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
    visited: set[SokobanState] = {initial_state}

    while not frontier.is_empty():
        node = frontier.pop()
        expanded += 1

        for child_state, action, step_cost in get_successors(board, node.state):
            if child_state in visited:
                continue

            visited.add(child_state)
            child_node = node.expand_child(child_state, action, step_cost)

            # Chequear goal al generar (no al expandir) — optimización de BFS
            if is_goal(board, child_state):
                return SearchResult(
                    success=True,
                    path=child_node.reconstruct_path(),
                    cost=child_node.path_cost,
                    expanded_nodes=expanded,
                    frontier_nodes=len(frontier),
                    time_elapsed=time.time() - start_time,
                )

            frontier.push(child_node)

    # Frontera vacía, no se encontró solución
    return SearchResult(
        success=False,
        path=[],
        cost=0,
        expanded_nodes=expanded,
        frontier_nodes=0,
        time_elapsed=time.time() - start_time,
    )
