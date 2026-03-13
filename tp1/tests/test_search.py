from tp1_search.sokoban.parser import parse_board_string
from tp1_search.sokoban.actions import apply_action
from tp1_search.sokoban.goal import is_goal
from tp1_search.search.bfs import bfs
from tp1_search.search.dfs import dfs
from tp1_search.search.iddfs import iddfs
from tp1_search.search.greedy import greedy
from tp1_search.search.astar import astar
from tp1_search.sokoban.heuristics import (
    manhattan_heuristic,
    euclidean_heuristic,
    dead_square_heuristic,
)


def _simple_board():
    """Tablero simple resoluble: 1 caja, 1 goal.

    #####
    #   #
    # $ #
    # . #
    # @ #
    #####
    """
    return parse_board_string("#####\n#   #\n# $ #\n# . #\n# @ #\n#####")


def _already_solved():
    """Tablero donde la caja ya está en el goal.

    ####
    # *#
    # @#
    ####
    """
    return parse_board_string("####\n# *#\n# @#\n####")


def _impossible_board():
    """Tablero imposible: caja en esquina, goal en otro lado.

    #####
    #$  #
    #   #
    #  .#
    # @ #
    #####
    """
    return parse_board_string("#####\n#$  #\n#   #\n#  .#\n# @ #\n#####")


def _astar_counterexample_board():
    """Tablero donde A* con visited-al-generar puede dar subóptimo.

    Este caso regresiona a costo 22 con la versión incorrecta de A* y
    debe devolver 20 (igual que BFS/IDDFS) con la implementación correcta.
    """
    return parse_board_string(
        "########\n#@     #\n#  $   #\n#     .#\n#  $####\n# .  # #\n########"
    )


def _validate_path(board, initial_state, result):
    """Helper: verifica que el camino retornado lleva al goal."""
    current = initial_state
    for direction in result.path:
        current = apply_action(board, current, direction)
        assert current is not None
    assert is_goal(board, current)


# =========================================================================
# BFS
# =========================================================================


class TestBFS:
    def test_finds_solution(self):
        board, state = _simple_board()
        result = bfs(board, state)
        assert result.success is True
        assert result.cost > 0
        assert len(result.path) == result.cost

    def test_solution_is_valid(self):
        """Verificar que el camino retornado realmente lleva al goal."""
        board, state = _simple_board()
        result = bfs(board, state)
        assert result.success
        _validate_path(board, state, result)

    def test_solution_is_optimal(self):
        """BFS debe encontrar el camino más corto."""
        board, state = _simple_board()
        result = bfs(board, state)
        # La solución óptima de este tablero es 6 pasos
        assert result.cost == 6

    def test_already_solved(self):
        board, state = _already_solved()
        result = bfs(board, state)
        assert result.success is True
        assert result.cost == 0
        assert result.path == []
        assert result.expanded_nodes == 0

    def test_impossible_returns_failure(self):
        board, state = _impossible_board()
        result = bfs(board, state)
        assert result.success is False
        assert result.path == []

    def test_metrics_populated(self):
        board, state = _simple_board()
        result = bfs(board, state)
        assert result.expanded_nodes > 0
        assert result.time_elapsed >= 0


# =========================================================================
# DFS
# =========================================================================


class TestDFS:
    def test_finds_solution(self):
        board, state = _simple_board()
        result = dfs(board, state)
        assert result.success is True
        assert result.cost > 0
        assert len(result.path) == result.cost

    def test_solution_is_valid(self):
        board, state = _simple_board()
        result = dfs(board, state)
        assert result.success
        _validate_path(board, state, result)

    def test_already_solved(self):
        board, state = _already_solved()
        result = dfs(board, state)
        assert result.success is True
        assert result.cost == 0
        assert result.path == []

    def test_impossible_returns_failure(self):
        board, state = _impossible_board()
        result = dfs(board, state)
        assert result.success is False
        assert result.path == []

    def test_metrics_populated(self):
        board, state = _simple_board()
        result = dfs(board, state)
        assert result.expanded_nodes > 0
        assert result.time_elapsed >= 0


# =========================================================================
# IDDFS
# =========================================================================


class TestIDDFS:
    def test_finds_solution(self):
        board, state = _simple_board()
        result = iddfs(board, state)
        assert result.success is True
        assert result.cost > 0
        assert len(result.path) == result.cost

    def test_solution_is_valid(self):
        board, state = _simple_board()
        result = iddfs(board, state)
        assert result.success
        _validate_path(board, state, result)

    def test_solution_is_optimal(self):
        """IDDFS debe encontrar la solución con menor profundidad."""
        board, state = _simple_board()
        result = iddfs(board, state)
        assert result.cost == 6

    def test_already_solved(self):
        board, state = _already_solved()
        result = iddfs(board, state)
        assert result.success is True
        assert result.cost == 0
        assert result.path == []

    def test_impossible_returns_failure(self):
        board, state = _impossible_board()
        result = iddfs(board, state)
        assert result.success is False
        assert result.path == []

    def test_metrics_populated(self):
        board, state = _simple_board()
        result = iddfs(board, state)
        assert result.expanded_nodes > 0
        assert result.time_elapsed >= 0


# =========================================================================
# Greedy (con manhattan)
# =========================================================================


class TestGreedy:
    def test_finds_solution(self):
        board, state = _simple_board()
        result = greedy(board, state, manhattan_heuristic)
        assert result.success is True
        assert result.cost > 0

    def test_solution_is_valid(self):
        board, state = _simple_board()
        result = greedy(board, state, manhattan_heuristic)
        assert result.success
        _validate_path(board, state, result)

    def test_already_solved(self):
        board, state = _already_solved()
        result = greedy(board, state, manhattan_heuristic)
        assert result.success is True
        assert result.cost == 0
        assert result.path == []

    def test_impossible_returns_failure(self):
        board, state = _impossible_board()
        result = greedy(board, state, manhattan_heuristic)
        assert result.success is False
        assert result.path == []

    def test_euclidean_finds_solution(self):
        board, state = _simple_board()
        result = greedy(board, state, euclidean_heuristic)
        assert result.success is True
        _validate_path(board, state, result)

    def test_dead_square_finds_solution(self):
        board, state = _simple_board()
        result = greedy(board, state, dead_square_heuristic)
        assert result.success is True
        _validate_path(board, state, result)

    def test_metrics_populated(self):
        board, state = _simple_board()
        result = greedy(board, state, manhattan_heuristic)
        assert result.expanded_nodes > 0
        assert result.time_elapsed >= 0


# =========================================================================
# A*
# =========================================================================


class TestAStar:
    def test_finds_solution(self):
        board, state = _simple_board()
        result = astar(board, state, manhattan_heuristic)
        assert result.success is True
        assert result.cost > 0

    def test_solution_is_valid(self):
        board, state = _simple_board()
        result = astar(board, state, manhattan_heuristic)
        assert result.success
        _validate_path(board, state, result)

    def test_solution_is_optimal(self):
        """A* con heurística admisible debe encontrar la solución óptima."""
        board, state = _simple_board()
        result = astar(board, state, manhattan_heuristic)
        assert result.cost == 6

    def test_optimal_with_euclidean(self):
        """A* con euclidiana (admisible) también debe dar óptimo."""
        board, state = _simple_board()
        result = astar(board, state, euclidean_heuristic)
        assert result.cost == 6

    def test_already_solved(self):
        board, state = _already_solved()
        result = astar(board, state, manhattan_heuristic)
        assert result.success is True
        assert result.cost == 0
        assert result.path == []

    def test_impossible_returns_failure(self):
        board, state = _impossible_board()
        result = astar(board, state, manhattan_heuristic)
        assert result.success is False
        assert result.path == []

    def test_metrics_populated(self):
        board, state = _simple_board()
        result = astar(board, state, manhattan_heuristic)
        assert result.expanded_nodes > 0
        assert result.time_elapsed >= 0

    def test_astar_expands_less_than_bfs(self):
        """A* con buena heurística debe expandir menos nodos que BFS."""
        board, state = _simple_board()
        bfs_result = bfs(board, state)
        astar_result = astar(board, state, manhattan_heuristic)
        assert astar_result.cost == bfs_result.cost  # misma optimalidad
        assert astar_result.expanded_nodes <= bfs_result.expanded_nodes

    def test_optimality_regression_case(self):
        """A* con manhattan debe empatar costo óptimo de BFS en este caso."""
        board, state = _astar_counterexample_board()
        bfs_result = bfs(board, state)
        astar_result = astar(board, state, manhattan_heuristic)

        assert bfs_result.success is True
        assert astar_result.success is True
        assert bfs_result.cost == 20
        assert astar_result.cost == bfs_result.cost
