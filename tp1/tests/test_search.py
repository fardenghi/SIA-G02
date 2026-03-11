from tp1_search.sokoban.parser import parse_board_string
from tp1_search.sokoban.actions import apply_action
from tp1_search.sokoban.goal import is_goal
from tp1_search.search.bfs import bfs


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

        # Reproducir el camino
        current = state
        for direction in result.path:
            current = apply_action(board, current, direction)
            assert current is not None

        assert is_goal(board, current)

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
