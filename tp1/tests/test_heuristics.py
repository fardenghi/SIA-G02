import math

from tp1_search.types import Position
from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.sokoban.parser import parse_board_string
from tp1_search.sokoban.heuristics import (
    manhattan_heuristic,
    euclidean_heuristic,
    dead_square_heuristic,
)


def _simple_board():
    """
    #####
    #   #
    # $ #
    # . #
    # @ #
    #####
    """
    return parse_board_string("#####\n#   #\n# $ #\n# . #\n# @ #\n#####")


def _dead_corner_board():
    """Caja en esquina (1,1) que es dead square, goal en (3,3).

    #####
    #$  #
    #   #
    #  .#
    # @ #
    #####
    """
    return parse_board_string("#####\n#$  #\n#   #\n#  .#\n# @ #\n#####")


# =========================================================================
# Manhattan
# =========================================================================


class TestManhattan:
    def test_zero_when_solved(self):
        board, _ = _simple_board()
        # Caja en el goal (3,2), jugador en (4,2)
        state = SokobanState(
            player=Position(4, 2),
            boxes=frozenset({Position(3, 2)}),
        )
        assert manhattan_heuristic(board, state) == 0.0

    def test_positive_when_not_solved(self):
        board, state = _simple_board()
        # Caja en (2,2), goal en (3,2) -> Manhattan = 1
        h = manhattan_heuristic(board, state)
        assert h == 1.0

    def test_multiple_boxes(self):
        board = Board(
            rows=6,
            cols=6,
            walls=frozenset(
                Position(r, c)
                for r in range(6)
                for c in range(6)
                if r in (0, 5) or c in (0, 5)
            ),
            goals=frozenset({Position(1, 1), Position(4, 4)}),
        )
        # Caja A en (1,1) -> dist a goal (1,1) = 0
        # Caja B en (4,4) -> dist a goal (4,4) = 0
        state = SokobanState(
            player=Position(3, 3),
            boxes=frozenset({Position(1, 1), Position(4, 4)}),
        )
        assert manhattan_heuristic(board, state) == 0.0

    def test_admissible(self):
        """Manhattan nunca debe superar el costo real."""
        board, state = _simple_board()
        h = manhattan_heuristic(board, state)
        # La solución óptima es 6 pasos; h=1 <= 6
        assert h <= 6


# =========================================================================
# Euclidean
# =========================================================================


class TestEuclidean:
    def test_zero_when_solved(self):
        board, _ = _simple_board()
        state = SokobanState(
            player=Position(4, 2),
            boxes=frozenset({Position(3, 2)}),
        )
        assert euclidean_heuristic(board, state) == 0.0

    def test_positive_when_not_solved(self):
        board, state = _simple_board()
        h = euclidean_heuristic(board, state)
        assert h > 0.0

    def test_less_or_equal_to_manhattan(self):
        """Euclidiana siempre <= Manhattan para misma configuración."""
        board, state = _simple_board()
        h_euc = euclidean_heuristic(board, state)
        h_man = manhattan_heuristic(board, state)
        assert h_euc <= h_man

    def test_admissible(self):
        board, state = _simple_board()
        h = euclidean_heuristic(board, state)
        assert h <= 6


# =========================================================================
# Dead square
# =========================================================================


class TestDeadSquare:
    def test_zero_when_no_dead(self):
        board, state = _simple_board()
        # Caja en (2,2) — no es dead square en este tablero
        assert dead_square_heuristic(board, state) == 0.0

    def test_inf_when_dead(self):
        board, state = _dead_corner_board()
        # Caja en (1,1) — esquina, es dead square
        h = dead_square_heuristic(board, state)
        assert h == math.inf

    def test_zero_when_solved(self):
        board, _ = _simple_board()
        state = SokobanState(
            player=Position(4, 2),
            boxes=frozenset({Position(3, 2)}),
        )
        assert dead_square_heuristic(board, state) == 0.0
