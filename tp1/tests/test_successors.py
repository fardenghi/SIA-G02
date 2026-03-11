from tp1_search.types import Position, Direction
from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState
from tp1_search.sokoban.actions import apply_action
from tp1_search.sokoban.successors import get_successors
from tp1_search.sokoban.goal import is_goal
from tp1_search.sokoban.parser import parse_board_string


def _make_simple():
    """Tablero 5x5 abierto con 1 caja y 1 goal.

    #####
    #   #
    # $ #
    # . #
    # @ #
    #####
    """
    return parse_board_string("#####\n#   #\n# $ #\n# . #\n# @ #\n#####")


# --- apply_action ---


class TestApplyAction:
    def test_move_to_empty(self):
        board, state = _make_simple()
        # Jugador en (4,2), mover LEFT -> (4,1) libre
        new = apply_action(board, state, Direction.LEFT)
        assert new is not None
        assert new.player == Position(4, 1)
        assert new.boxes == state.boxes  # cajas no cambian

    def test_move_to_wall(self):
        board, state = _make_simple()
        # Jugador en (4,2), mover DOWN -> (5,2) es pared
        new = apply_action(board, state, Direction.DOWN)
        assert new is None

    def test_push_box_to_empty(self):
        board, state = _make_simple()
        # Mover jugador a (3,2), luego UP empuja caja de (2,2) a (1,2)
        state = apply_action(board, state, Direction.UP)  # -> (3,2)
        assert state is not None
        new = apply_action(board, state, Direction.UP)  # empuja caja
        assert new is not None
        assert new.player == Position(2, 2)
        assert Position(1, 2) in new.boxes  # caja movida
        assert Position(2, 2) not in new.boxes

    def test_push_box_to_wall(self):
        board, state = _make_simple()
        # Mover jugador a (3,2) -> (2,2) empuja caja a (1,2)
        # Luego desde (2,2) UP empuja caja de (1,2) a (0,2) que es pared
        s1 = apply_action(board, state, Direction.UP)  # -> (3,2)
        assert s1 is not None
        s2 = apply_action(board, s1, Direction.UP)  # empuja caja a (1,2)
        assert s2 is not None
        s3 = apply_action(board, s2, Direction.UP)  # intenta empujar a (0,2) pared
        assert s3 is None

    def test_push_box_to_box(self):
        """No se puede empujar una caja contra otra caja."""
        board = Board(
            rows=5,
            cols=5,
            walls=frozenset(
                Position(r, c)
                for r in range(5)
                for c in range(5)
                if r in (0, 4) or c in (0, 4)
            ),
            goals=frozenset({Position(1, 2), Position(2, 2)}),
        )
        state = SokobanState(
            player=Position(3, 2),
            boxes=frozenset({Position(2, 2), Position(1, 2)}),
        )
        # UP empujaría (2,2) a (1,2) donde ya hay otra caja
        new = apply_action(board, state, Direction.UP)
        assert new is None

    def test_original_state_unchanged(self):
        """apply_action no muta el estado original."""
        board, state = _make_simple()
        original_player = state.player
        original_boxes = state.boxes
        apply_action(board, state, Direction.UP)
        assert state.player == original_player
        assert state.boxes == original_boxes


# --- get_successors ---


class TestGetSuccessors:
    def test_successor_count(self):
        board, state = _make_simple()
        # Jugador en (4,2): UP, LEFT, RIGHT son válidos. DOWN es pared.
        succs = get_successors(board, state)
        assert len(succs) == 3

    def test_successor_cost_is_one(self):
        board, state = _make_simple()
        for _, _, cost in get_successors(board, state):
            assert cost == 1

    def test_all_successors_are_different(self):
        board, state = _make_simple()
        states = [s for s, _, _ in get_successors(board, state)]
        assert len(states) == len(set(states))


# --- is_goal ---


class TestIsGoal:
    def test_not_goal_initial(self):
        board, state = _make_simple()
        assert not is_goal(board, state)

    def test_goal_when_box_on_target(self):
        board, _ = _make_simple()
        # Poner caja directamente en el goal (3,2)
        state = SokobanState(
            player=Position(4, 2),
            boxes=frozenset({Position(3, 2)}),
        )
        assert is_goal(board, state)
