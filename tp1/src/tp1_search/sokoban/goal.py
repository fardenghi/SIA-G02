from tp1_search.sokoban.board import Board
from tp1_search.sokoban.state import SokobanState


def is_goal(board: Board, state: SokobanState) -> bool:
    """True si todas las cajas están sobre un objetivo."""
    return state.boxes == board.goals
