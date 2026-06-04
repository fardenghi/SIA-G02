import pytest

from tp1_search.types import Position, Direction
from tp1_search.sokoban.state import SokobanState


# --- Position ---


class TestPosition:
    def test_access_by_name(self):
        p = Position(2, 3)
        assert p.row == 2
        assert p.col == 3

    def test_add_positions(self):
        p = Position(2, 3) + Position(1, -1)
        assert p == Position(3, 2)

    def test_add_direction_delta(self):
        p = Position(4, 4) + Direction.UP.delta
        assert p == Position(3, 4)

    def test_hasheable(self):
        p1 = Position(1, 2)
        p2 = Position(1, 2)
        assert hash(p1) == hash(p2)
        assert p1 in {p2}

    def test_immutable(self):
        p = Position(1, 2)
        with pytest.raises(AttributeError):
            p.row = 5  # type: ignore[misc]


# --- SokobanState ---


class TestSokobanState:
    def make_state(self, player=(2, 2), boxes=((3, 3), (4, 4))):
        return SokobanState(
            player=Position(*player),
            boxes=frozenset(Position(*b) for b in boxes),
        )

    def test_equality(self):
        s1 = self.make_state()
        s2 = self.make_state()
        assert s1 == s2

    def test_inequality_different_player(self):
        s1 = self.make_state(player=(2, 2))
        s2 = self.make_state(player=(0, 0))
        assert s1 != s2

    def test_inequality_different_boxes(self):
        s1 = self.make_state(boxes=((3, 3),))
        s2 = self.make_state(boxes=((4, 4),))
        assert s1 != s2

    def test_hasheable(self):
        s1 = self.make_state()
        s2 = self.make_state()
        assert hash(s1) == hash(s2)
        assert s1 in {s2}

    def test_immutable(self):
        s = self.make_state()
        with pytest.raises(AttributeError):
            s.player = Position(0, 0)  # type: ignore[misc]

    def test_is_box(self):
        s = self.make_state(boxes=((3, 3),))
        assert s.is_box(Position(3, 3))
        assert not s.is_box(Position(0, 0))

    def test_in_visited_set(self):
        s1 = self.make_state()
        s2 = self.make_state()
        s3 = self.make_state(player=(0, 0))
        visited = {s1}
        assert s2 in visited  # mismo estado
        assert s3 not in visited  # distinto estado
