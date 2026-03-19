import pytest

from tp1_search.config import load_config


def _write_config(tmp_path, content: str):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_accepts_single_heuristic_string(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
algorithm = "astar"
board = "boards/sokoban/level_01.txt"
heuristic = "manhattan"
""",
    )

    config = load_config(config_path)
    assert config.heuristics == ("manhattan",)


def test_load_config_accepts_heuristic_array(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
algorithm = "astar"
board = "boards/sokoban/level_01.txt"
heuristic = ["manhattan", "dead_square"]
""",
    )

    config = load_config(config_path)
    assert config.heuristics == ("manhattan", "dead_square")


def test_load_config_accepts_heuristics_alias(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
algorithm = "greedy"
board = "boards/sokoban/level_01.txt"
heuristics = ["euclidean", "dead_square"]
""",
    )

    config = load_config(config_path)
    assert config.heuristics == ("euclidean", "dead_square")


def test_load_config_accepts_weighted_hungarian(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
algorithm = "greedy"
board = "boards/sokoban/level_01.txt"
heuristic = "weighted_hungarian"
""",
    )

    config = load_config(config_path)
    assert config.heuristics == ("weighted_hungarian",)


def test_load_config_rejects_both_heuristic_keys(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
[search]
algorithm = "astar"
board = "boards/sokoban/level_01.txt"
heuristic = "manhattan"
heuristics = ["dead_square"]
""",
    )

    with pytest.raises(ValueError, match="solo una clave"):
        load_config(config_path)
