import json

from PIL import Image

from tp1_search.output.pygame_anim import export_gif


def test_export_gif_creates_animated_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    replay_path = tmp_path / "replay.json"
    gif_path = tmp_path / "anim.gif"

    replay = {
        "metadata": {
            "algorithm": "astar",
            "board_path": "boards/sokoban/level_01.txt",
            "success": True,
            "cost": 1,
            "expanded_nodes": 3,
            "frontier_nodes": 1,
            "time_elapsed": 0.01,
        },
        "board": {
            "rows": 5,
            "cols": 5,
            "walls": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
            "goals": [[2, 3]],
        },
        "moves": ["RIGHT"],
        "frames": [
            {"player": [2, 1], "boxes": [[2, 2]]},
            {"player": [2, 2], "boxes": [[2, 3]]},
        ],
    }
    replay_path.write_text(json.dumps(replay), encoding="utf-8")

    output = export_gif(replay_path, gif_path, cell_size=32, fps=4.0)

    assert output == gif_path
    assert gif_path.exists()

    with Image.open(gif_path) as img:
        assert img.format == "GIF"
        assert img.n_frames == 2
