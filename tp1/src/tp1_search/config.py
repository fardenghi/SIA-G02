import tomllib
from dataclasses import dataclass
from pathlib import Path


VALID_ALGORITHMS = {"bfs", "dfs", "greedy", "astar", "iddfs"}


@dataclass
class SearchConfig:
    """Configuración para una ejecución del motor de búsqueda."""

    algorithm: str
    board_path: str
    heuristic: str | None = None  # solo para greedy / astar
    dead_square_pruning: bool = True  # False para deshabilitar la poda de dead squares

    def validate(self) -> None:
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Algoritmo '{self.algorithm}' no válido. "
                f"Opciones: {', '.join(sorted(VALID_ALGORITHMS))}"
            )
        if not Path(self.board_path).exists():
            raise FileNotFoundError(f"Tablero no encontrado: {self.board_path}")
        if self.algorithm in {"greedy", "astar"} and self.heuristic is None:
            raise ValueError(f"El algoritmo '{self.algorithm}' requiere una heurística")


def load_config(path: str | Path) -> SearchConfig:
    """Carga una configuración desde un archivo TOML.

    Ejemplo de archivo:
        [search]
        algorithm = "bfs"
        board = "boards/sokoban/level_01.txt"
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    search = data.get("search", {})

    config = SearchConfig(
        algorithm=search.get("algorithm", ""),
        board_path=search.get("board", ""),
        heuristic=search.get("heuristic"),
        dead_square_pruning=search.get("dead_square_pruning", True),
    )
    config.validate()
    return config
