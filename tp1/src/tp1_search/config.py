import tomllib
from dataclasses import dataclass
from pathlib import Path


VALID_ALGORITHMS = {"bfs", "dfs", "greedy", "astar", "iddfs"}
VALID_HEURISTICS = {"manhattan", "euclidean", "dead_square", "hungarian"}
INFORMED_ALGORITHMS = {"greedy", "astar"}


@dataclass
class SearchConfig:
    """Configuración para una ejecución del motor de búsqueda."""

    algorithm: str
    board_path: str
    heuristics: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Algoritmo '{self.algorithm}' no válido. "
                f"Opciones: {', '.join(sorted(VALID_ALGORITHMS))}"
            )
        if not Path(self.board_path).exists():
            raise FileNotFoundError(f"Tablero no encontrado: {self.board_path}")
        if self.algorithm in INFORMED_ALGORITHMS and not self.heuristics:
            raise ValueError(f"El algoritmo '{self.algorithm}' requiere una heurística")

        invalid = [h for h in self.heuristics if h not in VALID_HEURISTICS]
        if invalid:
            raise ValueError(
                f"Heurística(s) no válida(s): {', '.join(invalid)}. "
                f"Opciones: {', '.join(sorted(VALID_HEURISTICS))}"
            )


def _parse_heuristics(value: object) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        return (value,)

    if isinstance(value, list):
        if not value:
            raise ValueError("El arreglo 'heuristic' no puede estar vacío")

        heuristics: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("'heuristic' debe ser string o arreglo de strings")
            heuristics.append(item)
        return tuple(heuristics)

    raise ValueError("'heuristic' debe ser string o arreglo de strings")


def load_config(path: str | Path) -> SearchConfig:
    """Carga una configuración desde un archivo TOML.

    Ejemplo de archivo:
        [search]
        algorithm = "bfs"
        board = "boards/sokoban/level_01.txt"

        # Para algoritmos informados:
        heuristic = "manhattan"
        # o
        # heuristic = ["manhattan", "dead_square"]
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    search = data.get("search", {})

    has_heuristic = "heuristic" in search
    has_heuristics = "heuristics" in search
    if has_heuristic and has_heuristics:
        raise ValueError("Usar solo una clave: 'heuristic' o 'heuristics'")

    raw_heuristics = (
        search.get("heuristic") if has_heuristic else search.get("heuristics")
    )

    config = SearchConfig(
        algorithm=search.get("algorithm", ""),
        board_path=search.get("board", ""),
        heuristics=_parse_heuristics(raw_heuristics),
    )
    config.validate()
    return config
