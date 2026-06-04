from dataclasses import dataclass

from tp1_search.types import Direction


@dataclass
class SearchResult:
    """Resultado de un algoritmo de búsqueda.

    Contiene:
      - Resultado (éxito/fracaso)
      - Costo de la solución
      - Cantidad de nodos expandidos
      - Cantidad de nodos en la frontera al finalizar
      - Solución (camino desde estado inicial al final)
      - Tiempo de procesamiento
    """

    success: bool
    path: list[Direction]
    cost: int
    expanded_nodes: int
    frontier_nodes: int
    max_frontier_nodes: int
    time_elapsed: float  # en segundos
