from __future__ import annotations

from dataclasses import dataclass

from tp1_search.types import Direction
from tp1_search.sokoban.state import SokobanState


@dataclass
class SearchNode:
    """Nodo del árbol de búsqueda.

    Envuelve un estado y agrega la metadata necesaria para la búsqueda:
    parent, action, depth y path_cost. Esto permite reconstruir el
    camino solución siguiendo los punteros parent.
    """

    state: SokobanState
    parent: SearchNode | None
    action: Direction | None
    depth: int
    path_cost: int

    @staticmethod
    def root(state: SokobanState) -> SearchNode:
        """Crea el nodo raíz (estado inicial, sin padre ni acción)."""
        return SearchNode(
            state=state,
            parent=None,
            action=None,
            depth=0,
            path_cost=0,
        )

    def expand_child(
        self, state: SokobanState, action: Direction, step_cost: int
    ) -> SearchNode:
        """Crea un nodo hijo a partir de este nodo."""
        return SearchNode(
            state=state,
            parent=self,
            action=action,
            depth=self.depth + 1,
            path_cost=self.path_cost + step_cost,
        )

    def reconstruct_path(self) -> list[Direction]:
        """Reconstruye la secuencia de acciones desde la raíz hasta este nodo."""
        actions: list[Direction] = []
        node: SearchNode | None = self
        while node is not None and node.action is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions
