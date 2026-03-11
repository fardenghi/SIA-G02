from abc import ABC, abstractmethod
from collections import deque

from tp1_search.search.node import SearchNode


class Frontier(ABC):
    """Interfaz común para todas las fronteras de búsqueda.

    Cada algoritmo usa una estructura distinta:
      - BFS  -> QueueFrontier  (FIFO: primero que entra, primero que sale)
      - DFS  -> StackFrontier  (LIFO: último que entra, primero que sale)
      - A*/Greedy -> PriorityFrontier (sale el de menor prioridad)
    """

    @abstractmethod
    def push(self, node: SearchNode) -> None: ...

    @abstractmethod
    def pop(self) -> SearchNode: ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...


class QueueFrontier(Frontier):
    """Frontera FIFO para BFS. Usa collections.deque."""

    def __init__(self) -> None:
        self._queue: deque[SearchNode] = deque()

    def push(self, node: SearchNode) -> None:
        self._queue.append(node)

    def pop(self) -> SearchNode:
        return self._queue.popleft()

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)
