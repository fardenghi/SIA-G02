"""Red de Hopfield (memoria asociativa) con regla de Hebb.

- Patrones en {+1, -1}, vectores aplanados de tamaño N.
- W = (1/N) * sum_p (xi_p xi_p^T), con diagonal nula.
- Actualización síncrona o asíncrona; en ambos casos se devuelve la historia
  de estados intermedios para visualizar la convergencia paso a paso.
"""
from __future__ import annotations

import numpy as np


def _sign(x: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """sgn(x) con convención: si x == 0 se mantiene el valor previo."""
    s = np.sign(x).astype(np.int8)
    s[s == 0] = prev[s == 0]
    return s


class HopfieldNetwork:
    def __init__(self, n_units: int):
        self.n_units = int(n_units)
        self.weights = np.zeros((self.n_units, self.n_units), dtype=np.float64)
        self.stored: list[np.ndarray] = []

    def store(self, patterns: list[np.ndarray] | np.ndarray) -> None:
        """Aprende los patrones con la regla de Hebb."""
        P = np.asarray(patterns, dtype=np.float64)
        if P.ndim == 1:
            P = P.reshape(1, -1)
        if P.shape[1] != self.n_units:
            raise ValueError(
                f"Patrones de tamaño {P.shape[1]}, esperados {self.n_units}"
            )
        W = P.T @ P / self.n_units
        np.fill_diagonal(W, 0.0)
        self.weights = W
        self.stored = [p.astype(np.int8) for p in P.astype(np.int8)]

    def energy(self, state: np.ndarray) -> float:
        s = state.astype(np.float64)
        return -0.5 * float(s @ self.weights @ s)

    def recall(
        self,
        query: np.ndarray,
        mode: str = "sync",
        max_steps: int = 50,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[float], bool]:
        """Itera hasta converger o agotar max_steps.

        Devuelve: (estado_final, historia_de_estados, historia_de_energia, convergio)
        La historia incluye el estado inicial.
        """
        if mode not in ("sync", "async"):
            raise ValueError(f"mode inválido: {mode}")

        state = query.astype(np.int8).copy()
        history: list[np.ndarray] = [state.copy()]
        energies: list[float] = [self.energy(state)]

        if mode == "sync":
            for _ in range(max_steps):
                new_state = _sign(self.weights @ state, state)
                history.append(new_state.copy())
                energies.append(self.energy(new_state))
                if np.array_equal(new_state, state):
                    return new_state, history, energies, True
                # Detectar ciclo de longitud 2 (oscilación clásica del modo síncrono)
                if len(history) >= 3 and np.array_equal(new_state, history[-3]):
                    return new_state, history, energies, False
                state = new_state
            return state, history, energies, False

        # async: actualizar una neurona a la vez en orden aleatorio
        rng = rng if rng is not None else np.random.default_rng()
        for _ in range(max_steps):
            order = rng.permutation(self.n_units)
            changed = False
            for i in order:
                h = float(self.weights[i] @ state)
                new_val = np.int8(1 if h > 0 else (-1 if h < 0 else state[i]))
                if new_val != state[i]:
                    state[i] = new_val
                    changed = True
            history.append(state.copy())
            energies.append(self.energy(state))
            if not changed:
                return state, history, energies, True
        return state, history, energies, False

    def is_stored(self, state: np.ndarray) -> int:
        """Devuelve el índice del patrón almacenado igual a `state`, o -1."""
        for i, p in enumerate(self.stored):
            if np.array_equal(state, p) or np.array_equal(state, -p):
                return i
        return -1


def add_noise(
    pattern: np.ndarray,
    noise_level: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Invierte un ~noise_level de los bits (Bernoulli por sitio)."""
    rng = rng if rng is not None else np.random.default_rng()
    mask = rng.random(pattern.shape) < noise_level
    noisy = pattern.copy()
    noisy[mask] = -noisy[mask]
    return noisy
