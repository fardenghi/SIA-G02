import numpy as np


class HopfieldNetwork:
    def __init__(self) -> None:
        self._W: np.ndarray | None = None
        self._n: int | None = None
        self._triu_i: np.ndarray | None = None
        self._triu_j: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self._n

    @property
    def weights(self) -> np.ndarray:
        return self._W

    def train(self, patterns: np.ndarray) -> "HopfieldNetwork":
        p, N = patterns.shape
        self._n = N
        self._triu_i, self._triu_j = np.triu_indices(N, k=1)

        # Only compute upper triangle, then mirror (w_ij = w_ji)
        w_vals = (
            np.sum(
                patterns[:, self._triu_i] * patterns[:, self._triu_j], axis=0
            )
            / N
        )

        self._W = np.zeros((N, N))
        self._W[self._triu_i, self._triu_j] = w_vals
        self._W[self._triu_j, self._triu_i] = w_vals
        return self

    def predict(
        self, query: np.ndarray, max_iter: int = 20
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        S = query.astype(float).copy()
        history = [S.copy()]

        for _ in range(max_iter):
            h = self._W @ S
            S_new = np.where(h != 0, np.sign(h), S)
            if np.array_equal(S_new, S):
                break
            S = S_new
            history.append(S.copy())

        return S, history

    def energy(self, state: np.ndarray) -> float:
        # H = -sum_{j>i} w_ij * S_i * S_j  (upper triangle only; factor of 2 cancels the ½)
        return float(
            -np.sum(
                self._W[self._triu_i, self._triu_j]
                * state[self._triu_i]
                * state[self._triu_j]
            )
        )
