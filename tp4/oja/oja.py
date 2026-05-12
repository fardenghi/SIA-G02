import numpy as np


class OjaNetwork:
    def __init__(
        self,
        input_dim: int,
        lr: float,
        epochs: int,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs

        self._rng = np.random.default_rng(seed)
        w = self._rng.standard_normal(input_dim)
        self.weights = w / np.linalg.norm(w)
        self.history: list[np.ndarray] = [self.weights.copy()]

    def train(self, X: np.ndarray) -> None:
        n = X.shape[0]
        step = 1
        for _ in range(self.epochs):
            order = self._rng.permutation(n)
            for i in order:
                x = X[i]
                y = float(self.weights @ x)
                eta = self.lr / step
                self.weights = self.weights + eta * y * (x - y * self.weights)
                step += 1
            self.history.append(self.weights.copy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights

    def component(self) -> np.ndarray:
        return self.weights / np.linalg.norm(self.weights)
