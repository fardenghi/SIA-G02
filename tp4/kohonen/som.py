import numpy as np


class SOM:
    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        input_dim: int,
        lr: float,
        lr_decay: str,
        radius: float,
        radius_decay: str,
        neighborhood_fn: str,
        epochs: int,
        seed: int = 42,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.input_dim = input_dim
        self.lr = lr
        self.lr_decay = lr_decay
        self.radius = radius
        self.radius_decay = radius_decay
        self.neighborhood_fn = neighborhood_fn
        self.epochs = epochs

        self._rng = np.random.default_rng(seed)
        self.weights = self._rng.standard_normal((grid_rows, grid_cols, input_dim))

        # precompute grid coordinate arrays for vectorised distance calculations
        rows_idx, cols_idx = np.meshgrid(np.arange(grid_rows), np.arange(grid_cols), indexing="ij")
        self._grid_r = rows_idx  # (R, C)
        self._grid_c = cols_idx  # (R, C)

    # ------------------------------------------------------------------
    # core internals
    # ------------------------------------------------------------------

    def _find_bmu(self, x: np.ndarray) -> tuple[int, int]:
        diff = self.weights - x          # (R, C, D)
        dists = np.linalg.norm(diff, axis=2)  # (R, C)
        idx = np.argmin(dists)
        return divmod(idx, self.grid_cols)

    def _neighborhood(self, bmu: tuple[int, int], sigma: float, mode: str) -> np.ndarray:
        if mode == "gaussian":
            return self._gaussian(bmu, sigma)
        return self._bubble(bmu, sigma)

    def _gaussian(self, bmu: tuple[int, int], sigma: float) -> np.ndarray:
        dr = self._grid_r - bmu[0]
        dc = self._grid_c - bmu[1]
        dist2 = dr ** 2 + dc ** 2
        return np.exp(-dist2 / (2 * sigma ** 2))

    def _bubble(self, bmu: tuple[int, int], radius: float) -> np.ndarray:
        dr = self._grid_r - bmu[0]
        dc = self._grid_c - bmu[1]
        dist = np.sqrt(dr ** 2 + dc ** 2)
        return (dist <= radius).astype(float)

    def _decay(self, value: float, t: int, T: int, mode: str) -> float:
        if mode == "exponential":
            # τ = T/log(σ₀) cuando value > 1 (radio), τ = T para learning rate
            tau = T / np.log(value) if value > 1 else T
            return value * np.exp(-t / tau)
        # linear
        return value * (1 - t / T)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_quantization_error(self, X: np.ndarray) -> float:
        coords = self.predict(X)
        dists = []
        for x, (r, c) in zip(X, coords):
            d = np.linalg.norm(x - self.weights[r, c])
            dists.append(d)
        return float(np.mean(dists))

    def train(self, X: np.ndarray) -> list[float]:
        # inicialización con muestras aleatorias del dataset
        n_neurons = self.grid_rows * self.grid_cols
        sample_idx = self._rng.choice(len(X), size=n_neurons, replace=True)
        self.weights = X[sample_idx].reshape(self.grid_rows, self.grid_cols, self.input_dim).copy()

        history = []
        T = self.epochs
        for t in range(T):
            lr_t = self._decay(self.lr, t, T, self.lr_decay)
            sigma_t = self._decay(self.radius, t, T, self.radius_decay)
            # clamp sigma to avoid division by zero in gaussian
            sigma_t = max(sigma_t, 1e-6)

            for i in self._rng.permutation(len(X)):
                x = X[i]
                bmu = self._find_bmu(x)
                h = self._neighborhood(bmu, sigma_t, self.neighborhood_fn)  # (R, C)
                delta = lr_t * h[:, :, np.newaxis] * (x - self.weights)
                self.weights += delta
            
            # Record quantization error at the end of each epoch
            history.append(self.get_quantization_error(X))
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        coords = np.array([self._find_bmu(x) for x in X])
        return coords  # (N, 2)

    def u_matrix(self) -> np.ndarray:
        u = np.zeros((self.grid_rows, self.grid_cols))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                nbr_dists = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                        d = np.linalg.norm(self.weights[r, c] - self.weights[nr, nc])
                        nbr_dists.append(d)
                u[r, c] = np.mean(nbr_dists)
        return u
