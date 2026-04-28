import numpy as np


class DenseLayer:
    def __init__(self, n_in, n_out, activation, activation_prime,
                 weight_init="xavier", rng=None):
        if rng is None:
            rng = np.random.default_rng()

        self.activation = activation
        self.activation_prime = activation_prime

        if weight_init == "xavier":
            limit = np.sqrt(1.0 / n_in)
            self.W = rng.uniform(-limit, limit, (n_in, n_out))
        elif weight_init == "normal":
            self.W = rng.normal(0.0, np.sqrt(1.0 / n_in), (n_in, n_out))
        elif weight_init == "uniform_small":
            self.W = rng.uniform(-0.5, 0.5, (n_in, n_out))
        else:
            raise ValueError(f"Unknown weight_init: {weight_init!r}")

        self.b = np.zeros(n_out)

        # Cached values from last forward pass (needed for backward)
        self.z = None
        self.a = None

    def forward(self, X):
        self.z = X @ self.W + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, delta_in, a_prev):
        """
        delta_in : ∂L/∂z_l  shape (batch, n_out)
        a_prev   : a_{l-1}  shape (batch, n_in)
        Returns  : dW (n_in, n_out), db (n_out,), delta_out = ∂L/∂a_{l-1} (batch, n_in)
        """
        dW = a_prev.T @ delta_in
        db = delta_in.sum(axis=0)
        delta_out = delta_in @ self.W.T
        return dW, db, delta_out
