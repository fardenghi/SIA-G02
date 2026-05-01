import numpy as np

from common.losses import cross_entropy
from common.mlp import MLP


class Ensemble:
    """Promedio de probabilidades sobre varios MLPs entrenados.

    Cada modelo debe tener output_activation='softmax' para que .forward()
    devuelva probabilidades comparables.
    """

    def __init__(self, models):
        self.models = list(models)
        if len(self.models) < 2:
            raise ValueError("Ensemble necesita al menos 2 modelos.")

    @classmethod
    def from_paths(cls, paths):
        return cls([MLP.load(p) for p in paths])

    def forward(self, X):
        probs = np.stack([m.forward(X) for m in self.models], axis=0)
        return probs.mean(axis=0)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X, y):
        probs = self.forward(X)
        loss_val = cross_entropy(y, probs)
        pred_cls = np.argmax(probs, axis=1)
        true_cls = np.argmax(y, axis=1) if y.ndim > 1 else y.astype(int)
        accuracy = float(np.mean(pred_cls == true_cls))
        return {"loss": loss_val, "accuracy": accuracy}
