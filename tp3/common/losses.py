import numpy as np


def mse(y_true, y_pred):
    # 0.5 factor so that grad = (y_pred - y_true) / N without a factor of 2
    return 0.5 * float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_true, y_pred):
    return (y_pred - y_true) / len(y_true)


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=-1)))


def cross_entropy_softmax_grad(y_true, y_pred):
    return (y_pred - y_true) / len(y_true)
