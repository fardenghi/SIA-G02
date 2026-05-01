import numpy as np

def step(h):
    return np.where(np.asarray(h) >= 0, 1, -1)

def linear(h):
    return h

def sigmoid(h, beta=1):
    return 1 / (1 + np.exp(-2 * beta * np.asarray(h)))

def sigmoid_prime(h, beta=1):
    s = sigmoid(h, beta)
    return 2 * beta * s * (1 - s)

def tanh_act(h, beta=1):
    return np.tanh(beta * np.asarray(h))

def tanh_prime(h, beta=1):
    t = np.tanh(beta * np.asarray(h))
    return beta * (1 - t ** 2)

def relu(h):
    return np.maximum(0.0, np.asarray(h))

def relu_prime(h):
    return (np.asarray(h) > 0).astype(float)

def softmax(h):
    h = np.asarray(h)
    e = np.exp(h - h.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def softmax_prime(h):
    s = softmax(h)
    return s * (1 - s)
