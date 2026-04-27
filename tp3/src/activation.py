import numpy as np

def step(h):
    return 1 if h >= 0 else -1

def linear(h):
    return h

def sigmoid(h, beta=1):
    return 1 / (1 + np.exp(-2 * beta * h))

def sigmoid_prime(h, beta=1):
    s = sigmoid(h, beta)
    return 2 * beta * s * (1 - s)

def tanh_act(h, beta=1):
    return np.tanh(beta * h)

def tanh_prime(h, beta=1):
    return beta * (1 - np.tanh(beta * h) ** 2)
