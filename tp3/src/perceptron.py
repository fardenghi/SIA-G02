import numpy as np
from src.activation import step, linear

class SimplePerceptron:
    def __init__(self, input_size, learning_rate=0.1, max_epochs=10,
                 activation=step, activation_prime=None):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.activation = activation
        self.activation_prime = activation_prime
        self.w = np.random.uniform(-0.5, 0.5, input_size)
        self.b = np.random.uniform(-0.5, 0.5)
        self.loss_history = []

    def predict(self, x):
        h = np.dot(self.w, x) + self.b
        return self.activation(h)

    def train(self, X, y):
        for epoch in range(self.max_epochs):
            errors = []
            for x_i, y_i in zip(X, y):
                h = np.dot(self.w, x_i) + self.b
                y_pred = self.activation(h)
                error = y_i - y_pred
                d = self.activation_prime(h) if self.activation_prime else 1
                self.w += self.lr * error * d * x_i
                self.b += self.lr * error * d
                errors.append(error ** 2)
            self.loss_history.append(np.mean(errors))
