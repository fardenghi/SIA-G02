import numpy as np
from src.activation import step, linear

class SimplePerceptron:
    def __init__(self, input_size, learning_rate=0.1, max_epochs=10, activation=step):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.activation = activation
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
                y_pred = self.predict(x_i)
                error = y_i - y_pred
                self.w += self.lr * error * x_i
                self.b += self.lr * error
                errors.append(error ** 2)
            self.loss_history.append(np.mean(errors))
