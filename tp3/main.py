import numpy as np
from src.perceptron import SimplePerceptron

def main():
    X = np.array([
        [-1,  1],
        [ 1, -1],
        [-1, -1],
        [ 1,  1],
    ])
    y = np.array([-1, -1, -1, 1])

    perceptron = SimplePerceptron(input_size=2, learning_rate=0.1, max_epochs=100)
    perceptron.train(X, y)

    print("Verificación AND:")
    for x_i, y_i in zip(X, y):
        pred = perceptron.predict(x_i)
        ok = "✓" if pred == y_i else "✗"
        print(f"  x={x_i}  esperado={y_i:+d}  predicho={pred:+d}  {ok}")


if __name__ == "__main__":
    main()
