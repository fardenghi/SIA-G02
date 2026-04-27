import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.perceptron import SimplePerceptron
from src.activation import step, linear, sigmoid, sigmoid_prime, tanh_act, tanh_prime

ACTIVATIONS = {
    "escalon":  (step,     None),
    "lineal":   (linear,   None),
    "sigmoide": (sigmoid,  sigmoid_prime),
    "tanh":     (tanh_act, tanh_prime),
}


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def get_dataset(cfg):
    problem = cfg["problem"]
    if problem == "AND":
        X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
        y = np.array([-1, -1, -1, 1])
        return X, y
    if problem == "y=x":
        rng = np.random.default_rng(cfg.get("seed", 42))
        x = rng.uniform(-10, 10, cfg.get("n_samples", 50))
        noise = rng.normal(0, cfg.get("noise", 0.5), len(x))
        return x.reshape(-1, 1), x + noise
    if problem == "y=tanh(x)":
        rng = np.random.default_rng(cfg.get("seed", 42))
        x = rng.uniform(-5, 5, cfg.get("n_samples", 50))
        noise = rng.normal(0, cfg.get("noise", 0.1), len(x))
        return x.reshape(-1, 1), np.tanh(x) + noise
    raise ValueError(f"Dataset desconocido: {problem!r}")


def run(cfg):
    p_cfg = cfg["perceptron"]
    activation_name = p_cfg["type"]
    activation, activation_prime = ACTIVATIONS[activation_name]

    X, y = get_dataset(cfg["dataset"])

    p = SimplePerceptron(
        input_size=X.shape[1],
        learning_rate=p_cfg["learning_rate"],
        max_epochs=p_cfg["max_epochs"],
        activation=activation,
        activation_prime=activation_prime,
    )
    p.train(X, y)

    print(f"Perceptrón: {activation_name}  |  lr={p_cfg['learning_rate']}  |  épocas={p_cfg['max_epochs']}")
    print(f"Problema: {cfg['dataset']['problem']}")
    print(f"MSE final: {p.loss_history[-1]:.4f}")

    if activation_name == "escalon":
        print()
        for x_i, y_i in zip(X, y):
            pred = p.predict(x_i)
            ok = "✓" if pred == y_i else "✗"
            print(f"  x={x_i}  esperado={y_i:+d}  predicho={pred:+d}  {ok}")
    else:
        w_str = "  ".join(f"w{i}={v:.4f}" for i, v in enumerate(p.w))
        print(f"{w_str}  b={p.b:.4f}")

    if cfg.get("plot", False):
        _plot(p, X, y, cfg)


def _plot(p, X, y, cfg):
    problem = cfg["dataset"]["problem"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Perceptrón {cfg['perceptron']['type']} — {problem}")

    ax = axes[0]
    if problem == "AND":
        colors = ["red" if yi == -1 else "blue" for yi in y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, zorder=3)
        if p.w[1] != 0:
            xs = np.linspace(-1.5, 1.5, 100)
            ax.plot(xs, -(p.w[0] * xs + p.b) / p.w[1], "k-")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_title("Frontera de decisión")
    elif problem == "y=x":
        x_vals = X[:, 0]
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.scatter(x_vals, y, alpha=0.6, label="datos")
        ax.plot(x_line, x_line, "k--", label="y=x")
        ax.plot(x_line, p.w[0] * x_line + p.b, "r-", label="perceptron")
        ax.legend(); ax.set_title("Ajuste lineal")
    elif problem == "y=tanh(x)":
        x_vals = X[:, 0]
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.scatter(x_vals, y, alpha=0.6, label="datos")
        ax.plot(x_line, np.tanh(x_line), "k--", label="y=tanh(x)")
        ax.plot(x_line, [p.predict(np.array([xi])) for xi in x_line], "r-", label="perceptron")
        ax.legend(); ax.set_title("Ajuste no lineal")

    axes[1].plot(p.loss_history)
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("MSE")
    axes[1].set_title("Curva de pérdida")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    run(load_config(config_path))
