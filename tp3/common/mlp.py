import json
import time

import numpy as np

from common.activations import softmax, softmax_prime, tanh_act, tanh_prime
from common.layers import DenseLayer
from common.losses import cross_entropy, cross_entropy_softmax_grad, mse, mse_grad

_ACTIVATIONS = {
    "tanh": (tanh_act, tanh_prime),
    "softmax": (softmax, softmax_prime),
}

_VALID_COMBOS = {("mse", "tanh"), ("cross_entropy", "softmax")}


class MLP:
    def __init__(self, layer_sizes, activation="tanh", output_activation="tanh",
                 loss="mse", weight_init="xavier", seed=42):
        if (loss, output_activation) not in _VALID_COMBOS:
            raise ValueError(
                f"Unsupported (loss, output_activation): ({loss!r}, {output_activation!r}). "
                f"Supported combinations: {_VALID_COMBOS}"
            )

        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.loss = loss
        self.weight_init = weight_init
        self.seed = seed
        self.history = []
        self._X = None

        act_fn, act_prime = _ACTIVATIONS[activation]
        out_fn, out_prime = _ACTIVATIONS[output_activation]

        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            is_output = (i == len(layer_sizes) - 2)
            fn = out_fn if is_output else act_fn
            prime = out_prime if is_output else act_prime
            self.layers.append(DenseLayer(n_in, n_out, fn, prime, weight_init, rng))

    # ------------------------------------------------------------------
    # Core forward / backward
    # ------------------------------------------------------------------

    def forward(self, X):
        self._X = X
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y_true):
        """Returns list of (dW, db) tuples indexed by layer."""
        out = self.layers[-1]
        if self.loss == "cross_entropy":
            delta = cross_entropy_softmax_grad(y_true, out.a)
        else:
            delta = mse_grad(y_true, out.a) * tanh_prime(out.z)

        grads = [None] * len(self.layers)
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            a_prev = self._X if l == 0 else self.layers[l - 1].a
            dW, db, delta_a = layer.backward(delta, a_prev)
            grads[l] = (dW, db)
            if l > 0:
                prev = self.layers[l - 1]
                delta = delta_a * prev.activation_prime(prev.z)

        return grads

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(self, X):
        pred = self.forward(X)
        if self.output_activation == "softmax":
            return np.argmax(pred, axis=1)
        return pred

    def evaluate(self, X, y):
        pred = self.forward(X)
        if self.loss == "cross_entropy":
            loss_val = cross_entropy(y, pred)
        else:
            loss_val = mse(y, pred)

        if pred.shape[1] > 1:
            pred_cls = np.argmax(pred, axis=1)
            true_cls = np.argmax(y, axis=1) if y.ndim > 1 else y.astype(int)
        else:
            pred_cls = np.sign(pred.ravel())
            true_cls = np.asarray(y).ravel()

        accuracy = float(np.mean(pred_cls == true_cls))
        return {"loss": loss_val, "accuracy": accuracy}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32, optimizer=None,
            patience=None, verbose=True, tracker=None,
            data_augmentation=False):
        rng = np.random.default_rng(self.seed)
        N = X_train.shape[0]
        lr = getattr(optimizer, "lr", 0.0)

        best_val_loss = np.inf
        best_weights = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            idx = rng.permutation(N)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            for start in range(0, N, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]
                
                if data_augmentation:
                    X_batch = self._augment_batch(X_batch, rng)

                self.forward(X_batch)
                grads = self.backward(y_batch)
                for l, (dW, db) in enumerate(grads):
                    optimizer.step(self.layers[l], dW, db)

            train_m = self.evaluate(X_train, y_train)
            if X_val is not None:
                val_m = self.evaluate(X_val, y_val)
            else:
                val_m = {"loss": None, "accuracy": None}

            elapsed = time.time() - t0
            entry = {
                "epoch": epoch,
                "loss_train": train_m["loss"],
                "acc_train": train_m["accuracy"],
                "loss_val": val_m["loss"],
                "acc_val": val_m["accuracy"],
                "time_s": elapsed,
                "lr": lr,
            }
            self.history.append(entry)

            if tracker is not None:
                tracker.record(entry)

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0):
                msg = (f"Epoch {epoch:>5}/{epochs}  "
                       f"loss_train={train_m['loss']:.4f}  "
                       f"acc_train={train_m['accuracy']:.4f}")
                if X_val is not None:
                    msg += (f"  loss_val={val_m['loss']:.4f}"
                            f"  acc_val={val_m['accuracy']:.4f}")
                msg += f"  ({elapsed:.2f}s)"
                print(msg)

            if patience is not None and X_val is not None:
                if val_m["loss"] < best_val_loss:
                    best_val_loss = val_m["loss"]
                    best_weights = [(l.W.copy(), l.b.copy()) for l in self.layers]
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} "
                                  f"(best val_loss={best_val_loss:.4f})")
                        for layer, (W, b) in zip(self.layers, best_weights):
                            layer.W = W
                            layer.b = b
                        break

        return self

    def _augment_batch(self, X_batch, rng):
        X_aug = np.empty_like(X_batch)
        for i in range(len(X_batch)):
            img = X_batch[i].reshape(28, 28)
            
            # Random shifts (-1, 0, 1) pixels
            shift_x = rng.integers(-2, 3) # -2 to 2
            shift_y = rng.integers(-2, 3)
            
            # Roll and pad horizontally
            if shift_x != 0:
                img = np.roll(img, shift_x, axis=1)
                if shift_x > 0: img[:, :shift_x] = 0
                else: img[:, shift_x:] = 0
                
            # Roll and pad vertically
            if shift_y != 0:
                img = np.roll(img, shift_y, axis=0)
                if shift_y > 0: img[:shift_y, :] = 0
                else: img[shift_y:, :] = 0
                
            X_aug[i] = img.flatten()
            
        # Add small Gaussian noise to the whole batch at once (much faster)
        noise = rng.normal(0, 0.05, X_aug.shape)
        X_aug = np.clip(X_aug + noise, 0, 1)
        
        return X_aug

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"W_{i}"] = layer.W
            arrays[f"b_{i}"] = layer.b
        meta = json.dumps({
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "loss": self.loss,
            "weight_init": self.weight_init,
            "seed": self.seed,
        })
        np.savez(path, meta=np.array(meta), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        mlp = cls(
            layer_sizes=meta["layer_sizes"],
            activation=meta["activation"],
            output_activation=meta["output_activation"],
            loss=meta["loss"],
            weight_init=meta["weight_init"],
            seed=meta["seed"],
        )
        for i, layer in enumerate(mlp.layers):
            layer.W = data[f"W_{i}"].copy()
            layer.b = data[f"b_{i}"].copy()
        return mlp
