from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import minibatches, one_hot


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)


def cross_entropy(y_true_oh: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    return float(-(y_true_oh * np.log(y_prob)).sum(axis=1).mean())


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def drelu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t**2


@dataclass(frozen=True)
class Activation:
    name: str
    f: callable
    df: callable


ACTIVATIONS: dict[str, Activation] = {
    "relu": Activation("relu", relu, drelu),
    "sigmoid": Activation("sigmoid", sigmoid, dsigmoid),
    "tanh": Activation("tanh", tanh, dtanh),
}


class MLP:
    """
    Fully-connected MLP with two hidden layers and softmax output.
    Implements backprop from scratch (NumPy only).
    """

    def __init__(
        self,
        input_dim: int = 784,
        h1: int = 128,
        h2: int = 64,
        num_classes: int = 10,
        activation: str = "relu",
        lr: float = 0.01,
        epochs: int = 12,
        batch_size: int = 128,
        momentum: float = 0.9,
        l2: float = 0.0,
        dropout: float = 0.0,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.h1 = int(h1)
        self.h2 = int(h2)
        self.num_classes = int(num_classes)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.momentum = float(momentum)
        self.l2 = float(l2)
        self.dropout = float(dropout)
        self.seed = int(seed)

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS)}")
        self.act = ACTIVATIONS[activation]

        self.params: dict[str, np.ndarray] = {}
        self.vel: dict[str, np.ndarray] = {}
        self.history_: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        self._rng = np.random.default_rng(self.seed)
        self._init_params()

    def _init_params(self) -> None:
        # He init for ReLU, Xavier-ish for sigmoid/tanh
        def init_w(fan_in: int, fan_out: int) -> np.ndarray:
            if self.act.name == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            return (self._rng.normal(0, 1, size=(fan_in, fan_out)).astype(np.float32) * scale).astype(
                np.float32
            )

        self.params["W1"] = init_w(self.input_dim, self.h1)
        self.params["b1"] = np.zeros((1, self.h1), dtype=np.float32)
        self.params["W2"] = init_w(self.h1, self.h2)
        self.params["b2"] = np.zeros((1, self.h2), dtype=np.float32)
        self.params["W3"] = init_w(self.h2, self.num_classes)
        self.params["b3"] = np.zeros((1, self.num_classes), dtype=np.float32)

        for k, v in self.params.items():
            self.vel[k] = np.zeros_like(v)

    def _forward(self, x: np.ndarray, training: bool) -> dict[str, np.ndarray]:
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        z1 = x @ W1 + b1
        a1 = self.act.f(z1).astype(np.float32)
        d1_mask = None
        if training and self.dropout > 0:
            keep_p = 1.0 - self.dropout
            d1_mask = (self._rng.random(a1.shape) < keep_p).astype(np.float32) / keep_p
            a1 = a1 * d1_mask

        z2 = a1 @ W2 + b2
        a2 = self.act.f(z2).astype(np.float32)
        d2_mask = None
        if training and self.dropout > 0:
            keep_p = 1.0 - self.dropout
            d2_mask = (self._rng.random(a2.shape) < keep_p).astype(np.float32) / keep_p
            a2 = a2 * d2_mask

        logits = a2 @ W3 + b3
        probs = softmax(logits).astype(np.float32)
        return {
            "x": x,
            "z1": z1,
            "a1": a1,
            "d1": d1_mask,
            "z2": z2,
            "a2": a2,
            "d2": d2_mask,
            "logits": logits,
            "probs": probs,
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        cache = self._forward(np.asarray(x, dtype=np.float32), training=False)
        return cache["probs"].argmax(axis=1)

    def _loss_and_acc(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        y = np.asarray(y).astype(int)
        y_oh = one_hot(y, self.num_classes)
        cache = self._forward(np.asarray(x, dtype=np.float32), training=False)
        loss = cross_entropy(y_oh, cache["probs"])
        if self.l2 > 0:
            loss += 0.5 * self.l2 * (
                float((self.params["W1"] ** 2).sum())
                + float((self.params["W2"] ** 2).sum())
                + float((self.params["W3"] ** 2).sum())
            )
        pred = cache["probs"].argmax(axis=1)
        acc = float((pred == y).mean())
        return loss, acc

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> "MLP":
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train).astype(int)
        x_val = np.asarray(x_val, dtype=np.float32)
        y_val = np.asarray(y_val).astype(int)

        for epoch in range(self.epochs):
            for xb, yb in minibatches(x_train, y_train, batch_size=self.batch_size, shuffle=True, seed=epoch):
                yb_oh = one_hot(yb, self.num_classes)
                cache = self._forward(xb.astype(np.float32), training=True)

                # Output delta for softmax+CE: (p - y)
                dlogits = (cache["probs"] - yb_oh).astype(np.float32) / xb.shape[0]

                grads: dict[str, np.ndarray] = {}
                grads["W3"] = cache["a2"].T @ dlogits + self.l2 * self.params["W3"]
                grads["b3"] = dlogits.sum(axis=0, keepdims=True)

                da2 = dlogits @ self.params["W3"].T
                if cache["d2"] is not None:
                    da2 = da2 * cache["d2"]
                dz2 = da2 * self.act.df(cache["z2"]).astype(np.float32)

                grads["W2"] = cache["a1"].T @ dz2 + self.l2 * self.params["W2"]
                grads["b2"] = dz2.sum(axis=0, keepdims=True)

                da1 = dz2 @ self.params["W2"].T
                if cache["d1"] is not None:
                    da1 = da1 * cache["d1"]
                dz1 = da1 * self.act.df(cache["z1"]).astype(np.float32)

                grads["W1"] = cache["x"].T @ dz1 + self.l2 * self.params["W1"]
                grads["b1"] = dz1.sum(axis=0, keepdims=True)

                # Momentum update
                for k in self.params:
                    self.vel[k] = self.momentum * self.vel[k] - self.lr * grads[k]
                    self.params[k] = self.params[k] + self.vel[k]

            tr_loss, tr_acc = self._loss_and_acc(x_train, y_train)
            va_loss, va_acc = self._loss_and_acc(x_val, y_val)
            self.history_["train_loss"].append(tr_loss)
            self.history_["train_acc"].append(tr_acc)
            self.history_["val_loss"].append(va_loss)
            self.history_["val_acc"].append(va_acc)

        return self

