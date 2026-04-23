from __future__ import annotations

import numpy as np


class Perceptron:
    """
    Binary perceptron with step activation.
    Uses labels y in {0,1}.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 25, seed: int = 42):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.w: np.ndarray | None = None
        self.history_: dict[str, list] = {"miscls": [], "w": []}

    @staticmethod
    def _add_bias(x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.ones((x.shape[0], 1), dtype=x.dtype), x], axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        xb = self._add_bias(np.asarray(x, dtype=np.float32))
        y_hat = (xb @ self.w >= 0).astype(int)
        return y_hat

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Perceptron":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y).astype(int)
        xb = self._add_bias(x)

        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, size=(xb.shape[1],)).astype(np.float32)

        for _ in range(self.epochs):
            miscls = 0
            for xi, di in zip(xb, y):
                yi = 1 if (xi @ self.w) >= 0 else 0
                err = di - yi
                if err != 0:
                    miscls += 1
                    self.w = self.w + self.lr * err * xi
            self.history_["miscls"].append(int(miscls))
            self.history_["w"].append(self.w.copy())
        return self

