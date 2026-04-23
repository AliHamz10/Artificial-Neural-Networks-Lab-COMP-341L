from __future__ import annotations

import numpy as np


class AdalineGD:
    """
    Adaline trained with batch gradient descent on MSE using continuous linear output.
    Labels can be {0,1} (internally used as floats).
    """

    def __init__(self, lr: float = 0.001, epochs: int = 50, seed: int = 42):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.w: np.ndarray | None = None
        self.history_: dict[str, list[float] | list[np.ndarray]] = {"mse": [], "w": []}

    @staticmethod
    def _add_bias(x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.ones((x.shape[0], 1), dtype=x.dtype), x], axis=1)

    def net(self, x: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        xb = self._add_bias(np.asarray(x, dtype=np.float32))
        return xb @ self.w

    def predict(self, x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        y_lin = self.net(x)
        return (y_lin >= threshold).astype(int)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AdalineGD":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        xb = self._add_bias(x)

        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, size=(xb.shape[1],)).astype(np.float32)

        n = xb.shape[0]
        for _ in range(self.epochs):
            y_hat = xb @ self.w
            err = y - y_hat
            grad = -(xb.T @ err) / n  # d/dw (1/2n * sum err^2) = -(X^T err)/n
            self.w = self.w - self.lr * grad
            mse = float((err**2).mean() / 2.0)
            self.history_["mse"].append(mse)
            self.history_["w"].append(self.w.copy())
        return self


def mse_loss_surface(
    x2: np.ndarray,
    y: np.ndarray,
    w0_range: tuple[float, float] = (-1.0, 1.0),
    w1_range: tuple[float, float] = (-2.0, 2.0),
    grid: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes MSE surface for 2D input with bias fixed at 0.
    Returns (W0, W1, Z) where Z is 1/2 * mean((y - (w0*x1 + w1*x2))^2).
    """
    x2 = np.asarray(x2, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    w0 = np.linspace(w0_range[0], w0_range[1], grid)
    w1 = np.linspace(w1_range[0], w1_range[1], grid)
    W0, W1 = np.meshgrid(w0, w1)
    z = np.zeros_like(W0, dtype=np.float32)
    x1 = x2[:, 0]
    x2v = x2[:, 1]
    for i in range(grid):
        for j in range(grid):
            y_hat = W0[i, j] * x1 + W1[i, j] * x2v
            z[i, j] = 0.5 * np.mean((y - y_hat) ** 2)
    return W0, W1, z

