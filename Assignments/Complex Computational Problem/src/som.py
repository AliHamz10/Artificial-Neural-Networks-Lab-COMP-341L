from __future__ import annotations

import numpy as np


class SOM:
    """
    Kohonen Self-Organizing Map (2D grid) with Gaussian neighborhood and exponential decay.
    """

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        input_dim: int,
        lr0: float = 0.5,
        sigma0: float | None = None,
        epochs: int = 10,
        seed: int = 42,
    ):
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.input_dim = int(input_dim)
        self.lr0 = float(lr0)
        self.sigma0 = float(sigma0) if sigma0 is not None else float(max(grid_h, grid_w) / 2.0)
        self.epochs = int(epochs)
        self.seed = int(seed)

        self.weights: np.ndarray | None = None  # (H,W,D)
        self._coords = np.stack(
            np.meshgrid(np.arange(self.grid_h), np.arange(self.grid_w), indexing="ij"),
            axis=-1,
        ).astype(np.int32)  # (H,W,2)

    def _init_weights(self, x: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        # sample from data distribution for faster convergence
        idx = rng.integers(0, x.shape[0], size=(self.grid_h, self.grid_w))
        self.weights = x[idx].astype(np.float32).copy()

    def _decay(self, t: int, max_t: int) -> tuple[float, float]:
        lam = max_t / np.log(self.sigma0 + 1e-12)
        lr_t = self.lr0 * np.exp(-t / lam)
        sigma_t = self.sigma0 * np.exp(-t / lam)
        return float(lr_t), float(sigma_t)

    def bmu(self, x: np.ndarray) -> tuple[int, int]:
        if self.weights is None:
            raise RuntimeError("SOM not fitted.")
        w = self.weights.reshape(-1, self.input_dim)  # (H*W,D)
        d = np.linalg.norm(w - x.reshape(1, -1), axis=1)
        idx = int(np.argmin(d))
        return idx // self.grid_w, idx % self.grid_w

    def bmu_batch(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("SOM not fitted.")
        w = self.weights.reshape(-1, self.input_dim)  # (M,D)
        # compute squared distances efficiently: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        x = x.astype(np.float32)
        w2 = (w**2).sum(axis=1, keepdims=True)  # (M,1)
        x2 = (x**2).sum(axis=1, keepdims=True).T  # (1,N)
        dist2 = w2 + x2 - 2 * (w @ x.T)  # (M,N)
        idx = dist2.argmin(axis=0)  # (N,)
        rows = (idx // self.grid_w).astype(np.int32)
        cols = (idx % self.grid_w).astype(np.int32)
        return np.stack([rows, cols], axis=1)  # (N,2)

    def fit(self, x: np.ndarray) -> "SOM":
        x = np.asarray(x, dtype=np.float32)
        if self.weights is None:
            self._init_weights(x)
        assert self.weights is not None

        # NOTE: Full-grid updates per sample are extremely slow for MNIST.
        # We update only a local neighborhood window (≈3σ radius) around the BMU.
        max_t = self.epochs * x.shape[0]
        t = 0
        for _ in range(self.epochs):
            # shuffle each epoch for robustness
            idx = np.random.permutation(x.shape[0])
            x_epoch = x[idx]
            for xi in x_epoch:
                bmu_r, bmu_c = self.bmu(xi)
                lr_t, sigma_t = self._decay(t, max_t)

                # Gaussian neighborhood over a local window
                rad = int(np.ceil(3.0 * sigma_t))
                r0 = max(0, bmu_r - rad)
                r1 = min(self.grid_h - 1, bmu_r + rad)
                c0 = max(0, bmu_c - rad)
                c1 = min(self.grid_w - 1, bmu_c + rad)

                coords = self._coords[r0 : r1 + 1, c0 : c1 + 1]
                w_local = self.weights[r0 : r1 + 1, c0 : c1 + 1]

                diff = coords - np.array([bmu_r, bmu_c], dtype=np.int32)
                dist2 = (diff[..., 0] ** 2 + diff[..., 1] ** 2).astype(np.float32)
                h = np.exp(-dist2 / (2.0 * (sigma_t**2 + 1e-12))).astype(np.float32)  # (h,w)

                self.weights[r0 : r1 + 1, c0 : c1 + 1] = w_local + lr_t * h[..., None] * (xi - w_local)
                t += 1
        return self

    def u_matrix(self) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("SOM not fitted.")
        w = self.weights
        umat = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                neigh = []
                if r > 0:
                    neigh.append(w[r - 1, c])
                if r < self.grid_h - 1:
                    neigh.append(w[r + 1, c])
                if c > 0:
                    neigh.append(w[r, c - 1])
                if c < self.grid_w - 1:
                    neigh.append(w[r, c + 1])
                if neigh:
                    neigh = np.stack(neigh, axis=0)
                    umat[r, c] = np.linalg.norm(neigh - w[r, c], axis=1).mean()
        return umat

    def quantization_error(self, x: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError("SOM not fitted.")
        x = np.asarray(x, dtype=np.float32)
        bmus = self.bmu_batch(x)  # (N,2)
        w = self.weights[bmus[:, 0], bmus[:, 1]]
        return float(np.linalg.norm(x - w, axis=1).mean())

    def quantization_error_per_sample(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("SOM not fitted.")
        x = np.asarray(x, dtype=np.float32)
        bmus = self.bmu_batch(x)
        w = self.weights[bmus[:, 0], bmus[:, 1]]
        return np.linalg.norm(x - w, axis=1)

    def topographic_error(self, x: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError("SOM not fitted.")
        x = np.asarray(x, dtype=np.float32)
        w = self.weights.reshape(-1, self.input_dim)  # (M,D)
        # distances squared
        w2 = (w**2).sum(axis=1, keepdims=True)  # (M,1)
        x2 = (x**2).sum(axis=1, keepdims=True).T  # (1,N)
        dist2 = w2 + x2 - 2 * (w @ x.T)  # (M,N)
        # best and second best indices
        best = dist2.argmin(axis=0)
        # set best to +inf and find second
        dist2[best, np.arange(dist2.shape[1])] = np.inf
        second = dist2.argmin(axis=0)
        br, bc = best // self.grid_w, best % self.grid_w
        sr, sc = second // self.grid_w, second % self.grid_w
        manhattan = np.abs(br - sr) + np.abs(bc - sc)
        return float((manhattan > 1).mean())
