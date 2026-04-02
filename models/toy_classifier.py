from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SoftmaxRegressor:
    in_dim: int
    out_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(scale=0.1, size=(self.in_dim, self.out_dim))
        self.b = np.zeros((self.out_dim,), dtype=np.float64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w + self.b
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit_soft_labels(
        self,
        x: np.ndarray,
        y_soft: np.ndarray,
        lr: float = 0.2,
        steps: int = 300,
        l2: float = 1e-4,
    ) -> None:
        n = x.shape[0]
        for _ in range(steps):
            pred = self.predict_proba(x)
            grad = (pred - y_soft) / n
            self.w -= lr * (x.T @ grad + l2 * self.w)
            self.b -= lr * grad.sum(axis=0)


def make_toy_data(n: int, seed: int, d: int = 8, c: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(c, d)) * 2.0
    y = rng.integers(0, c, size=n)
    x = means[y] + rng.normal(scale=1.0, size=(n, d))
    return x.astype(np.float64), y.astype(np.int64)
