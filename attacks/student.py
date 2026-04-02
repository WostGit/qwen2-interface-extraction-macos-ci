"""Simple linear softmax student with soft-label training."""

from __future__ import annotations

import numpy as np


class SoftmaxStudent:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(scale=0.01, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros((1, out_dim), dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(z)
        return ex / (ex.sum(axis=1, keepdims=True) + 1e-8)

    def fit(
        self,
        x: np.ndarray,
        y_soft: np.ndarray,
        epochs: int = 200,
        lr: float = 0.15,
        l2: float = 1e-4,
    ) -> None:
        n = x.shape[0]
        for _ in range(epochs):
            probs = self.predict_proba(x)
            grad_logits = (probs - y_soft) / n
            grad_w = x.T @ grad_logits + l2 * self.w
            grad_b = grad_logits.sum(axis=0, keepdims=True)
            self.w -= lr * grad_w
            self.b -= lr * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._softmax(x @ self.w + self.b)
