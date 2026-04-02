"""Toy classifier victim model."""

from __future__ import annotations

import numpy as np


class ToyVictim:
    def __init__(self, in_dim: int, num_classes: int, seed: int = 123):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(size=(in_dim, num_classes)).astype(np.float32)
        self.b = rng.normal(size=(1, num_classes)).astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w + self.b
        z -= z.max(axis=1, keepdims=True)
        ex = np.exp(z)
        return ex / (ex.sum(axis=1, keepdims=True) + 1e-8)
