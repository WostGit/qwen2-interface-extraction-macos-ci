from __future__ import annotations

import numpy as np

from experiments.common import InterfaceConfig


def apply_interface(probs: np.ndarray, cfg: InterfaceConfig) -> np.ndarray:
    """Compress full victim probabilities according to query interface."""
    out = np.zeros_like(probs)
    if cfg.name == "argmax":
        idx = np.argmax(probs, axis=-1)
        out[np.arange(len(probs)), idx] = 1.0
        return out

    if cfg.name == "topk":
        k = int(cfg.top_k or 1)
        top_idx = np.argpartition(-probs, kth=np.clip(k - 1, 0, probs.shape[1] - 1), axis=-1)[:, :k]
        rows = np.arange(len(probs))[:, None]
        out[rows, top_idx] = probs[rows, top_idx]
        z = out.sum(axis=-1, keepdims=True)
        z = np.where(z == 0.0, 1.0, z)
        out = out / z
        return out

    if cfg.name == "probs":
        return probs.copy()

    raise ValueError(f"Unknown interface: {cfg}")
