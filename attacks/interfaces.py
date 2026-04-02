"""Victim interface simulation and metrics."""

from __future__ import annotations

import numpy as np


def interface_observation(full_probs: np.ndarray, interface: str, topk: int = 3) -> np.ndarray:
    if interface == "probs":
        return full_probs
    if interface == "argmax":
        out = np.zeros_like(full_probs)
        out[np.arange(len(full_probs)), np.argmax(full_probs, axis=1)] = 1.0
        return out
    if interface in {"topk", "top2", "top3", "top5"}:
        k = topk
        idx = np.argsort(-full_probs, axis=1)[:, :k]
        out = np.zeros_like(full_probs)
        row = np.arange(len(full_probs))[:, None]
        vals = full_probs[row, idx]
        vals = vals / (vals.sum(axis=1, keepdims=True) + 1e-8)
        out[row, idx] = vals
        return out
    raise ValueError(f"Unknown interface: {interface}")


def top1_agreement(p: np.ndarray, q: np.ndarray) -> float:
    return float((np.argmax(p, axis=1) == np.argmax(q, axis=1)).mean())


def mean_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p2 = np.clip(p, eps, 1.0)
    q2 = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p2 * (np.log(p2) - np.log(q2)), axis=1)))
