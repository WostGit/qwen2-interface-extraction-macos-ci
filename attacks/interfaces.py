from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VictimResponse:
    probs: np.ndarray


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=-1, keepdims=True)


def apply_interface(probs: np.ndarray, interface: str) -> np.ndarray:
    """Project a full probability vector into a restricted API interface.

    interface values:
      - argmax: one-hot at max class/token
      - topk where k is integer suffix, e.g. top2, top5
      - probs: full distribution
    """
    probs = normalize(probs)
    if interface == "probs":
        return probs
    if interface == "argmax":
        out = np.zeros_like(probs)
        out[np.argmax(probs)] = 1.0
        return out
    if interface.startswith("top"):
        k = int(interface[3:])
        out = np.zeros_like(probs)
        idx = np.argpartition(probs, -k)[-k:]
        out[idx] = probs[idx]
        return normalize(out)
    raise ValueError(f"Unknown interface: {interface}")


def agreement(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.argmax(p) == np.argmax(q))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = normalize(p)
    q = normalize(q)
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))
