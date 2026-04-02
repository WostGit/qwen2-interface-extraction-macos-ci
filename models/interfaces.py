from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class InterfaceObservation:
    """Interface-specific victim response for one query."""

    vector: np.ndarray


def observe_interface(probs: np.ndarray, interface: str) -> InterfaceObservation:
    """Return interface-limited observation for a single probability vector."""
    if interface == "argmax":
        out = np.zeros_like(probs)
        out[np.argmax(probs)] = 1.0
        return InterfaceObservation(out)

    if interface.startswith("top") and interface != "topk":
        k = int(interface[3:])
    elif interface == "topk":
        k = 5
    elif interface == "probs":
        return InterfaceObservation(probs.copy())
    else:
        raise ValueError(f"Unsupported interface: {interface}")

    top_idx = np.argpartition(probs, -k)[-k:]
    out = np.zeros_like(probs)
    top_probs = probs[top_idx]
    out[top_idx] = top_probs / np.maximum(top_probs.sum(), 1e-12)
    return InterfaceObservation(out)


def mean_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=1)))


def top1_agreement(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == np.argmax(q, axis=1)))


def seed_everything(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
