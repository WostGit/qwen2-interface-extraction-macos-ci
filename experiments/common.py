from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import mlx.core as mx
import numpy as np


BUDGETS = [64, 128, 256, 512, 1024]


@dataclass(frozen=True)
class InterfaceConfig:
    name: str
    top_k: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def normalize_probs(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.maximum(x, eps)
    return x / x.sum(axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = normalize_probs(p, eps=eps)
    q = normalize_probs(q, eps=eps)
    return float(np.mean(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1)))


def top1_agreement(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=-1) == np.argmax(q, axis=-1)))


def available_interfaces() -> List[InterfaceConfig]:
    return [
        InterfaceConfig("argmax"),
        InterfaceConfig("topk", top_k=3),
        InterfaceConfig("probs"),
    ]


def llm_topk_interfaces() -> List[InterfaceConfig]:
    return [
        InterfaceConfig("argmax"),
        InterfaceConfig("topk", top_k=2),
        InterfaceConfig("topk", top_k=3),
        InterfaceConfig("topk", top_k=5),
        InterfaceConfig("probs"),
    ]


def format_interface_name(cfg: InterfaceConfig) -> str:
    if cfg.name == "topk" and cfg.top_k:
        return f"top{cfg.top_k}"
    return cfg.name
