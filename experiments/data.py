"""Data generation and feature helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_toy_features(n: int, dim: int, seed: int) -> np.ndarray:
    rng = set_seed(seed)
    x = rng.normal(size=(n, dim)).astype(np.float32)
    x[:, : dim // 2] += rng.normal(scale=0.3, size=(n, dim // 2))
    return x


def make_prompts(n: int) -> List[str]:
    templates = [
        "Summarize the key idea of {} in one sentence.",
        "Given topic {}, provide a short definition.",
        "Explain why {} matters for beginners.",
        "Write a concise bullet about {}.",
        "Complete the statement: {} is useful because",
    ]
    topics = [
        "linear algebra",
        "entropy",
        "macOS automation",
        "language models",
        "unit testing",
        "gradient descent",
        "open source",
        "API design",
        "privacy",
        "optimization",
    ]
    prompts = []
    for idx in range(n):
        prompts.append(templates[idx % len(templates)].format(f"{topics[idx % len(topics)]} #{idx}"))
    return prompts


def hash_text_features(texts: Iterable[str], dim: int = 256) -> np.ndarray:
    texts = list(texts)
    feats = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % dim
            feats[i, idx] += 1.0
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    return feats / norms
