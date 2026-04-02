from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

import mlx.core as mx
from mlx_lm import load


@dataclass
class QwenVictim:
    model_id: str = "Qwen/Qwen2-0.5B-Instruct"

    def __post_init__(self) -> None:
        self.model, self.tokenizer = load(self.model_id)

    @lru_cache(maxsize=8192)
    def next_token_probs(self, prompt: str) -> np.ndarray:
        toks = self.tokenizer.encode(prompt)
        x = mx.array([toks], dtype=mx.int32)
        logits = self.model(x)
        # [batch, seq, vocab]
        last = np.array(logits[:, -1, :])[0]
        last = last - np.max(last)
        e = np.exp(last)
        return (e / e.sum()).astype(np.float64)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)
