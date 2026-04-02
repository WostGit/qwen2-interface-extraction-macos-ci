"""MLX-backed next-token victim model helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import mlx.core as mx
import numpy as np
from mlx_lm import load


@dataclass
class LLMVictim:
    model_name: str

    def __post_init__(self) -> None:
        self.model, self.tokenizer = load(self.model_name)

    def next_token_probs(self, prompt: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [self.tokenizer.eos_token_id]
        x = mx.array([token_ids])
        logits, _ = self.model(x)
        last_logits = np.array(logits[:, -1, :], dtype=np.float32)
        last_logits -= last_logits.max(axis=1, keepdims=True)
        ex = np.exp(last_logits)
        probs = ex / (ex.sum(axis=1, keepdims=True) + 1e-8)
        return probs[0]

    def build_prompt_matrix(
        self,
        prompts: List[str],
        topn: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw = [self.next_token_probs(p) for p in prompts]
        arr = np.stack(raw)
        avg = arr.mean(axis=0)
        keep_idx = np.argsort(-avg)[:topn]
        reduced = arr[:, keep_idx]
        reduced = reduced / (reduced.sum(axis=1, keepdims=True) + 1e-8)
        return reduced, keep_idx

    def prompt_matrix_with_index(self, prompts: List[str], keep_idx: np.ndarray) -> np.ndarray:
        raw = [self.next_token_probs(p) for p in prompts]
        arr = np.stack(raw)
        reduced = arr[:, keep_idx]
        return reduced / (reduced.sum(axis=1, keepdims=True) + 1e-8)
