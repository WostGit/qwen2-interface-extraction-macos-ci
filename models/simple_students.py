from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ToyStudent(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


class TinyNextTokenStudent(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def __call__(self, context_tokens: mx.array) -> mx.array:
        # context_tokens: [batch, context_len]
        emb = self.embed(context_tokens)
        pooled = mx.mean(emb, axis=1)
        return self.proj(pooled)
