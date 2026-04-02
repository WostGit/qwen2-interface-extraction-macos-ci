from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from models.interfaces import mean_kl_divergence, observe_interface, seed_everything, top1_agreement


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    max_pool: int = 1200
    eval_size: int = 128
    candidate_tokens: int = 64


def _build_prompts(max_pool: int) -> list[str]:
    seeds = [
        "The quick brown fox",
        "In a future powered by AI",
        "A practical tip for debugging is",
        "The capital of France is",
        "On macOS with MLX",
        "Data extraction attacks can",
    ]
    prompts = []
    i = 0
    while len(prompts) < max_pool:
        base = seeds[i % len(seeds)]
        prompts.append(f"{base} sample #{i}:" )
        i += 1
    return prompts


def _load_mlx():
    from mlx_lm import load

    model, tokenizer = load(LLMConfig.model_name)
    return model, tokenizer


def _next_token_distribution(model, tokenizer, prompt: str, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    import mlx.core as mx

    tokenized = tokenizer(prompt, return_tensors="np")
    input_ids = mx.array(tokenized["input_ids"])
    logits = model(input_ids)
    last_logits = np.array(logits[:, -1, :])[0]

    shifted = last_logits - np.max(last_logits)
    probs = np.exp(shifted)
    probs /= np.sum(probs)

    top_idx = np.argpartition(probs, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
    top_probs = probs[top_idx]
    return top_idx.astype(np.int32), top_probs.astype(np.float64)


def _materialize_dataset(cfg: LLMConfig) -> tuple[np.ndarray, np.ndarray]:
    model, tokenizer = _load_mlx()
    prompts = _build_prompts(cfg.max_pool + cfg.eval_size)

    token_bank = {}
    rows = []
    for prompt in prompts:
        idx, probs = _next_token_distribution(model, tokenizer, prompt, cfg.candidate_tokens)
        for t in idx:
            token_bank[int(t)] = len(token_bank)
        rows.append((idx, probs))

    vocab_size = len(token_bank) + 1  # +1 tail bucket
    dense = np.zeros((len(rows), vocab_size), dtype=np.float64)

    for i, (idx, probs) in enumerate(rows):
        placed = 0.0
        for t, p in zip(idx, probs):
            dense[i, token_bank[int(t)]] = p
            placed += p
        dense[i, -1] = max(0.0, 1.0 - placed)
        dense[i] /= np.maximum(dense[i].sum(), 1e-12)

    return dense[: cfg.max_pool], dense[cfg.max_pool : cfg.max_pool + cfg.eval_size]


def run_llm_budget_sweep(
    budgets: Iterable[int],
    interfaces: Iterable[str],
    seeds: Iterable[int],
    cfg: LLMConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or LLMConfig()
    train_probs, eval_probs = _materialize_dataset(cfg)
    rows = []

    for seed in seeds:
        rng = seed_everything(seed)
        for interface in interfaces:
            for budget in budgets:
                indices = rng.integers(low=0, high=train_probs.shape[0], size=budget)
                observed = [observe_interface(train_probs[i], interface).vector for i in indices]
                q = np.mean(np.stack(observed, axis=0), axis=0, keepdims=True)
                q = np.repeat(q, eval_probs.shape[0], axis=0)

                rows.append(
                    {
                        "family": "llm_next_token",
                        "source": "qwen2_0.5b",
                        "interface": interface,
                        "budget": int(budget),
                        "seed": int(seed),
                        "agreement": top1_agreement(eval_probs, q),
                        "kl_divergence": mean_kl_divergence(eval_probs, q),
                    }
                )
    return pd.DataFrame(rows)


def run_llm_topk_sweep(fixed_budget: int, seeds: Iterable[int]) -> pd.DataFrame:
    interfaces = ["argmax", "top2", "top3", "top5", "probs"]
    return run_llm_budget_sweep([fixed_budget], interfaces, seeds)
