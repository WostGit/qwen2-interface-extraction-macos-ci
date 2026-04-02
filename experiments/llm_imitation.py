from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from mlx_lm import load

from attacks.interface_adapter import apply_interface
from experiments.common import InterfaceConfig, format_interface_name, kl_divergence, top1_agreement
from models.simple_students import TinyNextTokenStudent


DEFAULT_CORPUS = [
    "Machine learning systems can be extracted through black-box interactions.",
    "Richer interfaces often reveal more about model internals and outputs.",
    "Budget-limited attackers face a tradeoff between coverage and precision.",
    "Next-token imitation can be measured by agreement and divergence metrics.",
    "Deterministic seeding improves experiment reproducibility in CI.",
    "Top-k and probability outputs expose more signal than argmax labels.",
    "MLX enables efficient inference and training on Apple silicon hardware.",
    "Qwen2 is used here as the victim model for tiny imitation experiments.",
]


@dataclass
class LLMConfig:
    model_name: str
    budgets: List[int]
    seeds: List[int]
    context_len: int = 16
    eval_size: int = 128
    train_steps: int = 40
    batch_size: int = 64
    lr: float = 5e-3
    student_emb_dim: int = 64
    student_hidden_dim: int = 128


def _build_contexts(tokenizer, target_size: int, context_len: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    token_stream: List[int] = []
    while len(token_stream) < target_size * (context_len + 2):
        text = rng.choice(DEFAULT_CORPUS)
        ids = tokenizer.encode(text)
        if len(ids) < context_len + 1:
            continue
        token_stream.extend(ids)

    contexts = []
    i = 0
    while len(contexts) < target_size and i + context_len < len(token_stream):
        contexts.append(token_stream[i : i + context_len])
        i += 1
    return np.array(contexts, dtype=np.int32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _victim_next_token_probs(model, contexts: np.ndarray, batch_size: int) -> np.ndarray:
    outs = []
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i : i + batch_size]
        logits = model(mx.array(batch))
        next_logits = np.array(logits[:, -1, :])
        outs.append(_softmax(next_logits))
    return np.concatenate(outs, axis=0)


def _train_student(contexts: np.ndarray, targets: np.ndarray, cfg: LLMConfig) -> TinyNextTokenStudent:
    vocab_size = int(targets.shape[1])
    student = TinyNextTokenStudent(vocab_size, emb_dim=cfg.student_emb_dim, hidden_dim=cfg.student_hidden_dim)
    opt = optim.Adam(learning_rate=cfg.lr)

    x_mx = mx.array(contexts)
    y_mx = mx.array(targets)

    def loss_fn(m: TinyNextTokenStudent, xb: mx.array, yb: mx.array) -> mx.array:
        logits = m(xb)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(mx.sum(yb * log_probs, axis=-1))

    loss_and_grad = nn.value_and_grad(student, loss_fn)

    for _ in range(cfg.train_steps):
        loss, grads = loss_and_grad(student, x_mx, y_mx)
        opt.update(student, grads)
        mx.eval(student.parameters(), opt.state)

    return student


def _shrink_vocab(probs: np.ndarray, keep_k: int = 128) -> tuple[np.ndarray, np.ndarray]:
    token_mass = probs.mean(axis=0)
    idx = np.argsort(-token_mass)[:keep_k]
    shrunk = probs[:, idx]
    shrunk = shrunk / shrunk.sum(axis=-1, keepdims=True)
    return shrunk, idx


def run_llm_budget_sweep(cfg: LLMConfig, interfaces: List[InterfaceConfig]) -> pd.DataFrame:
    model, tokenizer = load(cfg.model_name)

    rows = []
    for seed in cfg.seeds:
        max_budget = max(cfg.budgets)
        all_contexts = _build_contexts(tokenizer, max_budget + cfg.eval_size, cfg.context_len, seed)
        train_pool = all_contexts[:max_budget]
        eval_contexts = all_contexts[max_budget : max_budget + cfg.eval_size]

        p_eval_full = _victim_next_token_probs(model, eval_contexts, cfg.batch_size)
        p_eval, vocab_idx = _shrink_vocab(p_eval_full, keep_k=128)

        for budget in cfg.budgets:
            c_train = train_pool[:budget]
            p_full = _victim_next_token_probs(model, c_train, cfg.batch_size)
            p_train = p_full[:, vocab_idx]
            p_train = p_train / p_train.sum(axis=-1, keepdims=True)

            for iface in interfaces:
                y_api = apply_interface(p_train, iface)
                student = _train_student(c_train, y_api, cfg)
                logits = np.array(student(mx.array(eval_contexts)))
                p_student = _softmax(logits)

                rows.append(
                    {
                        "source": "llm",
                        "interface": format_interface_name(iface),
                        "budget": budget,
                        "seed": seed,
                        "agreement": top1_agreement(p_eval, p_student),
                        "kl_divergence": kl_divergence(p_eval, p_student),
                    }
                )

    return pd.DataFrame(rows)
