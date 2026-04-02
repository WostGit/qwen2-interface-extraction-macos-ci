from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd

from attacks.interface_adapter import apply_interface
from experiments.common import InterfaceConfig, format_interface_name, kl_divergence, top1_agreement
from models.simple_students import ToyStudent


@dataclass
class ToyConfig:
    budgets: List[int]
    seeds: List[int]
    train_steps: int = 60
    in_dim: int = 12
    num_classes: int = 5
    hidden_dim: int = 32
    lr: float = 2e-2
    eval_size: int = 256


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _make_toy_victim(seed: int, in_dim: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 1, size=(in_dim, 24))
    w2 = rng.normal(0, 1, size=(24, num_classes))
    return w1, w2


def _victim_probs(x: np.ndarray, params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    w1, w2 = params
    h = np.tanh(x @ w1)
    logits = h @ w2
    return _softmax(logits)


def _train_student(x: np.ndarray, y_probs: np.ndarray, cfg: ToyConfig) -> ToyStudent:
    model = ToyStudent(cfg.in_dim, cfg.hidden_dim, cfg.num_classes)
    opt = optim.Adam(learning_rate=cfg.lr)

    x_mx = mx.array(x)
    y_mx = mx.array(y_probs)

    def loss_fn(m: ToyStudent, xb: mx.array, yb: mx.array) -> mx.array:
        logits = m(xb)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(mx.sum(yb * log_probs, axis=-1))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for _ in range(cfg.train_steps):
        loss, grads = loss_and_grad(model, x_mx, y_mx)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)

    return model


def run_toy_budget_sweep(exp_cfg: ToyConfig, interfaces: List[InterfaceConfig]) -> pd.DataFrame:
    rows = []
    for seed in exp_cfg.seeds:
        victim = _make_toy_victim(seed, exp_cfg.in_dim, exp_cfg.num_classes)
        rng = np.random.default_rng(seed + 123)

        x_eval = rng.normal(size=(exp_cfg.eval_size, exp_cfg.in_dim)).astype(np.float32)
        p_eval = _victim_probs(x_eval, victim)

        for budget in exp_cfg.budgets:
            x_train = rng.normal(size=(budget, exp_cfg.in_dim)).astype(np.float32)
            p_full = _victim_probs(x_train, victim)

            for iface in interfaces:
                y_api = apply_interface(p_full, iface)
                student = _train_student(x_train, y_api, exp_cfg)

                logits = np.array(student(mx.array(x_eval)))
                p_student = _softmax(logits)

                rows.append(
                    {
                        "source": "toy",
                        "interface": format_interface_name(iface),
                        "budget": budget,
                        "seed": seed,
                        "agreement": top1_agreement(p_eval, p_student),
                        "kl_divergence": kl_divergence(p_eval, p_student),
                    }
                )

    return pd.DataFrame(rows)
