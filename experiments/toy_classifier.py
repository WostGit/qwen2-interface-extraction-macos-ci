from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from models.interfaces import mean_kl_divergence, observe_interface, seed_everything, top1_agreement


@dataclass(frozen=True)
class ToyConfig:
    num_classes: int = 5
    num_features: int = 12
    num_train_pool: int = 2048
    num_eval: int = 512


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def _make_dataset(cfg: ToyConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = seed_everything(seed)
    x = rng.normal(size=(cfg.num_train_pool + cfg.num_eval, cfg.num_features))
    w = rng.normal(scale=0.8, size=(cfg.num_features, cfg.num_classes))
    logits = x @ w
    probs = _softmax(logits)
    return probs[: cfg.num_train_pool], probs[cfg.num_train_pool :]


def run_toy_budget_sweep(
    budgets: Iterable[int],
    interfaces: Iterable[str],
    seeds: Iterable[int],
    cfg: ToyConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ToyConfig()
    rows = []
    for seed in seeds:
        train_probs, eval_probs = _make_dataset(cfg, seed)
        for interface in interfaces:
            for budget in budgets:
                observed = []
                for i in range(min(budget, train_probs.shape[0])):
                    obs = observe_interface(train_probs[i], interface).vector
                    observed.append(obs)
                q = np.mean(np.stack(observed, axis=0), axis=0, keepdims=True)
                q = np.repeat(q, eval_probs.shape[0], axis=0)
                rows.append(
                    {
                        "family": "toy_classifier",
                        "source": "toy_linear_victim",
                        "interface": interface,
                        "budget": int(budget),
                        "seed": int(seed),
                        "agreement": top1_agreement(eval_probs, q),
                        "kl_divergence": mean_kl_divergence(eval_probs, q),
                    }
                )
    return pd.DataFrame(rows)
