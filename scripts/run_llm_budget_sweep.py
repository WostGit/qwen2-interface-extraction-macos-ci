from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.interfaces import interface_observation, mean_kl, top1_agreement
from attacks.student import SoftmaxStudent
from experiments.config import (
    BUDGETS,
    CACHE_DIR,
    INTERFACES_DAY1,
    LLM_EVAL_PROMPTS,
    LLM_MODEL_NAME,
    LLM_TRAIN_PROMPTS,
    LLM_VOCAB_TOPN,
    RESULTS_DIR,
    SEEDS,
    TOPK_MAP,
)
from experiments.data import hash_text_features, make_prompts
from models.llm import LLMVictim


def _load_or_build_cache(victim: LLMVictim):
    train_cache = CACHE_DIR / "llm_train_probs.npy"
    eval_cache = CACHE_DIR / "llm_eval_probs.npy"
    idx_cache = CACHE_DIR / "llm_keep_idx.npy"

    train_prompts = make_prompts(LLM_TRAIN_PROMPTS)
    eval_prompts = make_prompts(LLM_EVAL_PROMPTS)

    if train_cache.exists() and eval_cache.exists() and idx_cache.exists():
        y_train = np.load(train_cache)
        y_eval = np.load(eval_cache)
        keep_idx = np.load(idx_cache)
    else:
        y_train, keep_idx = victim.build_prompt_matrix(train_prompts, LLM_VOCAB_TOPN)
        y_eval = victim.prompt_matrix_with_index(eval_prompts, keep_idx)
        np.save(train_cache, y_train)
        np.save(eval_cache, y_eval)
        np.save(idx_cache, keep_idx)

    x_train = hash_text_features(train_prompts)
    x_eval = hash_text_features(eval_prompts)
    return x_train, y_train, x_eval, y_eval


def run(out_csv: Path) -> None:
    victim = LLMVictim(LLM_MODEL_NAME)
    x_pool, y_pool, x_eval, y_eval = _load_or_build_cache(victim)

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for budget in BUDGETS:
            idx = rng.choice(len(x_pool), size=budget, replace=budget > len(x_pool))
            x_q = x_pool[idx]
            y_q_full = y_pool[idx]
            for interface in INTERFACES_DAY1:
                y_q = interface_observation(y_q_full, interface, topk=TOPK_MAP.get(interface, 3))
                student = SoftmaxStudent(x_pool.shape[1], y_pool.shape[1], seed=seed)
                student.fit(x_q, y_q, epochs=300, lr=0.12)
                pred = student.predict_proba(x_eval)
                rows.append(
                    {
                        "source": "llm",
                        "interface": interface,
                        "budget": budget,
                        "seed": seed,
                        "agreement": top1_agreement(y_eval, pred),
                        "kl_divergence": mean_kl(y_eval, pred),
                    }
                )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "llm_budget_sweep.csv")
    args = parser.parse_args()
    run(args.out)
