from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from attacks.interfaces import apply_interface, agreement, kl_divergence
from experiments.common import set_seed
from models.qwen_victim import QwenVictim


INTERFACES = ["argmax", "top3", "probs"]
BUDGETS = [64, 128, 256, 512, 1024]


def make_prompts(n: int) -> list[str]:
    return [
        f"Complete the sentence with one likely next token: The number {i} in words is"
        for i in range(n)
    ]


def fit_student_global(train_targets: np.ndarray) -> np.ndarray:
    # lightweight student: a single global next-token distribution estimate.
    p = train_targets.mean(axis=0)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def run(seed: int, out_csv: Path, model_id: str) -> pd.DataFrame:
    set_seed(seed)
    victim = QwenVictim(model_id=model_id)
    train_prompts = make_prompts(1200)
    eval_prompts = make_prompts(128)

    victim_train = np.stack([victim.next_token_probs(p) for p in train_prompts])
    victim_eval = np.stack([victim.next_token_probs(p) for p in eval_prompts])

    rows: list[dict] = []
    for interface in INTERFACES:
        for budget in BUDGETS:
            idx = np.arange(budget)
            observed = np.stack([apply_interface(victim_train[i], interface) for i in idx])
            student_dist = fit_student_global(observed)
            pred_eval = np.repeat(student_dist[None, :], len(eval_prompts), axis=0)
            agr = np.mean([agreement(victim_eval[i], pred_eval[i]) for i in range(len(eval_prompts))])
            kl = np.mean([kl_divergence(victim_eval[i], pred_eval[i]) for i in range(len(eval_prompts))])
            rows.append(
                {
                    "source": "llm",
                    "interface": interface,
                    "budget": budget,
                    "seed": seed,
                    "agreement": agr,
                    "kl_divergence": kl,
                }
            )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--out_csv", type=Path, default=Path("results/llm_budget_sweep.csv"))
    args = parser.parse_args()
    run(args.seed, args.out_csv, args.model_id)
