from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from attacks.interfaces import apply_interface, agreement, kl_divergence
from experiments.common import set_seed, summarize_mean_std
from models.qwen_victim import QwenVictim


INTERFACES = ["argmax", "top2", "top3", "top5", "probs"]


def make_prompts(n: int) -> list[str]:
    return [f"Write one likely next token after: In year {1900 + i}," for i in range(n)]


def fit_student_global(train_targets: np.ndarray) -> np.ndarray:
    p = train_targets.mean(axis=0)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def run(seeds: list[int], budget: int, out_csv: Path, out_summary: Path, model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for seed in seeds:
        set_seed(seed)
        victim = QwenVictim(model_id=model_id)
        train_prompts = make_prompts(max(1200, budget))
        eval_prompts = make_prompts(128)
        victim_train = np.stack([victim.next_token_probs(p) for p in train_prompts])
        victim_eval = np.stack([victim.next_token_probs(p) for p in eval_prompts])

        for interface in INTERFACES:
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
    summary = summarize_mean_std(df, ["source", "interface", "budget"], ["agreement", "kl_divergence"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary.to_csv(out_summary, index=False)
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--out_csv", type=Path, default=Path("results/llm_topk_sweep.csv"))
    parser.add_argument("--out_summary", type=Path, default=Path("results/llm_topk_sweep_summary.csv"))
    args = parser.parse_args()
    run(args.seeds, args.budget, args.out_csv, args.out_summary, args.model_id)
