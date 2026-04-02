from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.interfaces import interface_observation, mean_kl, top1_agreement
from attacks.student import SoftmaxStudent
from experiments.config import (
    CACHE_DIR,
    LLM_FIXED_BUDGET,
    RESULTS_DIR,
    SEEDS,
    TOPK_MAP,
)
from experiments.data import hash_text_features, make_prompts


INTERFACES = ["argmax", "top2", "top3", "top5", "probs"]


def run(out_csv: Path) -> None:
    x_pool = hash_text_features(make_prompts(128))
    x_eval = hash_text_features(make_prompts(64))
    y_pool = np.load(CACHE_DIR / "llm_train_probs.npy")
    y_eval = np.load(CACHE_DIR / "llm_eval_probs.npy")

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_pool), size=LLM_FIXED_BUDGET, replace=True)
        x_q = x_pool[idx]
        y_q_full = y_pool[idx]
        for interface in INTERFACES:
            y_q = interface_observation(y_q_full, interface, topk=TOPK_MAP.get(interface, 3))
            student = SoftmaxStudent(x_pool.shape[1], y_pool.shape[1], seed=seed)
            student.fit(x_q, y_q, epochs=300, lr=0.12)
            pred = student.predict_proba(x_eval)
            rows.append(
                {
                    "source": "llm",
                    "interface": interface,
                    "budget": LLM_FIXED_BUDGET,
                    "seed": seed,
                    "agreement": top1_agreement(y_eval, pred),
                    "kl_divergence": mean_kl(y_eval, pred),
                }
            )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary = (
        df.groupby(["source", "interface", "budget"])[["agreement", "kl_divergence"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.to_csv(out_csv.with_name("llm_topk_sweep_summary.csv"), index=False)
    print(f"Wrote {out_csv} and summary")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "llm_topk_sweep.csv")
    args = parser.parse_args()
    run(args.out)
