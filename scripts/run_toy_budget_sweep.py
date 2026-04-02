from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.interfaces import interface_observation, mean_kl, top1_agreement
from attacks.student import SoftmaxStudent
from experiments.config import (
    BUDGETS,
    INTERFACES_DAY1,
    RESULTS_DIR,
    SEEDS,
    TOY_EVAL_SIZE,
    TOY_FEATURE_DIM,
    TOY_NUM_CLASSES,
    TOY_TRAIN_POOL,
    TOPK_MAP,
)
from experiments.data import make_toy_features
from models.toy import ToyVictim


def run(out_csv: Path) -> None:
    victim = ToyVictim(TOY_FEATURE_DIM, TOY_NUM_CLASSES, seed=77)
    x_pool = make_toy_features(TOY_TRAIN_POOL, TOY_FEATURE_DIM, seed=11)
    x_eval = make_toy_features(TOY_EVAL_SIZE, TOY_FEATURE_DIM, seed=29)
    y_pool = victim.predict_proba(x_pool)
    y_eval = victim.predict_proba(x_eval)

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for budget in BUDGETS:
            idx = rng.choice(len(x_pool), size=budget, replace=budget > len(x_pool))
            x_q = x_pool[idx]
            y_q_full = y_pool[idx]
            for interface in INTERFACES_DAY1:
                y_q = interface_observation(y_q_full, interface, topk=TOPK_MAP.get(interface, 3))
                student = SoftmaxStudent(TOY_FEATURE_DIM, TOY_NUM_CLASSES, seed=seed)
                student.fit(x_q, y_q, epochs=250, lr=0.2)
                pred = student.predict_proba(x_eval)
                rows.append(
                    {
                        "source": "toy",
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
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "toy_budget_sweep.csv")
    args = parser.parse_args()
    run(args.out)
