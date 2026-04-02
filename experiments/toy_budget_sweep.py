from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from attacks.interfaces import apply_interface, agreement, kl_divergence
from experiments.common import ensure_dir, set_seed
from models.toy_classifier import SoftmaxRegressor, make_toy_data


INTERFACES = ["argmax", "top3", "probs"]
BUDGETS = [64, 128, 256, 512, 1024]


def run(seed: int, out_csv: Path) -> pd.DataFrame:
    set_seed(seed)
    x_train, _ = make_toy_data(3000, seed=seed + 1)
    x_eval, _ = make_toy_data(600, seed=seed + 2)

    victim = SoftmaxRegressor(in_dim=x_train.shape[1], out_dim=5, seed=seed + 10)
    # train victim on synthetic hard labels for realism
    y_hard = np.argmax(victim.predict_proba(x_train), axis=1)
    y_onehot = np.eye(5)[y_hard]
    victim.fit_soft_labels(x_train, y_onehot, lr=0.15, steps=250)

    victim_train_probs = victim.predict_proba(x_train)
    victim_eval_probs = victim.predict_proba(x_eval)

    rows: list[dict] = []
    for interface in INTERFACES:
        for budget in BUDGETS:
            idx = np.arange(budget)
            queried = np.stack([apply_interface(victim_train_probs[i], interface) for i in idx])
            student = SoftmaxRegressor(in_dim=x_train.shape[1], out_dim=5, seed=seed + 100 + budget)
            student.fit_soft_labels(x_train[idx], queried, lr=0.2, steps=300)
            pred = student.predict_proba(x_eval)
            agr = np.mean([agreement(victim_eval_probs[i], pred[i]) for i in range(len(x_eval))])
            kl = np.mean([kl_divergence(victim_eval_probs[i], pred[i]) for i in range(len(x_eval))])
            rows.append(
                {
                    "source": "toy",
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
    parser.add_argument("--out_csv", type=Path, default=Path("results/toy_budget_sweep.csv"))
    args = parser.parse_args()
    run(args.seed, args.out_csv)
