from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import BUDGETS, available_interfaces, set_seed
from experiments.toy_classifier import ToyConfig, run_toy_budget_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/toy_budget_sweep.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    budgets = [64, 128] if args.quick else BUDGETS
    cfg = ToyConfig(budgets=budgets, seeds=args.seeds, train_steps=25 if args.quick else 60)

    set_seed(0)
    df = run_toy_budget_sweep(cfg, available_interfaces())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
