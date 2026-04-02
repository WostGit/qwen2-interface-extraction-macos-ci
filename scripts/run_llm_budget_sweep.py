from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.llm_imitation import run_llm_budget_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/llm_budget_sweep.csv"))
    args = parser.parse_args()

    budgets = [64, 128, 256, 512, 1024]
    interfaces = ["argmax", "topk", "probs"]
    seeds = [0]

    df = run_llm_budget_sweep(budgets=budgets, interfaces=interfaces, seeds=seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
