from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import BUDGETS, available_interfaces, set_seed
from experiments.llm_imitation import LLMConfig, run_llm_budget_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/llm_budget_sweep.csv"))
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    budgets = [64, 128] if args.quick else BUDGETS
    cfg = LLMConfig(
        model_name=args.model_name,
        budgets=budgets,
        seeds=args.seeds,
        eval_size=32 if args.quick else 128,
        train_steps=10 if args.quick else 40,
    )

    set_seed(0)
    df = run_llm_budget_sweep(cfg, available_interfaces())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
