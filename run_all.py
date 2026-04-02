from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from experiments.common import ensure_dir, plot_metric_vs_budget
from experiments.toy_budget_sweep import run as run_toy
from experiments.llm_budget_sweep import run as run_llm_budget
from experiments.llm_topk_sweep import run as run_llm_topk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    out = ensure_dir(args.results_dir)

    budget_frames = []
    for s in args.seeds:
        budget_frames.append(run_toy(s, out / f"toy_budget_seed{s}.csv"))
        budget_frames.append(run_llm_budget(s, out / f"llm_budget_seed{s}.csv", args.model_id))
    budget_df = pd.concat(budget_frames, ignore_index=True)
    budget_df.to_csv(out / "budget_sweep_all.csv", index=False)

    run_llm_topk(
        args.seeds,
        budget=256,
        out_csv=out / "llm_topk_sweep.csv",
        out_summary=out / "llm_topk_sweep_summary.csv",
        model_id=args.model_id,
    )

    plot_metric_vs_budget(
        budget_df,
        metric="agreement",
        out_file=out / "agreement_vs_budget.png",
        title="Agreement vs query budget",
    )
    plot_metric_vs_budget(
        budget_df,
        metric="kl_divergence",
        out_file=out / "kl_vs_budget.png",
        title="KL divergence vs query budget",
    )


if __name__ == "__main__":
    main()
