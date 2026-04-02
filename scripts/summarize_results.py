from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.config import RESULTS_DIR


def summarize(results_dir: Path) -> None:
    frames = [
        pd.read_csv(results_dir / "toy_budget_sweep.csv"),
        pd.read_csv(results_dir / "llm_budget_sweep.csv"),
        pd.read_csv(results_dir / "llm_topk_sweep.csv"),
    ]
    df = pd.concat(frames, ignore_index=True)
    summary = (
        df.groupby(["source", "interface", "budget"])[["agreement", "kl_divergence"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.to_csv(results_dir / "summary_mean_std.csv", index=False)
    print("Wrote summary_mean_std.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    summarize(args.results_dir)
