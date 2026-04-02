from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.config import PLOTS_DIR, RESULTS_DIR


def plot_budget(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, source in zip(axes, ["toy", "llm"]):
        sub = df[df["source"] == source]
        summary = (
            sub.groupby(["interface", "budget"])[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("budget")
        )
        for interface, part in summary.groupby("interface"):
            ax.plot(part["budget"], part["mean"], marker="o", label=interface)
            ax.fill_between(
                part["budget"],
                part["mean"] - part["std"].fillna(0),
                part["mean"] + part["std"].fillna(0),
                alpha=0.2,
            )
        ax.set_title(source.upper())
        ax.set_xlabel("query budget")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric)
    axes[1].legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(results_dir: Path) -> None:
    toy = pd.read_csv(results_dir / "toy_budget_sweep.csv")
    llm = pd.read_csv(results_dir / "llm_budget_sweep.csv")
    all_df = pd.concat([toy, llm], ignore_index=True)
    plot_budget(all_df, "agreement", PLOTS_DIR / "agreement_vs_budget.png")
    plot_budget(all_df, "kl_divergence", PLOTS_DIR / "kl_vs_budget.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    run(args.results_dir)
