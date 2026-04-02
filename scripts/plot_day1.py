from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for interface, group in df.groupby("interface"):
        means = group.groupby("budget")[metric].mean().sort_index()
        ax.plot(means.index, means.values, marker="o", label=interface)
    ax.set_xlabel("Query budget")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs budget")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    _plot_metric(df, "agreement", args.output_dir / "agreement_vs_budget.png")
    _plot_metric(df, "kl_divergence", args.output_dir / "kl_vs_budget.png")


if __name__ == "__main__":
    main()
