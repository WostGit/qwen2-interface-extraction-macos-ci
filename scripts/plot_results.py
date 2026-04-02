from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = (
        df.groupby(["source", "interface", "budget"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    for (source, interface), sdf in grouped.groupby(["source", "interface"]):
        sdf = sdf.sort_values("budget")
        label = f"{source}:{interface}"
        ax.plot(sdf["budget"], sdf["mean"], marker="o", label=label)
        ax.fill_between(
            sdf["budget"],
            sdf["mean"] - sdf["std"].fillna(0.0),
            sdf["mean"] + sdf["std"].fillna(0.0),
            alpha=0.15,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Query budget")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    _plot_metric(df, "agreement", args.output_dir / "agreement_vs_budget.png")
    _plot_metric(df, "kl_divergence", args.output_dir / "kl_vs_budget.png")


if __name__ == "__main__":
    main()
