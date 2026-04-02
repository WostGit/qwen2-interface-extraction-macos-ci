from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["source", "interface", "budget"], as_index=False)[["agreement", "kl_divergence"]]
        .agg(["mean", "std"])
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dfs = [pd.read_csv(p) for p in args.inputs]
    merged = pd.concat(dfs, ignore_index=True)
    summary = summarize(merged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
