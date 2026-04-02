from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def summarize_mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    return df.groupby(group_cols, as_index=False).agg(**agg)


def plot_metric_vs_budget(df: pd.DataFrame, metric: str, out_file: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    grouped = df.groupby(["source", "interface", "budget"], as_index=False)[metric].mean()
    for (source, interface), sdf in grouped.groupby(["source", "interface"]):
        sdf = sdf.sort_values("budget")
        plt.plot(sdf["budget"], sdf[metric], marker="o", label=f"{source}:{interface}")
    plt.xscale("log", base=2)
    plt.xlabel("Query budget")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()
