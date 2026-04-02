from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 1 extraction experiments end-to-end.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="Fast smoke config for CI/PRs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    quick_flag = ["--quick"] if args.quick else []

    toy_csv = args.output_dir / "toy_budget_sweep.csv"
    llm_csv = args.output_dir / "llm_budget_sweep.csv"
    topk_csv = args.output_dir / "llm_topk_sweep.csv"
    merged_csv = args.output_dir / "day1_budget_sweep_all.csv"
    summary_csv = args.output_dir / "day1_multi_seed_summary.csv"

    run([sys.executable, "scripts/run_toy_budget_sweep.py", "--output", str(toy_csv), *quick_flag])
    run([sys.executable, "scripts/run_llm_budget_sweep.py", "--output", str(llm_csv), *quick_flag])
    run([sys.executable, "scripts/run_llm_topk_sweep.py", "--output", str(topk_csv), *quick_flag])

    run(
        [
            sys.executable,
            "-c",
            (
                "import pandas as pd; "
                f"a=pd.read_csv(r'{toy_csv}'); b=pd.read_csv(r'{llm_csv}'); "
                f"pd.concat([a,b],ignore_index=True).to_csv(r'{merged_csv}',index=False)"
            ),
        ]
    )

    run(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--inputs",
            str(toy_csv),
            str(llm_csv),
            str(topk_csv),
            "--output",
            str(summary_csv),
        ]
    )

    run(
        [
            sys.executable,
            "scripts/plot_results.py",
            "--input-csv",
            str(merged_csv),
            "--output-dir",
            str(args.output_dir),
        ]
    )

    print("Done. Outputs:")
    for p in [toy_csv, llm_csv, topk_csv, merged_csv, summary_csv]:
        print(" -", p)


if __name__ == "__main__":
    main()
