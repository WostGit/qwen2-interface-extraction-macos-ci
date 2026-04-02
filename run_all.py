from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    root = Path(__file__).parent
    run([sys.executable, str(root / "scripts/run_toy_budget_sweep.py")])
    run([sys.executable, str(root / "scripts/run_llm_budget_sweep.py")])
    run([sys.executable, str(root / "scripts/run_llm_topk_fixed.py")])

    run(
        [
            sys.executable,
            str(root / "scripts/plot_day1.py"),
            "--input",
            str(root / "results/llm_budget_sweep.csv"),
            "--output-dir",
            str(root / "results"),
        ]
    )


if __name__ == "__main__":
    main()
