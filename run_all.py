from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    run([sys.executable, "scripts/run_toy_budget_sweep.py"])
    run([sys.executable, "scripts/run_llm_budget_sweep.py"])
    run([sys.executable, "scripts/run_llm_topk_sweep.py"])
    run([sys.executable, "scripts/summarize_results.py"])
    run([sys.executable, "scripts/plot_results.py"])


if __name__ == "__main__":
    main()
