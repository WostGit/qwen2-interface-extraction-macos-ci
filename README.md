# qwen2-interface-extraction-macos-ci

Research repo for black-box model extraction experiments on GitHub Actions macOS runners using MLX.

## What is implemented

- **Family 1: toy classifier baseline** (`scripts/run_toy_budget_sweep.py`)
- **Family 2: tiny LLM next-token imitation** with **Qwen2-0.5B via MLX** (`scripts/run_llm_budget_sweep.py`)
- Victim API interfaces: `argmax`, `topk`, and `probs`
- Day 1 budget sweep: `64, 128, 256, 512, 1024`
- Fixed-budget top-k sweep for LLM: `argmax, top2, top3, top5, probs`
- Multi-seed summaries with mean/std (`scripts/summarize_results.py`)
- Plots:
  - `results/plots/agreement_vs_budget.png`
  - `results/plots/kl_vs_budget.png`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Outputs are written under `results/` as CSV files and PNG plots.

## CI

The workflow `.github/workflows/macos_mlx_extraction.yml` runs the full pipeline on `macos-14`, caches model downloads, and uploads all CSV/PNG outputs as artifacts.
