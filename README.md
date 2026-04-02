# qwen2-interface-extraction-macos-ci

Day-1 research repo for black-box model extraction experiments on GitHub Actions macOS runners using MLX end-to-end.

## What is implemented

- **Two experiment families**:
  1. Toy classifier baseline (`source=toy`)
  2. Tiny LLM next-token imitation with `Qwen/Qwen2-0.5B-Instruct` (`source=llm`)
- **Victim interfaces**:
  - Budget sweep: `argmax`, `top3`, `probs`
  - Fixed-budget top-k sweep: `argmax`, `top2`, `top3`, `top5`, `probs`
- **Day-1 sweeps**:
  - Query budgets: `64, 128, 256, 512, 1024`
  - Metrics logged per `source/interface/budget/seed`: `agreement`, `kl_divergence`
  - Multi-seed run (`0,1,2`) and summary mean/std CSV for fixed-budget top-k sweep
- **Plots**:
  - `results/agreement_vs_budget.png`
  - `results/kl_vs_budget.png`

## Repository structure

- `experiments/` sweep logic and plotting helpers
- `models/` toy model and MLX Qwen victim wrapper
- `attacks/` interface projection + metrics
- `scripts/` small per-job entry scripts
- `results/` outputs (CSV, PNG)
- `.github/workflows/` macOS CI workflow

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --results_dir results --seeds 0 1 2
```

## CI behavior

Workflow `.github/workflows/day1-macos-mlx.yml` runs all experiments on `macos-14`, caches pip and Hugging Face model files, and uploads all `results/*.csv` and `results/*.png` as artifacts.

## Notes on practicality

To keep runtime practical on GitHub Actions macOS runners, the LLM student in Day-1 is intentionally lightweight (global distribution estimator) while still exposing the interface-information effect on KL and agreement.
