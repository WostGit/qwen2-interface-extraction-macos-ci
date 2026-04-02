# qwen2-interface-extraction-macos-ci

Research repository for black-box model extraction experiments run on **GitHub Actions macOS runners** with **Python + MLX**.

## Day 1 goals implemented

- Two experiment families:
  1. **Toy classifier baseline** extraction.
  2. **Tiny LLM next-token imitation** against `Qwen/Qwen2-0.5B` via `mlx-lm`.
- Victim API interfaces compared:
  - `argmax`
  - `topk`
  - `probs`
- Budget sweep over query counts: `64, 128, 256, 512, 1024`.
- Logged metrics per source/interface/budget/seed:
  - `agreement`
  - `kl_divergence`
- Plot generation:
  - `agreement_vs_budget.png`
  - `kl_vs_budget.png`
- Fixed-budget LLM top-k sweep at one budget:
  - `argmax`, `top2`, `top3`, `top5`, `probs`
- Multi-seed run reporting mean/std summaries.
- CI artifacts upload CSV summaries and PNG plots.

## Repository structure

- `experiments/` core experiment logic
- `models/` student model implementations
- `attacks/` black-box query adapters
- `scripts/` focused runners per job
- `results/` CSV and plot outputs
- `.github/workflows/` CI orchestration for macOS
- `run_all.py` one-command Day 1 entrypoint

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --output-dir results
```

For a faster smoke run:

```bash
python run_all.py --quick --output-dir results
```

## Notes on CI practicality

- Uses modest default train steps and compact datasets.
- Caches Hugging Face and MLX downloads.
- Uses deterministic seeding for reproducibility.
- Includes a `--quick` mode suitable for PR checks.
