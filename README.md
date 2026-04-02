# qwen2-interface-extraction-macos-ci

Day-1 research scaffold for black-box extraction experiments on macOS GitHub Actions runners with MLX.

## What it runs

Two experiment families:

1. **Toy classifier baseline** (synthetic probabilistic victim).
2. **Tiny LLM next-token imitation** using **Qwen2-0.5B** via MLX.

Three primary victim interfaces for the budget sweep:
- `argmax`
- `topk` (k=5)
- `probs`

Day-1 outputs:
- budget sweep over query counts: `64, 128, 256, 512, 1024`
- per-run metrics for each `source/interface/budget/seed`:
  - `agreement`
  - `kl_divergence`
- plots:
  - `agreement_vs_budget.png`
  - `kl_vs_budget.png`
- fixed-budget LLM top-k sweep:
  - `argmax`, `top2`, `top3`, `top5`, `probs`
  - includes multi-seed aggregate mean/std CSV

## Repo layout

- `experiments/` — experiment family logic
- `models/` — interface simulation and shared metrics
- `attacks/` — attack package placeholder for Day-2+
- `scripts/` — small scripts per job
- `results/` — CSV/PNG outputs
- `.github/workflows/` — macOS CI workflow
- `run_all.py` — single Day-1 entrypoint

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

## Determinism and runtime

- Seeds are fixed in scripts.
- Query budgets capped to keep CI practical.
- Workflow caches pip packages and MLX model downloads.

