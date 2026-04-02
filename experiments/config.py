"""Experiment configuration constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CACHE_DIR = RESULTS_DIR / "cache"

BUDGETS = [64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2]

TOY_NUM_CLASSES = 5
TOY_FEATURE_DIM = 12
TOY_TRAIN_POOL = 400
TOY_EVAL_SIZE = 200

LLM_MODEL_NAME = "mlx-community/Qwen2-0.5B-Instruct-4bit"
LLM_VOCAB_TOPN = 32
LLM_TRAIN_PROMPTS = 128
LLM_EVAL_PROMPTS = 64
LLM_FIXED_BUDGET = 512

INTERFACES_DAY1 = ["argmax", "topk", "probs"]
TOPK_MAP = {
    "argmax": 1,
    "top2": 2,
    "top3": 3,
    "top5": 5,
    "topk": 3,
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
