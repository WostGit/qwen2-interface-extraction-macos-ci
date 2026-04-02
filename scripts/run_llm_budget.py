from experiments.llm_budget_sweep import run

if __name__ == "__main__":
    run(seed=0, out_csv="results/llm_budget_sweep.csv", model_id="Qwen/Qwen2-0.5B-Instruct")
