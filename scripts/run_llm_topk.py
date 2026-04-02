from experiments.llm_topk_sweep import run

if __name__ == "__main__":
    run(
        seeds=[0, 1, 2],
        budget=256,
        out_csv="results/llm_topk_sweep.csv",
        out_summary="results/llm_topk_sweep_summary.csv",
        model_id="Qwen/Qwen2-0.5B-Instruct",
    )
