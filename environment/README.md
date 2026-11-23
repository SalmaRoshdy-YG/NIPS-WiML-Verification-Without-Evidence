# Environment (as used for the corrected runs; Trillium, Alliance Canada)

Modules: `StdEnv/2023 gcc/12.3 python/3.11 arrow/25.0.0 cuda/12.2 opencv`, then a Python 3.11 venv.

| Package | Version |
|---|---|
| vllm | 0.25.0 |
| torch | 2.11.0 |
| transformers | 5.17.0 |
| datasets | 5.0.0 |
| tokenizers | 0.23.2 |
| numpy | 2.3.3 |

The v1 abstract's runs used transformers 5.14.1 on another cluster, so expect agreement within
sampling noise, not identical digits. These versions come from `pip list` on the venv that produced
the results in this repo; I did not rebuild a venv from this file.

Hardware: one 80 GB H100 per shard. Peak host memory measured at 7 GB.

Models (local copies under `$STUDY/hf/models/`): Qwen3-8B and Qwen3-14B, bf16.
Data: CodeJudgeBench, Hugging Face `mattymchen/codejudgebench`, config `codegen`, loaded with
`datasets.load_dataset` (offline copies; set `HF_HUB_OFFLINE=1`). HumanEvalFix pairs: `data/hef_pairs.jsonl`.

Sampling (CodeJudgeBench's Qwen3 settings, verbatim): temperature 0.6, top_p 0.95, top_k 20, min_p 0,
max_tokens 32768, max_model_len 40960, seed 1234. MARCH's checker takes 3 samples and a majority vote.
