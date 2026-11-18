# LLaDA2 KV-only prompt prefill evidence

Supporting material for [SGLang PR #51122](https://github.com/sgl-project/sglang/pull/51122).

## Versions and interpretation

The submitted change is based on upstream `0f1509829b3920780aaea05bfc72b33db350a2e8`, with [#48647](https://github.com/sgl-project/sglang/pull/48647) as a prerequisite. The feature commit is `d49c028f0077ca2978a682a54dc9a28efd4543de` and its source tree is `622403fda69e38539aa09be04f304ee33dd5b625`.

`legacy_*` files describe a **SGLang 0.5.17 prototype**, Torch 2.11.0+cu128 and FlashInfer 0.6.15.post1. They are not timings for the submitted mainline patch. This earlier experiment used one A-B-C traversal, without independent repetitions. Its very small full-dataset end-to-end differences should not be treated as a stable speedup.

Both experiments use LLaDA2.0-mini on one RTX 6000 Ada 48 GB, BF16, LowConfidence/FDFO with 32-token blocks, FlashInfer attention, and Triton MoE. Model/config/tokenizer fingerprints are in `model_fingerprints.json`. Comparisons are within each software/hardware environment; do not compare absolute times across the two environments.

## Legacy full experiment

`legacy_latency.csv` contains all three arms' aggregate first-text and end-to-end latencies, in milliseconds. A is the original 0.5.17 implementation, B backports #48647, and C additionally skips the final FFN/MoE, output norm, and LM head, with a separate CUDA Graph variant. Its server parameters are in `legacy_server_config.json`.

- Matrix: nominal token counts 32/64/128/256/512/1024/2048/4096, batch 1/4. Each bin has 16 full prompts, four per source dataset, with actual length in `[N, N + 31]`.
- Full sets: GSM8K test 1,319; HumanEval 164; MBPP official test IDs 11–510, 500. Each is measured at batch 1 and 4 with all three arms.
- 12,666 valid logical requests, 4,222 paired triples, all corresponding output token IDs equal. There are 1,353,551 compared token positions per arm.
- Every accepted measurement is bracketed by GPU process-ownership samples, including samples during execution. No external compute PID was observed. Four preliminary measurements without complete monitoring were excluded and repeated; see `legacy_audit.json`.
- First-text reduction against B: 7.51%/6.08% at 1024, and 9.56%/7.46% at 4096 (batch 1/4). The corresponding end-to-end reductions are 1.47%/1.12% and 3.59%/3.09%.
- Full-dataset first-text reductions are 1.61–3.62%; end-to-end reductions are 0.02–0.22%. About 6–7% of GSM8K, 45% of HumanEval, and 47–48% of MBPP hit the 512-token cap. This is generation-latency and output-equivalence evidence, not task accuracy or pass@1.

## Inputs and reconstruction

`matrix_source_manifest.jsonl` preserves the selected public dataset IDs/row indices, inclusive conversation turn, natural lengths, order, and token-ID hashes. It contains no prompt bodies or generated text. All 128 selections were checked against the original source records.

Download the four pinned files listed in `source_files.json` into `SOURCE_DIR`, retaining the indicated local filenames. The URLs and complete source-file SHA256 values identify the exact inputs. Use `reconstruct_matrix.py` with the checkpoint/tokenizer identified in `model_fingerprints.json`:

```bash
python reconstruct_matrix.py --source-dir SOURCE_DIR \
  --model MODEL_PATH --output matrix_inputs.jsonl
```

The script uses Transformers and PyArrow, verifies source hashes, applies the model's chat template to each complete selected conversation, and checks the actual token count and token-ID hash. ShareGPT uses conversation IDs; OpenHermes and WildChat use zero-based source row indices; OpenOrca uses its `id` column. `/turnN` includes all conversation entries through index N. The script reads the JSON sources in memory.

For the full sets, the prototype used each GSM8K `question` as one user message, each HumanEval `prompt` as one user message, and the official MBPP test split with this user prompt:

```python
text = (f"You are an expert Python programmer, and here is your task: {row['text']} "
        "Your code should pass these tests:\n\n"
        + "\n".join(row["test_list"]) + "\n[BEGIN]\n")
```

All use `apply_chat_template(..., tokenize=True, add_generation_prompt=True)`. The legacy client shuffled each cohort with seed 20261117 before batching. Mainline timings use the manifest order directly; arm-to-arm order is identical within each experiment.

## Measurement definition

Greedy generation, EOS enabled, maximum 512 new tokens. Flush the prefix cache before every batch. Batch 4 is one HTTP request containing four logical requests, timed individually. First-text time begins immediately before the POST and ends at the first SSE response containing non-whitespace text. End-to-end time ends at the logical request's `finish_reason` response.

The submitted streaming client is `benchmark/dllm/bench_kv_only_prefill.py`:

```bash
python benchmark/dllm/bench_kv_only_prefill.py \
  --base-url http://127.0.0.1:30000 --input-jsonl matrix_inputs.jsonl \
  --input-length 4096 --batch-size 4 --max-new-tokens 512 \
  --output results.jsonl
```

Run against separate servers with the opt-in set to 0 (B) and 1 (C). A uses the unmodified base commit. Keep all other server arguments identical and keep the GPU free of unrelated workloads. The mainline report and exact launch arguments will be added after the two ordered repetitions finish.
