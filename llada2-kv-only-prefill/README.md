# LLaDA2 KV-only prompt prefill evidence

Supporting material for [SGLang PR #51122](https://github.com/sgl-project/sglang/pull/51122).

## Versions and interpretation

The GPU-measured change is based on upstream `0f1509829b3920780aaea05bfc72b33db350a2e8`, with [#48647](https://github.com/sgl-project/sglang/pull/48647) as a prerequisite. The measured feature commit is `d49c028f0077ca2978a682a54dc9a28efd4543de` and its source tree is `622403fda69e38539aa09be04f304ee33dd5b625`.

The final PR head `9d8e719a35f67e0009021f14e3e71e6400b9c68e` additionally merges upstream `c2e9301c5bd5d8d6fe72898fbb20edede62ff020`; its tree is `d25542479e02c2356a2d0913179300b47641ad6f`. This resolves removal of adjacent deprecated environment variables and replaces the serving test's removed eager CLI alias with separate prefill/decode switches. The core KV-only implementation files are byte-identical to the measured head. On the merged source, 25 unit tests, real ServerArgs eligibility, new CLI parsing, and pre-commit passed; GitHub Lint passed as well. See `mainline_sync_validation.json`. The CPU/CLI checks reused the original isolated environment; GPU regression and timing were not rerun after this sync because the GPU was occupied. Upstream CI still needs to provide fresh dependency/GPU coverage.

`legacy_*` files describe a **SGLang 0.5.17 prototype**, Torch 2.11.0+cu128 and FlashInfer 0.6.15.post1. They are not timings for the submitted mainline patch. This earlier experiment used one A-B-C traversal, without independent repetitions. Its very small full-dataset end-to-end differences should not be treated as a stable speedup.

Both experiments use LLaDA2.0-mini on one RTX 6000 Ada 48 GB, BF16, LowConfidence/FDFO with 32-token blocks, FlashInfer attention, and Triton MoE. Model/config/tokenizer fingerprints are in `model_fingerprints.json`. Comparisons are within each software/hardware environment; do not compare absolute times across the two environments.

## Mainline results

**Mainline controlled rerun completed.** The three arms are A: unmodified upstream `0f150982`; B: this feature head with the opt-in disabled, retaining #48647's full-forward behavior; and C: the same head with the opt-in enabled. B therefore includes the disabled-path guards; B/C is a feature-on/off ablation.

The run uses natural lengths 1024–1053 and 4096–4127, batch 1/4, 16 full source prompts per bin (four each from ShareGPT, WildChat, OpenHermes, and OpenOrca), and two fresh-process orders A-B-C / C-B-A. Each table entry averages 16 prompts × 2 repetitions. All 384 requests completed; 128 three-arm pairs and the cross-round comparisons have identical output token-ID hashes (42,674 output token positions per arm). No external compute PID was sampled during any arm; the maximum sampling gap was 1.152 seconds.

First-text latency:

| Input bin | Batch | A mean (ms) | B mean (ms) | C mean (ms) | C vs B reduction |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 714.64 | 643.87 | 597.73 | 7.17% |
| 1024 | 4 | 1315.51 | 1190.88 | 1117.80 | 6.14% |
| 4096 | 1 | 2348.69 | 1956.97 | 1776.15 | 9.24% |
| 4096 | 4 | 4770.19 | 3832.14 | 3550.38 | 7.35% |

End-to-end latency:

| Input bin | Batch | A mean (ms) | B mean (ms) | C mean (ms) | C vs B reduction |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 4366.16 | 4309.54 | 4262.28 | 1.10% |
| 1024 | 4 | 7755.02 | 7635.94 | 7568.40 | 0.88% |
| 4096 | 1 | 5952.06 | 5560.60 | 5398.81 | 2.91% |
| 4096 | 4 | 9874.29 | 8938.40 | 8673.91 | 2.96% |

At 4096, the two individual runs reduced first-text time by 9.17–9.31% (batch 1) and 7.34–7.37% (batch 4); end-to-end reductions were 2.88–2.94% and 2.92–3.00%. This is a small, two-repetition single-device study, not a statistical-significance claim. 44 of 128 requests per arm reached the 512-token cap. The three full classic datasets below were not rerun on this mainline head.

[Measurements, source IDs/hashes, exact configuration, and reproduction scripts](https://github.com/cyb4188/sglang/tree/bench/llada2-kv-only-prefill/llada2-kv-only-prefill).

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

The script uses Transformers and PyArrow, verifies source hashes, applies the model's chat template to each complete selected conversation, and checks the actual token count and token-ID hash. ShareGPT uses conversation IDs (the last source record wins for duplicate IDs, matching the original verification); OpenHermes and WildChat use zero-based source row indices; OpenOrca uses its `id` column. `/turnN` includes all conversation entries through index N. It streams the large JSON source arrays.

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

Run against separate servers with the opt-in set to 0 (B) and 1 (C). B and C use the same feature head: B retains #48647's full-forward behavior with the new early exits disabled. Thus this is a feature-on/off ablation; B includes the disabled-path guard overhead. A uses the unmodified base commit. Keep all other server arguments identical and keep the GPU free of unrelated workloads.

`run_mainline.py` runs the controlled two-round comparison. Its orders are A-B-C and C-B-A, each in fresh server processes. For every input-bin/batch-size cohort it runs two warmup batches with the same 512-token cap before measuring the 16 selected requests. This differs from the standalone client's shorter 32-token warmup. Monitoring samples GPU process ownership about once per second throughout each arm and rejects an arm if any external compute PID is observed.

Prepare a Python 3.12 environment with the feature checkout's pinned dependencies. The recorded environment disables the optional Rust extensions (`SGLANG_BUILD_RUST_EXTS=none`). From a checkout containing both commits:

```bash
export BENCH_WORKSPACE=/path/to/new/benchmark-workspace
export MODEL_PATH=/path/to/local/LLaDA2.0-mini
export CUDA_HOME=/usr/local/cuda-13.1
mkdir -p "$BENCH_WORKSPACE/tmp"
git worktree add --detach "$BENCH_WORKSPACE/source_a" 0f1509829b3920780aaea05bfc72b33db350a2e8
git worktree add --detach "$BENCH_WORKSPACE/source_lf" d49c028f0077ca2978a682a54dc9a28efd4543de
python reconstruct_matrix.py --source-dir SOURCE_DIR --model "$MODEL_PATH" \
  --output "$BENCH_WORKSPACE/matrix_manifest.jsonl"
python run_mainline.py
python summarize_mainline.py
```

The driver sets `PYTHONPATH` to the appropriate checkout for each server and writes individual measurements and launch arguments below `BENCH_WORKSPACE/mainline_results`. It uses a local checkpoint in offline mode, seed 42, context 8192, total KV token budget 24576, maximum running requests 4, chunked/max prefill 512, static memory fraction 0.8, and decode graph batch sizes 1/2/3/4. The complete server argument list is in the driver. This configuration differs from the legacy prototype; compare arms within one environment only.
