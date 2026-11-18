"""Task-local three-arm validation on the approved isolated environment."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(os.environ["BENCH_WORKSPACE"]).resolve()
OUT = ROOT / "mainline_results"
OUT.mkdir(exist_ok=True)
URL = "http://127.0.0.1:31373"
spec = importlib.util.spec_from_file_location(
    "bench", ROOT / "source_lf/benchmark/dllm/bench_kv_only_prefill.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
records = [json.loads(s) for s in (ROOT / "matrix_manifest.jsonl").read_text().splitlines()]


def gpu_pids():
    value = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {int(s.strip()) for s in value.splitlines() if s.strip().isdigit()}


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def run_arm(repeat, arm):
    while gpu_pids():
        print("Waiting for an idle GPU", flush=True)
        time.sleep(600)
    source = ROOT / ("source_a" if arm == "A" else "source_lf")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(source / "python"),
        SGLANG_ENABLE_DLLM_KV_ONLY_PREFILL=str(int(arm == "C")),
        CUDA_VISIBLE_DEVICES="0",
        CUDA_HOME=os.environ["CUDA_HOME"],
        HF_HUB_OFFLINE="1",
        TMPDIR=str(ROOT / "tmp"),
        FLASHINFER_WORKSPACE_BASE=str(ROOT / "flashinfer"),
        TRITON_CACHE_DIR=str(ROOT / "triton"),
        TVM_FFI_CACHE_DIR=str(ROOT / "tvm_ffi"),
    )
    args = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", os.environ["MODEL_PATH"],
        "--host", "127.0.0.1", "--port", "31373",
        "--trust-remote-code", "--dtype", "bfloat16", "--tp-size", "1",
        "--mem-fraction-static", "0.8", "--context-length", "8192",
        "--max-total-tokens", "24576", "--max-running-requests", "4",
        "--chunked-prefill-size", "512", "--max-prefill-tokens", "512",
        "--attention-backend", "flashinfer", "--moe-runner-backend", "triton",
        "--dllm-algorithm", "LowConfidence", "--dllm-fdfo",
        "--cuda-graph-bs-decode", "1", "2", "3", "4",
        "--random-seed", "42", "--log-level", "warning",
    ]
    (OUT / f"r{repeat}_{arm}_command.json").write_text(json.dumps(args, indent=2))
    with (OUT / f"r{repeat}_{arm}_server.log").open("w") as log:
        process = subprocess.Popen(args, cwd=source, env=env, stdout=log, stderr=log,
                                   start_new_session=True)
        stop = threading.Event()
        contaminated = threading.Event()
        guard_error = []

        def sample():
            with (OUT / f"r{repeat}_{arm}_gpu.jsonl").open("w") as output:
                while not stop.is_set():
                    try:
                        own = {process.pid} | {
                            p.pid for p in psutil.Process(process.pid).children(recursive=True)
                        }
                        pids = gpu_pids()
                        foreign = pids - own
                        output.write(json.dumps({"monotonic": time.monotonic(),
                                                 "gpu_pids": sorted(pids),
                                                 "foreign": sorted(foreign)}) + "\n")
                        output.flush()
                        if foreign:
                            contaminated.set()
                    except Exception as error:
                        guard_error.append(repr(error))
                        contaminated.set()
                        return
                    stop.wait(1)

        thread = threading.Thread(target=sample)
        thread.start()
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{arm} server exited {process.returncode}")
                try:
                    if requests.get(URL + "/health_generate", timeout=10).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Server startup timeout")
            with requests.Session() as session:
                for length in (1024, 4096):
                    selected = [r for r in records if r["bin"] == length]
                    assert len(selected) == 16
                    for batch_size in (1, 4):
                        path = OUT / f"r{repeat}_{arm}_n{length}_b{batch_size}.jsonl"
                        for _ in range(2):
                            bench.measure(session, URL, selected[:batch_size], 512, 360)
                        with path.open("w") as output:
                            for offset in range(0, len(selected), batch_size):
                                if contaminated.is_set():
                                    raise RuntimeError(f"GPU sample invalid: {guard_error}")
                                rows = bench.measure(session, URL,
                                                     selected[offset:offset + batch_size], 512, 360)
                                for row in rows:
                                    row.update(arm=arm, repeat=repeat, batch_size=batch_size,
                                               nominal_tokens=length)
                                    output.write(json.dumps(row) + "\n")
                                output.flush()
                        print(f"Completed repeat={repeat} arm={arm} tokens={length} batch={batch_size}", flush=True)
                time.sleep(1.2)
                if contaminated.is_set():
                    raise RuntimeError(f"GPU sample invalid: {guard_error}")
        finally:
            stop.set()
            thread.join(timeout=10)
            stop_owned(process)
    print(f"Finished repeat={repeat} arm={arm}", flush=True)


for repeat, order in enumerate(("ABC", "CBA"), 1):
    for arm in order:
        run_arm(repeat, arm)
(OUT / "completed.json").write_text(json.dumps({"ok": True, "orders": ["ABC", "CBA"]}))
print("ALL THREE-ARM MAINLINE MEASUREMENTS COMPLETE", flush=True)
