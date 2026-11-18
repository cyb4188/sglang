"""Audit and summarize the completed controlled mainline comparison."""

import os
import collections
import json
import statistics
from pathlib import Path

ROOT = Path(os.environ["BENCH_WORKSPACE"]).resolve()
OUT = ROOT / "mainline_results"
if not (OUT / "completed.json").exists():
    raise SystemExit("The two-round benchmark is still running")
rows = []
for path in sorted(OUT.glob("r[12]_[ABC]_n*_b*.jsonl")):
    part = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(part) == 16, (path, len(part))
    rows.extend(part)
assert len(rows) == 384, len(rows)
source_rows = [
    json.loads(line)
    for line in (ROOT / "matrix_manifest.jsonl").read_text().splitlines()
]
source = {(r["bin"], r["dataset"], r["uid"]): r for r in source_rows}
paired = collections.defaultdict(dict)
for row in rows:
    expected = source[row["nominal_tokens"], row["dataset"], row["uid"]]
    assert row["input_sha256"] == expected["input_sha"]
    assert row["input_tokens"] == expected["input_len"]
    assert row["first_text_ms"] is None or 0 <= row["first_text_ms"] <= row["e2e_ms"]
    key = (
        row["repeat"],
        row["nominal_tokens"],
        row["batch_size"],
        row["dataset"],
        row["uid"],
    )
    assert row["arm"] not in paired[key], key
    paired[key][row["arm"]] = row
mismatches = []
for key, arms in paired.items():
    assert set(arms) == set("ABC"), key
    for field in ("input_sha256", "output_sha256", "output_tokens", "finish_reason"):
        if not arms["A"][field] == arms["B"][field] == arms["C"][field]:
            mismatches.append({"key": key, "field": field})
cohorts = []
for length in (1024, 4096):
    for batch_size in (1, 4):
        stats = {}
        for arm in "ABC":
            values = [
                r
                for r in rows
                if r["nominal_tokens"] == length
                and r["batch_size"] == batch_size
                and r["arm"] == arm
            ]
            text = [
                r["first_text_ms"] for r in values if r["first_text_ms"] is not None
            ]
            stats[arm] = {
                "requests": len(values),
                "first_text_requests": len(text),
                "mean_first_text_ms": statistics.mean(text),
                "mean_e2e_ms": statistics.mean(r["e2e_ms"] for r in values),
                "capped": sum(r["finish_reason"]["type"] == "length" for r in values),
            }
        rounds = {}
        for repeat in (1, 2):
            b = [
                r
                for r in rows
                if r["repeat"] == repeat
                and r["nominal_tokens"] == length
                and r["batch_size"] == batch_size
                and r["arm"] == "B"
            ]
            c = [
                r
                for r in rows
                if r["repeat"] == repeat
                and r["nominal_tokens"] == length
                and r["batch_size"] == batch_size
                and r["arm"] == "C"
            ]
            rounds[str(repeat)] = {
                metric + "_reduction_percent": 100
                * (
                    1
                    - statistics.mean(r[metric] for r in c if r[metric] is not None)
                    / statistics.mean(r[metric] for r in b if r[metric] is not None)
                )
                for metric in ("first_text_ms", "e2e_ms")
            }
        cohorts.append(
            {
                "nominal_tokens": length,
                "batch_size": batch_size,
                "stats": stats,
                "c_vs_b_percent": {
                    metric: 100 * (1 - stats["C"][metric] / stats["B"][metric])
                    for metric in ("mean_first_text_ms", "mean_e2e_ms")
                },
                "per_round_c_vs_b": rounds,
            }
        )
monitor = {}
for path in sorted(OUT.glob("r[12]_[ABC]_gpu.jsonl")):
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    monitor[path.stem] = {
        "samples": len(samples),
        "external_pid_samples": sum(bool(r["foreign"]) for r in samples),
        "max_sample_gap_seconds": max(
            b["monotonic"] - a["monotonic"] for a, b in zip(samples, samples[1:])
        ),
    }
    assert not monitor[path.stem]["external_pid_samples"], path
assert len(monitor) == 6
across_rounds = collections.defaultdict(set)
for row in rows:
    key = (
        row["nominal_tokens"],
        row["batch_size"],
        row["dataset"],
        row["uid"],
        row["arm"],
    )
    across_rounds[key].add(row["output_sha256"])
summary = {
    "requests": len(rows),
    "paired_triplets": len(paired),
    "mismatches": mismatches,
    "cross_round_output_mismatches": sum(
        len(values) != 1 for values in across_rounds.values()
    ),
    "output_token_positions_per_arm": {
        arm: sum(r["output_tokens"] for r in rows if r["arm"] == arm) for arm in "ABC"
    },
    "cohorts": cohorts,
    "gpu_monitor": monitor,
    "orders": ["ABC", "CBA"],
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
print(json.dumps(summary, indent=2))
