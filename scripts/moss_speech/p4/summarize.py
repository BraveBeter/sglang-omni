#!/usr/bin/env python3
"""Aggregate completed P4 evidence without mutating raw experiment reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float]) -> dict[str, float]:
    return {
        name: float(np.percentile(values, percentile))
        for name, percentile in [("p50", 50), ("p95", 95), ("max", 100)]
    }


def summarize(args: Any) -> dict[str, Any]:
    reports = {}
    evidence = []
    for name in ("http", "workload", "cli", "soak"):
        path = Path(getattr(args, name)) / "report.json"
        data = json.loads(path.read_text())
        assert data["pass"] is True, f"{name} gate failed: {path}"
        assert data["checks"] and all(data["checks"].values())
        reports[name] = data
        evidence.append(dict(kind=name, path=str(path), sha256=sha(path)))
    for name in args.cpu:
        path = Path(name)
        text = path.read_text()
        assert re.search(r"\b\d+ passed\b", text), path
        assert not re.search(r"\b\d+ (?:failed|skipped|errors?)\b", text), path
        evidence.append(
            dict(
                kind="cpu",
                path=str(path),
                sha256=sha(path),
                passed=int(re.search(r"(\d+) passed", text).group(1)),
            )
        )
    workload = reports["workload"]
    warmup = [r for r in workload["rounds"] if r["kind"] == "warmup"]
    measured = [r for r in workload["rounds"] if r["kind"] == "measured"]
    assert len(warmup) == len(measured) == 3
    rows = [r for r in workload["requests"] if r["id"].startswith("r")]
    assert {r["id"] for r in rows} == {f"r{i}-{j}" for i in range(6) for j in range(24)}
    hashes = {}
    for r in rows:
        assert hashes.setdefault(r["request_hash"], r["grid_hash"]) == r["grid_hash"]
    sample = [r for r in rows if int(r["id"].split("-")[0][1:]) >= 3]
    elapsed = sum(r["seconds"] for r in measured)
    snapshots = workload["memory"]
    idle = [r["memory"]["used"] for r in measured]
    process_peaks = {}
    for snapshot in snapshots:
        for pid, value in snapshot["processes"].items():
            process_peaks[pid] = max(value, process_peaks.get(pid, 0))
    stage_by_pid = {}
    stage_events = {}
    for path in (Path(args.workload) / "events").glob("events_*.jsonl"):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            stage_by_pid[str(e["pid"])] = e["stage"]
            if e["request_id"] in {r["id"] for r in sample}:
                stage_events.setdefault((e["stage"], e["request_id"]), {})[
                    e["event_name"]
                ] = e["timestamp_ns"]
    stage_elapsed = {}
    for (stage, _), ev in stage_events.items():
        start = "scheduler_prefill_start" if stage == "ar_engine" else "stage_dispatch"
        end = "model_path_end" if stage == "ar_engine" else "stage_complete"
        if start in ev and end in ev:
            stage_elapsed.setdefault(stage, []).append((ev[end] - ev[start]) / 1e9)
    log = Path(args.workload).with_suffix(".log").read_text()
    native_batch = max(map(int, re.findall(r"#running-req: (\d+)", log)))
    assert native_batch == 4
    return dict(
        phase=4,
        pass_=True,
        gates={
            key: True
            for key in [
                "G1_http_four_modes",
                "G2_validation",
                "G3_lifecycle",
                "G4_native_load",
            ]
        },
        evidence=evidence,
        limits=dict(
            prompt_grid_rows=512,
            requested_new_grid_rows=512,
            context_length=1024,
            native_running=4,
            pipeline_waiting=20,
            kv_positions=4096,
            kv_bytes_per_position=163840,
            boundary_actual_generated_rows=reports["http"]["boundary_generated_rows"],
        ),
        metrics=dict(
            measured_requests=len(sample),
            warmup_requests=len(rows) - len(sample),
            measured_rounds=measured,
            concurrency_probes=[
                r for r in workload["rounds"] if r["kind"] == "concurrency"
            ],
            rps=len(sample) / elapsed,
            latency_seconds=distribution([r["seconds"] for r in sample]),
            generated_grid_rows=sum(r["completion_tokens"] for r in sample),
            max_prompt_grid_rows=max(r["prompt_tokens"] for r in sample),
            sampled_device_peak_gib=max(r["used"] for r in snapshots) / 2**30,
            measured_idle_spread_mib=(max(idle) - min(idle)) / 2**20,
            process_sampled_peak_gib={
                stage_by_pid.get(pid, pid): value / 2**30
                for pid, value in process_peaks.items()
            },
            stage_dispatch_to_complete_seconds={
                stage: distribution(values) for stage, values in stage_elapsed.items()
            },
            actual_native_batch_size=native_batch,
            waveform_soak_requests=len(reports["soak"]["requests"]),
        ),
        limitations=[
            "NVML sampled usage is not an instantaneous allocator peak.",
            "AR stage elapsed starts at prefill, excluding admission waiting; codec stage elapsed includes its queue.",
            "The native workload replaces P1 synthetic codes/background AR; throughput is not a like-for-like P1 speedup.",
            "Run 3865 checks long-run grid hashes; the separate soak checks serial/interleaved HTTP WAV hashes.",
            "Longer contexts, 24GB, streaming, graph/radix/quantization/TP>1 remain unqualified.",
            "Broad HTTP M4A regression has a reproduced pre-existing torchcodec/torch ABI failure.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("http", "workload", "cli", "soak"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cpu", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args)
    result["pass"] = result.pop("pass_")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
