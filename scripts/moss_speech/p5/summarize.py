#!/usr/bin/env python3
"""Verify completed P5 evidence and export a compact, text-free delivery record."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from quality import MODES, build_cases, sha256, validate_pairs


def distribution(values: list[float]) -> dict[str, float]:
    return {
        key: float(np.percentile(values, percentile))
        for key, percentile in (("p50", 50), ("p95", 95), ("max", 100))
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    evidence = {}

    def record(path: Path) -> None:
        evidence[str(path)] = sha256(path)

    cases = build_cases(args.manifest)
    ids = [c["id"] for c in cases]
    assert len(ids) == 32 and len(set(ids)) == 32
    record(args.manifest)
    reports = {}
    for name in ("reference", "native", "evaluation"):
        path = getattr(args, name)
        record(path)
        data = json.loads(path.read_text())
        assert data["pass"] is True, f"Incomplete or failed {name}"
        assert data["manifest_sha256"] == sha256(args.manifest)
        rows = data["results"]
        assert len(rows) == len(ids) and {r["id"] for r in rows} == set(ids)
        assert all(not r.get("error") for r in rows)
        reports[name] = data
    ref, nat, ev = (reports[n] for n in ("reference", "native", "evaluation"))
    assert ref["precision"]["logits_to_keep"] == 0
    assert nat["reference_sha256"] == ev["reference_sha256"] == sha256(args.reference)
    assert ev["native_sha256"] == sha256(args.native)
    assert validate_pairs(ref["results"], nat["results"], ids)["pass"]
    assert nat["futures_empty"] and nat["processes_exit"]
    assert len(nat["processes"]) == 4 and all(
        not p["alive"] and p["exitcode"] == 0 for p in nat["processes"]
    )
    assert nat["quality_gate_thresholds"] is False
    assert nat["quality_calibrated"] is False
    assert ev["gate_thresholds"] is False and ev["calibrated"] is False
    assert ev["human_mos"] is None
    record(args.config)
    assert nat["config_sha256"] == sha256(args.config)
    assert nat["source_sha256"]
    for name, digest in nat["source_sha256"].items():
        assert sha256(Path(name)) == digest, f"Source changed after GPU CI: {name}"
    acoustic_rows = [r for r in ev["results"] if r["mode"].endswith("s")]
    assert len(acoustic_rows) == 16
    wav_hashes = {}
    for row in ev["results"]:
        assert row["reference_native_error"]["errors"] == 0
        if row not in acoustic_rows:
            continue
        for name in ("reference", "native"):
            directory = args.native.parent
            if name == "reference":
                directory /= "reference_wav"
            wav = directory / f"{row['id']}.wav"
            acoustic = row[name]["acoustics"]
            assert acoustic["valid"] and acoustic["samples"] > 0
            assert acoustic["sample_rate"] == 24000
            assert acoustic["sha256"] == sha256(wav)
            wav_hashes[str(wav)] = sha256(wav)
        assert row["reference"]["acoustics"] == row["native"]["acoustics"]
    for mode in MODES:
        for lang in ("en", "zh"):
            rows = [r for r in ev["results"] if r["mode"] == mode and r["lang"] == lang]
            summary = ev["summary"][f"{mode}_{lang}"]
            assert len(rows) == summary["total"] == 4 and summary["failed"] == 0
            for name in ("reference", "native"):
                counted = [r[name]["annotation_error"] for r in rows if mode != "s2s"]
                if counted:
                    assert sum(x["errors"] for x in counted) == summary[name]["errors"]
                    assert (
                        sum(x["reference_units"] for x in counted)
                        == summary[name]["reference_units"]
                    )
                assert summary[name]["truncated"] == sum(
                    r[name]["finish_reason"] == "length" for r in rows
                )
    cpu = {}
    for path in args.cpu:
        text = path.read_text()
        match = re.search(r"\b(\d+) passed\b", text)
        assert match and not re.search(
            r"\b\d+ (?:failed|skipped|errors?)\b", text
        ), path
        cpu[str(path)] = int(match.group(1))
        record(path)
    record(args.hardware)
    hardware = json.loads(args.hardware.read_text())
    assert hardware["pass"] is False and "4090" in hardware["gpu"]
    assert len(hardware["results"]) == 32 and hardware["processes_exit"]
    failed = [r["id"] for r in hardware["results"] if r.get("error")]
    assert len(failed) == 7
    record(args.hardware_log)
    assert "CUDA out of memory" in args.hardware_log.read_text()
    for path in args.evidence:
        record(path)
    return {
        "phase": 5,
        "pass": True,
        "gates": {
            "G1_quality_evidence": True,
            "G2_hardware_outcome": True,
            "G3_local_ci": True,
            "G4_documentation_and_provenance": True,
        },
        "upstream_redistribution_ready": False,
        "hosted_ci_executed": False,
        "quality_gate_thresholds": False,
        "quality_calibrated": False,
        "human_mos": None,
        "sample": {
            "manifest_sha256": sha256(args.manifest),
            "source_rows": len(cases) // 4,
            "unique_source_audios": len(
                {
                    s["source_sha256"]
                    for s in json.loads(args.manifest.read_text())["samples"]
                }
            ),
            "mode_requests": len(cases),
            "unique_request_bodies": len(
                {json.dumps(c["body"], sort_keys=True) for c in cases}
            ),
        },
        "numerical_profile": ref["precision"],
        "native": {
            "gpu": nat["gpu"],
            "gpu_uuid": nat["gpu_uuid"],
            "equal_full_grid_rows": sum(len(r["grid"]) for r in nat["results"]),
            "bit_equal_wav_pairs": len(acoustic_rows),
            "truncated_ids": [
                r["id"] for r in nat["results"] if r["finish_reason"] == "length"
            ],
            "processes": nat["processes"],
            "sampled_peak_gib": max(m["used"] for m in nat["memory"]) / 2**30,
            "http_latency_s": {
                mode: distribution(
                    [r["seconds"] for r in nat["results"] if r["mode"] == mode]
                )
                for mode in MODES
            },
            "http_rtf": {
                mode: distribution(
                    [r["native"]["rtf"] for r in acoustic_rows if r["mode"] == mode]
                )
                for mode in ("t2s", "s2s")
            },
            "audio_duration_s": distribution(
                [r["native"]["acoustics"]["duration_s"] for r in acoustic_rows]
            ),
            "max_clipping_fraction": max(
                r["native"]["acoustics"]["clipping_fraction"] for r in acoustic_rows
            ),
        },
        "quality": ev["summary"],
        "asr_checkpoint_sha256": ev["asr_sha256"],
        "asr_unique_waveform_language_pairs": ev["asr_unique_waveform_language_pairs"],
        "cpu_passed_by_log": cpu,
        "hardware_24gb": {
            "supported": False,
            "gpu": hardware["gpu"],
            "gpu_uuid": hardware["gpu_uuid"],
            "http_success": 32 - len(failed),
            "http_failures": failed,
            "sampled_peak_gib": max(m["used"] for m in hardware["memory"]) / 2**30,
            "failure": "FP32 vocoder CUDA OOM; see raw process traceback",
            "processes": hardware["processes"],
        },
        "source_sha256_at_native_start": nat["source_sha256"],
        "evidence_sha256": evidence,
        "waveform_sha256": wav_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "reference",
        "native",
        "evaluation",
        "config",
        "hardware",
        "hardware-log",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu", type=Path, nargs="+", required=True)
    parser.add_argument("--evidence", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    result = summarize(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
