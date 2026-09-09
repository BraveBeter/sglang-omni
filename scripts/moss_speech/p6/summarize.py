#!/usr/bin/env python3
"""Verify P6 artifacts and emit a compact report without corpus text or audio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.moss_speech.p5.quality import build_cases, sha256, validate_pairs


def distribution(values: list[float]) -> dict[str, float]:
    return {
        k: float(np.percentile(values, p))
        for k, p in (("p50", 50), ("p95", 95), ("max", 100))
    }


def main() -> None:
    p = argparse.ArgumentParser()
    for name in (
        "manifest",
        "reference",
        "native",
        "comparison",
        "evaluation",
        "offline",
        "baseline",
        "codec",
        "memory",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-amendments", type=Path)
    a = p.parse_args()
    receipt: dict[str, Any] = {"pass": False, "evidence_sha256": {}}

    def load(path: Path) -> dict:
        receipt["evidence_sha256"][str(path)] = sha256(path)
        r = json.loads(path.read_text())
        assert r["pass"] is True, f"Failed or incomplete report: {path}"
        return r

    ref, native, comp, ev, offline, baseline, codec = [
        load(getattr(a, name))
        for name in (
            "reference",
            "native",
            "comparison",
            "evaluation",
            "offline",
            "baseline",
            "codec",
        )
    ]
    repo = Path(__file__).resolve().parents[3]
    amendments = {}
    if a.source_amendments:
        amendment_receipt = load(a.source_amendments)
        amendments = amendment_receipt["changes"]
        assert set(amendments) == {"sglang_omni/models/moss_speech/__init__.py"}
        receipt["runtime_source_amendments"] = amendments
    runtime_sources = {}
    for relative, expected in native["source_sha256"].items():
        if relative.startswith(
            ("sglang_omni/models/moss_speech/", "sglang_omni/serve/")
        ) or relative in {
            "scripts/moss_speech/p6/validate_http.py",
            "scripts/moss_speech/p6/lifecycle.py",
            "scripts/moss_speech/p6/make_config.py",
        }:
            actual = sha256(repo / relative)
            if actual != expected:
                amendment = amendments.get(relative, {})
                assert amendment.get("before_sha256") == expected
                assert amendment.get("after_sha256") == actual
            runtime_sources["sglang-omni/" + relative] = expected
    receipt["runtime_source_sha256"] = runtime_sources
    receipt["postprocessor_sha256"] = sha256(Path(__file__))
    memory = load(a.memory)
    assert memory["expandable_segments_observed"] and all(memory["wave_equal"])
    assert memory["codes"] == 512
    cases = build_cases(a.manifest)
    ids = [c["id"] for c in cases]
    canonical = [r for r in native["results"] if r["id"] in ids]
    assert validate_pairs(ref["results"], canonical, ids)["pass"]
    assert validate_pairs(ref["results"], offline["results"], ids)["pass"]
    assert len(ev["results"]) == len(ids) == 32
    assert comp["parent_native_sha256"] == sha256(a.native)
    assert ev["native_sha256"] == sha256(a.comparison.parent / "report.json")
    assert (
        native["reference_sha256"] == offline["reference_sha256"] == sha256(a.reference)
    )
    assert (
        native["lifecycle_requested"]
        and native["lifecycle"]
        and all(native["lifecycle"].values())
    )
    assert native["processes_exit"] and offline["processes_exit"]
    assert native["release_events"]["disconnect-ar_done_pending_flush"]
    assert all(
        e["metadata"]["session_active"] is False
        for e in native["release_events"]["disconnect-ar_done_pending_flush"]
    )
    assert (
        ev["gate_thresholds"] is False
        and ev["calibrated"] is False
        and ev["human_mos"] is None
    )
    assert all(r["reference_native_error"]["errors"] == 0 for r in ev["results"])
    assert len(comp["comparisons"]) == 16 and all(
        r["pcm_equal"] and r["voice_equal"] for r in comp["comparisons"]
    )
    audio = [r for r in canonical if r["mode"].endswith("s")]
    timeline = [
        json.loads(line)
        for path in (a.native.parent / "events").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    early_http = {}
    for row in audio:
        submitted = min(
            e["timestamp_ns"]
            for e in timeline
            if e["request_id"] == row["id"]
            and e["stage"] == "preprocessing"
            and e["event_name"] == "stage_input_received"
        )
        ar_end = max(
            e["timestamp_ns"]
            for e in timeline
            if e["request_id"] == row["id"]
            and e["stage"] == "ar_engine"
            and e["event_name"] == "model_path_end"
        )
        # HTTP starts before backend admission. This is a conservative bound on
        # (AR end - actual first PCM receipt), without mixing absolute clocks.
        early_http[row["id"]] = (ar_end - submitted) / 1e9 - row["ttfa_s"]
    assert len(early_http) == 16 and min(early_http.values()) > 0
    wav_hashes, seams = {}, []
    for r in audio:
        rid = r["id"]
        # The unchanged offline path must still reproduce all P5 delivered WAVs.
        old, new = a.baseline.parent / f"{rid}.wav", a.offline.parent / f"{rid}.wav"
        assert sha256(old) == sha256(new)
        wav_hashes[rid] = sha256(new)
        raw = a.native.parent / f"{rid}.pcm"
        receipt["evidence_sha256"][str(raw)] = sha256(raw)
        pcm = np.frombuffer(raw.read_bytes(), dtype="<i2").astype(np.float32) / 32768
        cursor = 0
        for chunk in r["codec_ledger"][:-1]:
            cursor += chunk["samples"]
            seams.append(abs(float(pcm[cursor] - pcm[cursor - 1])))
    receipt.update(
        pass_=True,
        canonical_requests=len(canonical),
        total_completed_requests=len(native["results"]),
        full_grid_rows=sum(len(r["grid"]) for r in canonical),
        audio_pcm_equal=16,
        offline_wav_equal=len(wav_hashes),
        offline_wav_sha256=wav_hashes,
        profile="chunk-cudnn-deterministic-v1",
        http_chunk_size=5,
        component_profiles=[5, 25],
        lifecycle=native["lifecycle"],
        processes=native["processes"],
        gpu=native["gpu"],
        ttfa_s=distribution([r["ttfa_s"] for r in audio]),
        rtf=distribution([r["rtf"] for r in audio]),
        http_first_pcm_margin_lower_bound_s=early_http,
        first_pcm_before_ar_end=sum(
            bool(r.get("first_pcm_before_ar_end")) for r in audio
        ),
        codec_memory={
            "reserved_peak_gib": max(s["reserved"] for s in memory["samples"])
            / 1024**3,
            "live_peak_gib": max(s["peak_allocated"] for s in memory["samples"])
            / 1024**3,
            "release_mib": [
                s["allocated"] / 1024**2
                for s in memory["samples"]
                if s["label"].endswith("released")
            ],
            "allocator": memory["allocator"],
        },
        peak_gpu_gib=max(m["used"] for m in native["memory"]) / 1024**3,
        idle_gpu_mib=[m["used"] / 1024**2 for m in native["idle_memory"]],
        seam_absolute_jump=distribution(seams),
        seam_metric_is_quality_gate=False,
        quality=ev["summary"],
        gate_thresholds=False,
        calibrated=False,
        human_mos=None,
        limitations=[
            "A80080GB TP1 eager only",
            "No 24GB support",
            "No quality or latency SLA",
            "Two S2S cases reach the 512-row limit",
            "Weight/HF-code redistribution terms unresolved",
        ],
    )
    receipt["pass"] = receipt.pop("pass_")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
