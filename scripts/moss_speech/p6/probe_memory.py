#!/usr/bin/env python3
"""Observe live allocations versus the CUDA allocator cache on a long stream."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("codec-path", "state", "out-dir", "reference"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    state = torch.load(a.state, map_location="cpu", weights_only=False)
    voice = {
        "prompt_token": torch.tensor([state["voice_token_ids"]], dtype=torch.int32),
        "prompt_feat": torch.tensor([state["voice_feat"]]),
        "embedding": torch.tensor([state["voice_embedding"]]),
    }
    codes = []
    for text, code in state["output_grid"]:
        if text == 151667:
            if code == 16384:
                break
            codes.append(code)
    codec = MossSpeechStreamingCodec(a.codec_path)
    reference = torch.load(a.reference, map_location="cpu", weights_only=False)
    assert reference["codes"] == codes
    report = {
        "pass": False,
        "samples": [],
        "codes": len(codes),
        "wave_equal": [],
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "reserved_limit_gib": 8,
    }

    def observe(label: str) -> None:
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        report["samples"].append(
            dict(
                label=label,
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                inactive_split=stats["inactive_split_bytes.all.current"],
                allocation_retries=stats["num_alloc_retries"],
                oom=stats["num_ooms"],
                sessions=len(codec.sessions),
            )
        )
        (a.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    observe("loaded")
    codec.begin("warmup", voice, seed=0)
    codec.push("warmup", codes[:1], final=True)
    observe("warmed")
    for repeat in range(2):
        waves = []
        codec.begin(str(repeat), voice, seed=0)
        for i, code in enumerate(codes):
            waves.extend(codec.push(str(repeat), [code]))
            if (i + 1) % 25 == 0:
                observe(f"{repeat}:code-{i + 1}")
        waves.extend(codec.push(str(repeat), [], final=True))
        report["wave_equal"].append(
            len(waves) == len(reference["waves"])
            and all(torch.equal(x, y) for x, y in zip(waves, reference["waves"]))
        )
        observe(f"{repeat}:released")
    codec.close()
    observe("closed")
    loaded = next(s["allocated"] for s in report["samples"] if s["label"] == "warmed")
    released = [s for s in report["samples"] if s["label"].endswith("released")]
    report["pass"] = all(
        s["sessions"] == 0 and abs(s["allocated"] - loaded) < 1024**2 for s in released
    )
    report["expandable_segments_observed"] = any(
        segment.get("is_expandable", False)
        for segment in torch.cuda.memory._snapshot()["segments"]
    )
    report["pass"] = (
        report["expandable_segments_observed"]
        and report["pass"]
        and all(report["wave_equal"])
        and all(
            s["reserved"] <= 8 * 1024**3
            and s["allocation_retries"] == 0
            and s["oom"] == 0
            for s in report["samples"]
        )
    )
    (a.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
