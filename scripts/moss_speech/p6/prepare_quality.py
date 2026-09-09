#!/usr/bin/env python3
"""Check complete streaming evidence and stage canonical cases for P5 scoring."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from scripts.moss_speech.p5.quality import build_cases, sha256, validate_pairs


def main() -> None:
    p = argparse.ArgumentParser()
    for name in (
        "manifest",
        "reference",
        "native-dir",
        "audio-reference",
        "out-dir",
        "voice-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    (a.out_dir / "reference_wav").mkdir()
    native = json.loads((a.native_dir / "report.json").read_text())
    reference = json.loads(a.reference.read_text())
    audio_ref = json.loads((a.audio_reference / "report.json").read_text())
    cases = build_cases(a.manifest)
    ids = {c["id"] for c in cases}
    canonical = [r for r in native["results"] if r["id"] in ids]
    receipt = dict(
        parent_native_sha256=sha256(a.native_dir / "report.json"),
        audio_reference_sha256=sha256(a.audio_reference / "report.json"),
        manifest_sha256=sha256(a.manifest),
        reference_sha256=sha256(a.reference),
        source_sha256=sha256(Path(__file__)),
        parent_pass=native["pass"],
        extra_request_count=len(native["results"]) - len(canonical),
        comparisons=[],
    )
    pairs = validate_pairs(reference["results"], canonical, sorted(ids))
    try:
        assert native["pass"] and audio_ref["pass"] and pairs["pass"]
        assert audio_ref["manifest_sha256"] == sha256(a.manifest)
        for case in cases:
            if not case["mode"].endswith("s"):
                continue
            rid = case["id"]
            actual = (a.native_dir / f"{rid}.pcm").read_bytes()
            expected = (a.audio_reference / f"{rid}.pcm").read_bytes()
            state = torch.load(
                a.native_dir / f"{rid}.pt", map_location="cpu", weights_only=False
            )
            frozen = torch.load(
                a.voice_dir / f"{rid}.pt", map_location="cpu", weights_only=False
            )
            same_voice = all(
                state[k] == frozen[k]
                for k in (
                    "voice_token_ids",
                    "voice_feat",
                    "voice_embedding",
                    "effective_seed",
                )
            )
            receipt["comparisons"].append(
                dict(
                    id=rid,
                    pcm_equal=actual == expected,
                    voice_equal=same_voice,
                    samples=len(actual) // 2,
                    native_sha256=sha256(a.native_dir / f"{rid}.pcm"),
                    reference_sha256=sha256(a.audio_reference / f"{rid}.pcm"),
                )
            )
            shutil.copy2(a.native_dir / f"{rid}.wav", a.out_dir / f"{rid}.wav")
            shutil.copy2(
                a.audio_reference / f"{rid}.wav",
                a.out_dir / "reference_wav" / f"{rid}.wav",
            )
        receipt["pass"] = all(
            r["pcm_equal"] and r["voice_equal"] for r in receipt["comparisons"]
        )
        report = {**receipt, "results": canonical, "pairs": pairs}
        (a.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        (a.out_dir / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(0 if receipt["pass"] else 1)


if __name__ == "__main__":
    main()
