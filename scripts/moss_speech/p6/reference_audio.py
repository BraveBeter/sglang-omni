#!/usr/bin/env python3
"""Export independent streaming audio for every frozen P5 audio case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.moss_speech.p5.quality import build_cases, sha256
from scripts.moss_speech.p6.reference_codec import decode_reference, make_reference


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("manifest", "reference", "voice-dir", "codec-path", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    source = json.loads(a.reference.read_text())
    assert source["pass"] and source["manifest_sha256"] == sha256(a.manifest)
    rows = {r["id"]: r for r in source["results"]}
    report = {
        "pass": False,
        "profile": "chunk-cudnn-deterministic-v1",
        "chunk_size": 5,
        "reference_sha256": sha256(a.reference),
        "manifest_sha256": sha256(a.manifest),
        "source_sha256": sha256(Path(__file__)),
        "results": [],
    }
    try:
        ref = make_reference(a.codec_path, 5, a.out_dir)
        report["weight_sha256"] = sha256(a.codec_path / "flow/flow-chunk-5.pt")
        for case in build_cases(a.manifest):
            if not case["mode"].endswith("s"):
                continue
            rid = case["id"]
            voice_path = a.voice_dir / f"{rid}.pt"
            voice = torch.load(voice_path, map_location="cpu", weights_only=False)
            codes = []
            for text, code in rows[rid]["grid"]:
                if text == 151667:
                    if code == 16384:
                        break
                    codes.append(code)
            waves, ledger = decode_reference(
                ref, codes, voice, 5, case["body"]["seed"], rid
            )
            wave = torch.cat(waves)
            assert wave.numel() == len(codes) * 1920 and torch.isfinite(wave).all()
            pcm = (np.clip(wave.numpy(), -1, 1) * 32767.0).astype("<i2")
            (a.out_dir / f"{rid}.pcm").write_bytes(pcm.tobytes())
            sf.write(a.out_dir / f"{rid}.wav", pcm, 24000, subtype="PCM_16")
            torch.save(
                {"waves": waves, "ledger": ledger, "codes": codes},
                a.out_dir / f"{rid}.pt",
            )
            report["results"].append(
                dict(
                    id=rid,
                    samples=wave.numel(),
                    chunks=len(waves),
                    voice_state_sha256=sha256(voice_path),
                    pcm_sha256=sha256(a.out_dir / f"{rid}.pcm"),
                    ledger=ledger,
                    caches_empty=not ref.model.hift_cache_dict,
                )
            )
            print(rid, wave.numel(), len(waves), flush=True)
            (a.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["pass"] = len(report["results"]) == 16 and all(
            r["caches_empty"] for r in report["results"]
        )
    finally:
        (a.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
