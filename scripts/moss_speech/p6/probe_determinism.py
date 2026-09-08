#!/usr/bin/env python3
"""Diagnose repeatability before changing a frozen streaming profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--codec-path", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = torch.load(a.state, map_location="cpu", weights_only=False)
    voice = dict(
        prompt_token=torch.tensor([source["voice_token_ids"]], dtype=torch.int32),
        prompt_feat=torch.tensor([source["voice_feat"]]),
        embedding=torch.tensor([source["voice_embedding"]]),
    )
    codes = [
        row[1] for row in source["output_grid"] if row[0] == 151667 and row[1] < 16384
    ]
    codec = MossSpeechStreamingCodec(a.codec_path, chunk_size=5)
    report = {}
    for mode in ("default", "cudnn", "deterministic"):
        torch.backends.cudnn.deterministic = mode != "default"
        torch.use_deterministic_algorithms(mode == "deterministic")
        output = []
        try:
            for rep in range(6):
                rid = f"{mode}-{rep}"
                codec.begin(rid, voice, seed=0)
                wave = torch.cat(codec.push(rid, codes[:28], final=True))
                output.append(wave)
            report[mode] = dict(
                equal=all(torch.equal(output[0], o) for o in output),
                max_abs=max(float((output[0] - o).abs().max()) for o in output),
            )
        except Exception as exc:
            report[mode] = {"error": repr(exc)}
        print(mode, report[mode], flush=True)
    (a.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
