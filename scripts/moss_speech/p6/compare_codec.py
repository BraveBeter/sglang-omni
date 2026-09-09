#!/usr/bin/env python3
"""Compare serving codec chunks with separately exported locked reference chunks."""
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
    p.add_argument("--codec-path", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = torch.load(args.state, map_location="cpu", weights_only=False)
    voice = dict(
        prompt_token=torch.tensor([source["voice_token_ids"]], dtype=torch.int32),
        prompt_feat=torch.tensor([source["voice_feat"]]),
        embedding=torch.tensor([source["voice_embedding"]]),
    )
    reference = json.loads((args.reference_dir / "report.json").read_text())
    assert reference["pass"]
    report = {
        "pass": False,
        "cases": {},
        "interleaved": {},
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        for chunk in (5, 25):
            codec = MossSpeechStreamingCodec(args.codec_path, chunk_size=chunk)
            cases = reference["profiles"][str(chunk)]["cases"]
            for rid, meta in cases.items():
                data = torch.load(
                    args.reference_dir / f"{rid}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                cpu_rng, gpu_rng = (
                    torch.get_rng_state().clone(),
                    torch.cuda.get_rng_state().clone(),
                )
                codec.begin(rid, voice, seed=meta["seed"])
                waves = []
                state = codec.sessions[rid]
                for code in data["codes"]:
                    waves.extend(codec.push(rid, [code]))
                waves.extend(codec.push(rid, [], final=True))
                same_shapes = len(waves) == len(data["waves"]) and all(
                    a.shape == b.shape for a, b in zip(waves, data["waves"])
                )
                equal = same_shapes and all(
                    torch.equal(a, b) for a, b in zip(waves, data["waves"])
                )
                report["cases"][rid] = dict(
                    equal=equal,
                    samples=sum(w.numel() for w in waves),
                    chunks=len(waves),
                    max_abs=(
                        max(
                            float((a - b).abs().max())
                            for a, b in zip(waves, data["waves"])
                        )
                        if same_shapes
                        else None
                    ),
                    rng_restored=torch.equal(cpu_rng, torch.get_rng_state())
                    and torch.equal(gpu_rng, torch.cuda.get_rng_state()),
                    caches_empty=not codec.sessions,
                )
                torch.save(
                    {"waves": waves, "ledger": state.ledger}, args.out_dir / f"{rid}.pt"
                )
                print(rid, report["cases"][rid], flush=True)
            longest = max(m["n"] for m in cases.values())
            pair = {
                seed: torch.load(
                    args.reference_dir / f"c{chunk}_n{longest}_s{seed}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for seed in (0, 1)
            }
            actual = {seed: [] for seed in pair}
            for seed in pair:
                codec.begin(f"interleave-{seed}", voice, seed=seed)
            codec.begin("cancel", voice, seed=9)
            codec.push("cancel", pair[0]["codes"][: chunk * 2 + 3])
            codec.cleanup("cancel")
            codec.cleanup("cancel")
            for i in range(longest):
                for seed in (1, 0):
                    actual[seed].extend(
                        codec.push(f"interleave-{seed}", [pair[seed]["codes"][i]])
                    )
            for seed in pair:
                actual[seed].extend(codec.push(f"interleave-{seed}", [], final=True))
            report["interleaved"][str(chunk)] = (
                all(
                    torch.equal(torch.cat(actual[s]), torch.cat(pair[s]["waves"]))
                    for s in pair
                )
                and not codec.sessions
            )
            codec.close()
            del codec
            torch.cuda.empty_cache()
        report["pass"] = all(
            r["equal"] and r["rng_restored"] and r["caches_empty"]
            for r in report["cases"].values()
        ) and all(report["interleaved"].values())
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
