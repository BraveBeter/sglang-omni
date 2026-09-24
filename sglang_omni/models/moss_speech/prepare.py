# SPDX-License-Identifier: Apache-2.0
"""Prepare the optional native packages and fixed voice before starting serving.

Run on the deployment host, with its Python environment and allocated GPU.
Outputs are deployment artifacts, never repository files or downloadable weights.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def prepare_prefix(codec_path: Path, voice_wav: Path, out: Path, device: int) -> Path:
    import torch

    from .components.cfm_prefix_cache import (
        build_prepared_manifest,
        runtime_environment,
    )
    from .components.codec_adapter import MossSpeechCodecAdapter
    from .components.streaming_codec import MossSpeechStreamingCodec

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    adapter = MossSpeechCodecAdapter(
        codec_path, device=f"cuda:{device}", load_decoder=False
    )
    try:
        voice = adapter.encode_voice_ref(voice_wav)
        payload = {
            name: getattr(voice, name).detach().cpu().contiguous()
            for name in ("prompt_token", "prompt_feat", "embedding")
        }
    finally:
        adapter.close()
    torch.save(payload, out / "voice.pt")
    shutil.copyfile(voice_wav, out / "reference.wav")
    # Use valid audio codes from this voice as deterministic startup controls.
    codes = payload["prompt_token"].flatten().tolist()[:36]
    manifest = build_prepared_manifest(
        voice_path=out / "voice.pt",
        wave_path=out / "reference.wav",
        codec_root=codec_path,
        source_root=Path(__file__).parent,
        environment=runtime_environment(torch.device(f"cuda:{device}")),
        warmup_audio_codes=codes,
    )
    manifest["voice"]["path"] = "voice.pt"
    manifest["reference_wave"]["path"] = "reference.wav"
    path = out / "manifest.json"
    # Validate the same complete CFM profile before publishing a usable manifest.
    pending = out / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    codec = MossSpeechStreamingCodec(
        codec_path,
        device=f"cuda:{device}",
        chunk_size=25,
        codec_optimizations=True,
        stream_startup_schedule="6_8_align25",
        codec_final_bias_alignment=True,
        codec_final_bias_elision=True,
        codec_qkv_multistream=True,
    )
    try:
        receipt = dict(
            prefix=codec.prepare_prefix_cache(pending),
            graphs=codec.prepare_full_graphs(),
        )
        (out / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        codec.close()
    pending.rename(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    norm = sub.add_parser(
        "norm", help="Build and numerically verify the CUDA norm package"
    )
    norm.add_argument("--out", required=True, type=Path)
    norm.add_argument("--nvcc", default="nvcc")
    epoch = sub.add_parser(
        "epoch", help="Build and verify the CPU state metadata package"
    )
    epoch.add_argument("--out", required=True, type=Path)
    epoch.add_argument("--cxx", default="g++")
    prefix = sub.add_parser(
        "prefix", help="Encode the fixed voice and verify CFM preparation"
    )
    prefix.add_argument("--codec-path", required=True, type=Path)
    prefix.add_argument("--voice-wav", required=True, type=Path)
    prefix.add_argument("--out", required=True, type=Path)
    prefix.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.command == "norm":
        from .native_build import build_norm

        result = build_norm(args.out, args.nvcc)
    elif args.command == "epoch":
        from .native_build import build_epoch

        result = build_epoch(args.out, args.cxx)
    else:
        result = prepare_prefix(args.codec_path, args.voice_wav, args.out, args.device)
    print(result)


if __name__ == "__main__":
    main()
