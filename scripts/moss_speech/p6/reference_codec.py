#!/usr/bin/env python3
"""Independent locked-HF/CosyVoice codec reference; execute in a GPU allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_reference(codec: Path, chunk: int, out: Path) -> Any:
    from cosyvoice.cli.model import CosyVoice2Model

    cls = get_class_from_dynamic_module(
        "modeling_moss_speech_codec.AudioDecoder", str(codec), local_files_only=True
    )
    config = out / f"config_chunk_{chunk}.yaml"
    config.write_text(
        (codec / "flow/config.yaml")
        .read_text()
        .replace("chunk_size: 5", f"chunk_size: {chunk}")
    )
    decoder = cls(
        config,
        codec / f"flow/flow-chunk-{chunk}.pt",
        codec / "flow/hift.pt",
        codec / "flow/campplus.onnx",
        device="cuda",
    ).eval()
    # Invoke the unchanged upstream token2wav method, without constructing an LLM.
    model = CosyVoice2Model.__new__(CosyVoice2Model)
    model.device, model.fp16 = torch.device("cuda"), False
    model.flow, model.hift = decoder.flow, decoder.hift
    model.hift_cache_dict = defaultdict(lambda: None)
    model.mel_cache_len, model.source_cache_len = 8, 8 * 480
    model.speech_window = np.hamming(2 * model.source_cache_len)
    return SimpleNamespace(decoder=decoder, model=model, config=config)


def decode_reference(
    ref: Any, codes: list[int], voice: dict, chunk: int, seed: int, rid: str
) -> tuple[list[torch.Tensor], list[dict]]:
    model = ref.model
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = dict(
        prompt_token=torch.tensor([voice["voice_token_ids"]], dtype=torch.int32),
        prompt_feat=torch.tensor([voice["voice_feat"]]),
        embedding=torch.tensor([voice["voice_embedding"]]),
        uuid=rid,
    )
    prompt_len = len(voice["voice_token_ids"])
    first = chunk + (-prompt_len) % chunk
    cursor, waves, ledger = 0, [], []
    hop = first
    try:
        while len(codes) - cursor >= hop + model.flow.pre_lookahead_len:
            available = cursor + hop + model.flow.pre_lookahead_len
            start = time.perf_counter()
            wave = (
                model.token2wav(
                    torch.tensor([codes[:available]]),
                    token_offset=cursor,
                    stream=True,
                    finalize=False,
                    **kwargs,
                )
                .detach()
                .cpu()
                .reshape(-1)
            )
            cursor += hop
            waves.append(wave)
            ledger.append(
                dict(
                    available=available,
                    committed_codes=cursor,
                    samples=wave.numel(),
                    final=False,
                    seconds=time.perf_counter() - start,
                )
            )
            hop = chunk
        start = time.perf_counter()
        wave = (
            model.token2wav(
                torch.tensor([codes]), token_offset=cursor, finalize=True, **kwargs
            )
            .detach()
            .cpu()
            .reshape(-1)
        )
        waves.append(wave)
        ledger.append(
            dict(
                available=len(codes),
                committed_codes=len(codes),
                samples=wave.numel(),
                final=True,
                seconds=time.perf_counter() - start,
            )
        )
        return waves, ledger
    finally:
        model.hift_cache_dict.pop(rid, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    voice = torch.load(args.state, map_location="cpu", weights_only=False)
    codes = []
    for text, audio in voice["output_grid"]:
        if text != 151667:
            continue
        if audio == 16384:
            break
        codes.append(audio)
    report = dict(
        pass_=False,
        gpu=torch.cuda.get_device_name(0),
        profile="chunk-cudnn-deterministic-v1",
        input_sha256=sha(args.state),
        source_sha256=sha(Path(__file__)),
        profiles={},
        legacy_helpers={},
    )
    report["pass"] = report.pop("pass_")
    try:
        for chunk in (5, 25):
            ref = make_reference(args.codec_path, chunk, args.out_dir)
            lengths = sorted(
                {
                    1,
                    chunk - 1,
                    chunk,
                    chunk + 1,
                    chunk + 3,
                    chunk * 2,
                    chunk * 2 + 3,
                    chunk * 3 + 13,
                }
            )
            profile = dict(
                weight_sha256=sha(args.codec_path / f"flow/flow-chunk-{chunk}.pt"),
                config_sha256=sha(ref.config),
                cases={},
            )
            report["profiles"][str(chunk)] = profile
            for n in lengths:
                for seed in (0, 1):
                    rid = f"c{chunk}_n{n}_s{seed}"
                    waves, ledger = decode_reference(
                        ref, codes[:n], voice, chunk, seed, rid
                    )
                    combined = torch.cat(waves)
                    torch.save(
                        {"waves": waves, "ledger": ledger, "codes": codes[:n]},
                        args.out_dir / f"{rid}.pt",
                    )
                    profile["cases"][rid] = dict(
                        n=n,
                        seed=seed,
                        samples=combined.numel(),
                        finite=bool(torch.isfinite(combined).all()),
                        chunks=len(waves),
                        ledger=ledger,
                        correct_length=combined.numel() == n * 1920,
                        caches_empty=not ref.model.hift_cache_dict,
                    )
                    print(rid, profile["cases"][rid], flush=True)
            if chunk == 5:
                for helper in ("stream_inference", "streaming_inference"):
                    kwargs = dict(
                        prompt_token=torch.tensor(
                            [voice["voice_token_ids"]], dtype=torch.int32
                        ),
                        prompt_feat=torch.tensor([voice["voice_feat"]]),
                        embedding=torch.tensor([voice["voice_embedding"]]),
                    )
                    try:
                        if helper == "stream_inference":
                            result = ref.decoder.stream_inference(
                                torch.tensor([codes[:13]]), block_size=5, **kwargs
                            )
                        else:
                            _, mel, tokens = ref.decoder.streaming_inference(
                                torch.tensor([codes[:8]]),
                                uuid="legacy",
                                is_finalize=False,
                                **kwargs,
                            )
                            result = ref.decoder.streaming_inference(
                                torch.tensor([codes[8:13]]),
                                uuid="legacy",
                                prev_mel=mel,
                                prev_token=tokens,
                                **kwargs,
                            )[0]
                        report["legacy_helpers"][helper] = {"samples": result.numel()}
                    except Exception as exc:
                        report["legacy_helpers"][helper] = {"error": repr(exc)}
                    finally:
                        ref.decoder.hift_cache_dict.clear()
                        ref.decoder.mel_overlap_dict.clear()
            del ref
            torch.cuda.empty_cache()
        report["pass"] = all(
            c["correct_length"] and c["finite"] and c["caches_empty"]
            for p in report["profiles"].values()
            for c in p["cases"].values()
        )
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
