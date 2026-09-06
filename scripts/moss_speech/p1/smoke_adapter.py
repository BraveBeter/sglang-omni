#!/usr/bin/env python3
"""GPU smoke test for the MOSS-Speech codec adapter (T1.2, G1).

Runs in the TARGET environment (.venv-omni, offline) on a compute node and
verifies component-selective loading, basic encode/decode sanity against the
P0 contract numbers, voice determinism, and idempotent cleanup. Full numeric
acceptance is T1.3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import torch  # noqa: E402

from sglang_omni.models.moss_speech.components.codec_adapter import MossSpeechCodecAdapter  # noqa: E402


def _vram() -> dict:
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", required=True)
    parser.add_argument("--voice-wav", required=True, help="24k-capable reference wav (any sr)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    out: dict = {"node": os.uname().nodename, "gpu": torch.cuda.get_device_name(0)}

    # --- encoder-only ------------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    enc_adapter = MossSpeechCodecAdapter(args.codec_path, load_encoder=True, load_decoder=False)
    assert enc_adapter.encoder_loaded and not enc_adapter.decoder_loaded
    assert enc_adapter._decoder is None, "encoder-only must not construct AudioDecoder"
    out["encoder_only"] = _vram()
    codes = enc_adapter.encode([args.voice_wav])[0]
    out["encoder_only"]["n_codes"] = len(codes)
    out["encoder_only"]["code_min"] = int(min(codes))
    out["encoder_only"]["code_max"] = int(max(codes))
    voice = enc_adapter.encode_voice_ref(args.voice_wav)
    out["encoder_only"]["voice_shapes"] = {
        "prompt_token": list(voice.prompt_token.shape),
        "prompt_feat": list(voice.prompt_feat.shape),
        "embedding": list(voice.embedding.shape),
    }
    voice2 = enc_adapter.encode_voice_ref(args.voice_wav)
    out["encoder_only"]["voice_deterministic"] = bool(
        torch.equal(voice.prompt_token, voice2.prompt_token)
        and torch.equal(voice.prompt_feat, voice2.prompt_feat)
        and torch.equal(voice.embedding, voice2.embedding)
    )
    enc_adapter.close()
    enc_adapter.close()
    torch.cuda.empty_cache()

    # --- decoder-only ------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    dec_adapter = MossSpeechCodecAdapter(args.codec_path, load_encoder=False, load_decoder=True)
    assert dec_adapter.decoder_loaded and not dec_adapter.encoder_loaded
    assert dec_adapter._encoder is None, "decoder-only must not construct WhisperVQ"
    out["decoder_only"] = _vram()
    try:
        dec_adapter.encode([args.voice_wav])
        raise AssertionError("decode-only adapter must reject encode()")
    except Exception as e:
        out["decoder_only"]["encode_rejected"] = type(e).__name__
    torch.cuda.empty_cache()

    # --- full stack --------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    adapter = MossSpeechCodecAdapter(args.codec_path)
    out["full"] = _vram()
    n = 200
    sr, wav = adapter.decode(codes[:n], voice, request_id="smoke-1")
    out["full"]["decode"] = {
        "sr": sr,
        "n_samples": int(wav.shape[0]),
        "mono": wav.dim() == 1,
        "finite": bool(torch.isfinite(wav).all()),
        "implied_hop_ms": round(wav.shape[0] / n / sr * 1000, 2),
    }
    adapter.cleanup("smoke-1")
    adapter.cleanup("smoke-1")  # idempotent
    try:
        adapter.decode([99999], voice, request_id="smoke-bad")
        raise AssertionError("out-of-range code must be rejected")
    except ValueError as e:
        out["full"]["bad_code_rejected"] = str(e)[:60]
    adapter.close()

    text = json.dumps(out, indent=1)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text + "\n")
    # hard gates from the P0 contract
    assert 200 <= out["encoder_only"]["n_codes"] <= 4000
    assert out["encoder_only"]["code_min"] >= 0 and out["encoder_only"]["code_max"] < 16384
    assert out["encoder_only"]["voice_deterministic"]
    assert out["full"]["decode"]["sr"] == 24000
    assert out["full"]["decode"]["finite"]
    assert 70 <= out["full"]["decode"]["implied_hop_ms"] <= 90
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
