#!/usr/bin/env python3
"""Codec contract verification for MOSS-Speech-Codec (P0 / T0.7).

Measures encoder/decoder behavior against the contract expectations:
frame rate 12.5 Hz, code range, input/output sample rates, mono output,
prompt-speech voice conditioning, decode determinism, streaming API surface.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import torch  # noqa: E402
from transformers import AutoModel  # noqa: E402

import run_reference as rr  # noqa: E402


def wav_meta(wav: torch.Tensor, sr: int) -> Dict[str, Any]:
    arr = wav.detach().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return {
        "sr": sr,
        "n_samples": int(arr.shape[0]),
        "duration_s": round(arr.shape[0] / sr, 3),
        "dtype": str(arr.dtype),
        "blake2b": hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", default="models/MOSS-Speech-Codec")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rr._install_torchaudio_load_shim()
    codec = AutoModel.from_pretrained(args.codec_path, trust_remote_code=True).to("cuda").eval()
    res: Dict[str, Any] = {}

    assets = Path(args.assets_dir)
    cn, en = str(assets / "prompt-cn.wav"), str(assets / "prompt-en.wav")

    import soundfile as sf

    info_cn = sf.info(cn)
    res["input_assets"] = {
        "prompt-cn.wav": {"sr": info_cn.samplerate, "channels": info_cn.channels, "dur_s": round(info_cn.duration, 2)},
    }

    # --- encoder ------------------------------------------------------------
    codes_cn = codec.encode([cn])[0]
    codes_en = codec.encode([en])[0]
    res["encoder"] = {
        "cn_n_codes": len(codes_cn),
        "en_n_codes": len(codes_en),
        "cn_dur_s": round(info_cn.duration, 2),
        "implied_frame_rate_cn": round(len(codes_cn) / info_cn.duration, 3),
        "code_min": int(min(codes_cn)),
        "code_max": int(max(codes_cn)),
        "codebook_size_claim": 16384,
        "batch_encode_lens": [len(c) for c in codec.encode([cn, en])],
    }
    # tensor input path (assumes 16k)
    wav16, _ = sf.read(cn, dtype="float32")
    import numpy as np

    wav16t = torch.from_numpy(np.tile(wav16[:16000], (1,))).unsqueeze(0)  # 1s mono
    codes_t = codec.encode([wav16t])[0]
    res["encoder"]["tensor_input_1s_n_codes"] = len(codes_t)

    # --- decoder ------------------------------------------------------------
    torch.manual_seed(0)
    out_cn = codec.decode(torch.tensor([codes_cn[:200]]).reshape(1, 1, -1), prompt_speech=cn)
    wav_cn, meta1 = out_cn["syn_wav_list"][0], None
    meta1 = wav_meta(wav_cn, 24000)
    torch.manual_seed(0)
    out_cn2 = codec.decode(torch.tensor([codes_cn[:200]]).reshape(1, 1, -1), prompt_speech=cn)
    meta2 = wav_meta(out_cn2["syn_wav_list"][0], 24000)
    torch.manual_seed(0)
    out_en = codec.decode(torch.tensor([codes_cn[:200]]).reshape(1, 1, -1), prompt_speech=en)
    meta3 = wav_meta(out_en["syn_wav_list"][0], 24000)

    res["decoder"] = {
        "decode_200codes_with_cn_prompt": meta1,
        "decode_same_seed_rerun": {"blake2b": meta2["blake2b"], "deterministic": meta1["blake2b"] == meta2["blake2b"]},
        "decode_same_codes_en_prompt": {"blake2b": meta3["blake2b"], "differs_from_cn_prompt": meta3["blake2b"] != meta1["blake2b"]},
        "implied_hop_ms": round(meta1["duration_s"] / 200 * 1000, 2),
        "output_mono": wav_cn.ndim == 1 or wav_cn.shape[0] == 1,
        "output_sr": 24000,
    }

    # --- streaming API surface ----------------------------------------------
    dec = codec.audio_decoder
    res["streaming_api"] = {
        "methods": [n for n in ("offline_inference", "stream_inference", "streaming_inference") if hasattr(dec, n)],
        "signatures": {
            n: str(inspect.signature(getattr(dec, n)))[:400]
            for n in ("stream_inference", "streaming_inference")
            if hasattr(dec, n)
        },
        "scratch_config": {
            "sample_rate": dec.scratch_configs.get("sample_rate"),
            "keys": list(dec.scratch_configs.keys())[:12],
        },
        "flow_files": sorted(p.name for p in (Path(args.codec_path) / "flow").glob("*.pt")),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
