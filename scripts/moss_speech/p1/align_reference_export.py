#!/usr/bin/env python3
"""Reference-side export for the T1.3 alignment run.

Runs the LOCKED REFERENCE codec (HF remote code + GitHub cosyvoice via
PYTHONPATH) under `.venv-p0` and dumps codes / conditioning / waveforms /
hashes / RNG snapshots for the frozen manifest (see 02_alignment.md).
The torchaudio soundfile shim from P0 is applied (reference environment
cannot use TorchCodec on the cluster).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torchaudio
from transformers import AutoModel

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "p0")
)  # run_reference shim
import run_reference as rr  # noqa: E402

rr._install_torchaudio_load_shim()

SOSP, EOSP = 151646, 16384


def blake2b(t: torch.Tensor) -> str:
    return hashlib.blake2b(
        t.detach().cpu().contiguous().numpy().tobytes(), digest_size=16
    ).hexdigest()


def rng_snapshot() -> dict:
    return {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def rng_equal(a: dict, b: dict) -> bool:
    return (
        a["py"] == b["py"]
        and np.array_equal(np.asarray(a["np"][1]), np.asarray(b["np"][1]))
        and torch.equal(a["torch"], b["torch"])
        and len(a["cuda"]) == len(b["cuda"])
        and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))
    )


def extract_codes_from_grid(grid: torch.Tensor) -> list:
    MPAD = 151667
    text_ch, audio_ch = grid[:, 0].tolist(), grid[:, 1].tolist()
    # Audio-output grids: the sosp/object_ref boundary lives in the prompt, and
    # the generated text channel streams modality_pad while audio codes are
    # emitted; eosp (16384) on the audio channel ends the segment when present.
    stop = audio_ch.index(EOSP) if EOSP in audio_ch else len(audio_ch)
    seg = list(zip(text_ch, audio_ch))[:stop]
    if seg and all(t == MPAD for t, _ in seg):
        return [int(a) for _, a in seg]
    if SOSP in text_ch:  # safety path for prompt-inclusive grids
        start = text_ch.index(SOSP) + 1
        return [int(t) for t in audio_ch[start:stop]]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", default="models/MOSS-Speech-Codec")
    parser.add_argument("--fixtures", default="artifacts/p0/fixtures")
    parser.add_argument("--assets", default="repos/MOSS-Speech/assets")
    parser.add_argument("--out", default="artifacts/p1/alignment/reference")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cn, en = f"{args.assets}/prompt-cn.wav", f"{args.assets}/prompt-en.wav"
    codec = (
        AutoModel.from_pretrained(args.codec_path, trust_remote_code=True)
        .to("cuda")
        .eval()
    )

    result: dict = {
        "env": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        }
    }

    # ---------------- encode manifest ----------------
    enc: dict = {}
    wav16_cn, sr_cn = torchaudio.load(cn)
    wav16_en, sr_en = torchaudio.load(en)
    inputs = {"cn": cn, "en": en}
    tensors = {"cn": (wav16_cn, sr_cn), "en": (wav16_en, sr_en)}

    for rep in range(2):
        for name, path in inputs.items():
            enc[f"single_{name}_r{rep}"] = codec.encode([path])[0]
    for bs in (1, 4, 128):
        batch = [cn, en] if bs == 1 else [cn, en] * 2 if bs == 4 else [cn, en] * 64
        codes = codec.encode(batch, batch_size=bs)
        # per-position sample identity for mixed-length batch proof
        enc[f"batch_bs{bs}"] = {
            "lens": [len(c) for c in codes],
            "first_cn": codes[0],
            "second_en": codes[1],
            "last_item": codes[-1],
        }
    for name, (wav, sr) in tensors.items():
        enc[f"tuple_{name}"] = codec.encode([(wav, sr)])[0]
    wav16 = wav16_cn
    enc["tensor16k_cn"] = codec.encode([wav16[0].cuda()])[0]
    result["encode"] = enc

    # ---------------- decode manifest ----------------
    fx = Path(args.fixtures)
    cases = {
        "t2s_cn": extract_codes_from_grid(torch.load(fx / "t2s_cn" / "tokens_grid.pt")),
        "s2s_cn": extract_codes_from_grid(torch.load(fx / "s2s_cn" / "tokens_grid.pt")),
        "mixed": extract_codes_from_grid(
            torch.load(
                Path("artifacts/p0/runs_reference_greedy/mixed_t2_s2s/tokens.pt")
            )[0]
        ),
        "single": [100],
        "long500": None,
    }
    cases["long500"] = (cases["mixed"] * ((500 // max(len(cases["mixed"]), 1)) + 1))[
        :500
    ]
    result["codes_meta"] = {k: len(v) for k, v in cases.items()}

    dec: dict = {}
    rng_before_after = {}
    for case, codes in cases.items():
        for vname, vpath in (("cn", cn), ("en", en)):
            for seed in (0, 1):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                if (case, vname, seed) == ("mixed", "cn", 0):
                    rng_before_after["before"] = rng_snapshot()
                try:
                    r = codec.decode(
                        torch.tensor([codes]).reshape(1, 1, -1), prompt_speech=vpath
                    )
                    wav = r["syn_wav_list"][0].detach().cpu()
                    dec[f"{case}_{vname}_s{seed}"] = {
                        "n": int(wav.shape[-1]),
                        "blake2b": blake2b(wav),
                        "finite": bool(torch.isfinite(wav).all()),
                    }
                    torch.save(wav, out / f"wav_{case}_{vname}_s{seed}.pt")
                except Exception as e:
                    dec[f"{case}_{vname}_s{seed}"] = {
                        "error": f"{type(e).__name__}: {str(e)[:120]}"
                    }
                if (case, vname, seed) == ("mixed", "cn", 0):
                    rng_before_after["after"] = rng_snapshot()
    result["decode"] = dec
    result["rng_unchanged_by_reference_decode"] = rng_equal(
        rng_before_after["before"], rng_before_after["after"]
    )

    # ---------------- conditioning (reference internals) ----------------
    cond: dict = {}
    for vname, vpath in (("cn", cn), ("en", en)):
        prompt_wav, orig_sr = torchaudio.load(vpath)
        if orig_sr != 24000:
            prompt_wav = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=24000
            )(prompt_wav)
        speech_feat, _ = codec._extract_speech_feat(prompt_wav)
        speech_token = torch.tensor(codec.encode([vpath])[0]).unsqueeze(0)
        token_len = min(int(speech_feat.shape[1] / 4), speech_token.shape[1])
        prompt_16k = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(
            prompt_wav
        )
        emb = codec._extract_spk_embedding(prompt_16k)
        cond[vname] = {
            "token_len": token_len,
            "codes_first20": speech_token[0, :token_len][:20].tolist(),
            "codes_all": speech_token[0, :token_len].tolist(),
            "feat_shape": list(speech_feat[:, : 4 * token_len].shape),
            "feat_blake2b": blake2b(speech_feat[:, : 4 * token_len]),
            "emb_blake2b": blake2b(emb),
        }
    result["conditioning"] = cond

    # ---------------- edge probes ----------------
    edges: dict = {}
    try:
        r = codec.decode(torch.zeros(1, 1, 0, dtype=torch.long), prompt_speech=cn)
        wav = r["syn_wav_list"][0]
        edges["empty_codes"] = {
            "n": int(wav.shape[-1]),
            "blake2b": blake2b(wav.detach().cpu()),
        }
    except Exception as e:
        edges["empty_codes"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    try:
        r = codec.decode(torch.tensor([[[20000]]]), prompt_speech=cn)
        wav = r["syn_wav_list"][0]
        edges["code_20000"] = {
            "n": int(wav.shape[-1]),
            "blake2b": blake2b(wav.detach().cpu()),
        }
    except Exception as e:
        edges["code_20000"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    result["edges"] = edges

    (out / "reference_export.json").write_text(json.dumps(result, indent=1))
    torch.save(cases, out / "cases_codes.pt")
    print(
        json.dumps(
            {
                k: (
                    v
                    if k != "encode"
                    else {
                        kk: len(vv) if isinstance(vv, list) else "..."
                        for kk, vv in v.items()
                    }
                )
                for k, v in result.items()
            },
            indent=1,
        )[:1200]
    )
    print("REFERENCE EXPORT DONE")


if __name__ == "__main__":
    main()
