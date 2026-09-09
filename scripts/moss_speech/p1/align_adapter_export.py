#!/usr/bin/env python3
"""Adapter-side export for the T1.3 alignment run.

Runs the ADAPTER (vendored closure) under `.venv-omni` in a clean
environment and produces the same manifest as the reference export, plus the
isolation/failure-injection probes (A7/A8/A9).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from sglang_omni.models.moss_speech.components.codec_adapter import (
    MossSpeechCodecAdapter,
)

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
    parser.add_argument("--out", default="artifacts/p1/alignment/adapter")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cn, en = f"{args.assets}/prompt-cn.wav", f"{args.assets}/prompt-en.wav"
    adapter = MossSpeechCodecAdapter(args.codec_path)

    result: dict = {
        "env": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
        }
    }

    # ---------------- encode manifest (path / tuple / tensor forms) --------
    enc: dict = {}
    for rep in range(2):
        enc[f"single_cn_r{rep}"] = adapter.encode([cn])[0]
        enc[f"single_en_r{rep}"] = adapter.encode([en])[0]
    import soundfile as sf

    wav_cn, sr_cn = sf.read(cn, dtype="float32", always_2d=True)
    wav_en, sr_en = sf.read(en, dtype="float32", always_2d=True)
    t_cn = torch.from_numpy(wav_cn.T)
    t_en = torch.from_numpy(wav_en.T)
    for bs in (1, 4, 128):
        batch = [cn, en] if bs == 1 else [cn, en] * 2 if bs == 4 else [cn, en] * 64
        codes = adapter.encode(batch, batch_size=bs)
        enc[f"batch_bs{bs}"] = {
            "lens": [len(c) for c in codes],
            "first_cn": codes[0],
            "second_en": codes[1],
            "last_item": codes[-1],
        }
    enc["tuple_cn"] = adapter.encode([(t_cn, sr_cn)])[0]
    enc["tuple_en"] = adapter.encode([(t_en, sr_en)])[0]
    enc["tensor16k_cn"] = adapter.encode([t_cn[0].cuda()])[
        0
    ]  # (T,) 44.1k treated per contract? NO — see note
    result["encode"] = enc

    # ---------------- voices ------------------------------------------------
    voices = {"cn": adapter.encode_voice_ref(cn), "en": adapter.encode_voice_ref(en)}
    cond: dict = {}
    for vname, v in voices.items():
        cond[vname] = {
            "token_len": int(v.prompt_token.shape[1]),
            "codes_first20": v.prompt_token[0][:20].tolist(),
            "codes_all": v.prompt_token[0].tolist(),
            "feat_shape": list(v.prompt_feat.shape),
            "feat_blake2b": blake2b(v.prompt_feat),
            "emb_blake2b": blake2b(v.embedding),
        }
    result["conditioning"] = cond

    # ---------------- decode manifest + RNG isolation -----------------------
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
    rng_reports: dict = {}
    for case, codes in cases.items():
        for vname in ("cn", "en"):
            for seed in (0, 1):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                if (case, vname, seed) == ("mixed", "cn", 0):
                    before = rng_snapshot()
                try:
                    sr, wav = adapter.decode(
                        codes, voices[vname], request_id=f"{case}-{vname}-s{seed}"
                    )
                    dec[f"{case}_{vname}_s{seed}"] = {
                        "n": int(wav.shape[-1]),
                        "blake2b": blake2b(wav),
                        "finite": bool(torch.isfinite(wav).all()),
                        "sr": sr,
                    }
                    torch.save(wav, out / f"wav_{case}_{vname}_s{seed}.pt")
                except Exception as e:
                    dec[f"{case}_{vname}_s{seed}"] = {
                        "error": f"{type(e).__name__}: {str(e)[:120]}"
                    }
                if (case, vname, seed) == ("mixed", "cn", 0):
                    rng_reports["decode_rng_unchanged"] = rng_equal(
                        before, rng_snapshot()
                    )
    result["decode"] = dec
    enc_before = rng_snapshot()
    _ = adapter.encode([cn])
    _ = adapter.encode_voice_ref(cn)
    result["rng_unchanged_by_encode_and_voice"] = rng_equal(enc_before, rng_snapshot())
    result.update(rng_reports)

    # ---------------- edges -------------------------------------------------
    edges: dict = {}
    try:
        sr, wav = adapter.decode([], voices["cn"], request_id="empty")
        edges["empty_codes"] = {"n": int(wav.shape[-1]), "blake2b": blake2b(wav)}
    except Exception as e:
        edges["empty_codes"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    try:
        sr, wav = adapter.decode([20000], voices["cn"], request_id="oor")
        edges["code_20000"] = {"n": int(wav.shape[-1])}
    except Exception as e:
        edges["code_20000"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    result["edges"] = edges

    # ---------------- isolation (A8) ----------------------------------------
    # Seeded per call: HiFT consumes global RNG, so equality requires the same
    # pre-call seed (this is exactly the request-scoped RNG contract for P2).
    iso: dict = {}

    def _seeded_decode(codes, voice, rid, seed=7):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return adapter.decode(codes, voice, request_id=rid)

    _, solo_a1 = _seeded_decode(cases["mixed"], voices["cn"], "iso-a1")
    _, solo_b1 = _seeded_decode(cases["single"], voices["en"], "iso-b1")
    _, inter_a = _seeded_decode(cases["mixed"], voices["cn"], "iso-a2")
    _, inter_b = _seeded_decode(cases["single"], voices["en"], "iso-b2")
    iso["interleaved_matches_solo"] = bool(
        blake2b(solo_a1) == blake2b(inter_a) and blake2b(solo_b1) == blake2b(inter_b)
    )
    # failure injection: bad voice shape -> error -> cleanup -> recovery
    from sglang_omni.models.moss_speech.components.voice import VoiceConditioning

    bad = VoiceConditioning(  # genuinely invalid: wrong mel dimension
        prompt_token=torch.zeros(1, 4, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 16, 40),
        embedding=torch.zeros(1, 192),
        meta={},
    )
    try:
        adapter.decode(cases["mixed"], bad, request_id="fail-1")
        iso["bad_voice_raised"] = "NO (unexpected)"
    except Exception as e:
        iso["bad_voice_raised"] = type(e).__name__
    adapter.cleanup("fail-1")
    adapter.cleanup("fail-1")  # duplicate cleanup must be safe
    _, recov = _seeded_decode(cases["mixed"], voices["cn"], "recov")
    iso["recovery_matches_solo"] = bool(blake2b(recov) == blake2b(solo_a1))
    iso["sessions_after_finalize"] = len(adapter._sessions)
    result["isolation"] = iso

    adapter.close()
    (out / "adapter_export.json").write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "codes_meta",
                    "isolation",
                    "edges",
                    "rng_unchanged_by_encode_and_voice",
                    "decode_rng_unchanged",
                )
            },
            indent=1,
        )
    )
    print("ADAPTER EXPORT DONE")


if __name__ == "__main__":
    main()
