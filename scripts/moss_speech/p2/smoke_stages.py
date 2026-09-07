#!/usr/bin/env python3
"""Stage-factory smoke test for MOSS-Speech (T2.4, GPU).

Exercises the real factories on a compute node (`.venv-omni`, offline):

1. preprocessing: text + audio requests -> codes equal to the P1 reference
   (per-code), grid sanity, voice payload fields populated;
2. audio_vocoder: decodes P1-alignment codes with the transported voice and
   the request effective_seed -> waveform blake2b equal to the P1 reference
   case; RNG restored after the call; same-seed/different-request equality;
3. text_decode: decodes the P0 t2t fixture grid -> text equal to fixture;
4. ar_engine: pre-checks pass, then the tagged P3 boundary error.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import numpy as np
import soundfile as sf
import torch

from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.models.moss_speech.request_builders import RequestValidationError
from sglang_omni.models.moss_speech.stages import (
    MossSpeechARNotImplemented,
    create_ar_engine_executor,
    create_audio_vocoder_executor,
    create_preprocessing_executor,
    create_text_decode_executor,
)
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest

SOSP, EOSP, MPAD = 151646, 16384, 151667


def blake2b(t) -> str:
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _omni(**chat_kwargs) -> OmniRequest:
    chat_kwargs.setdefault("messages", [{"role": "user", "content": "Introduce yourself in one sentence."}])
    return OmniRequest(inputs=_build_chat_generate_request(ChatCompletionRequest(model="m", **chat_kwargs)))


def _b64_wav(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _payload(rid: str, omni: OmniRequest, data=None) -> "StagePayload":
    from sglang_omni.proto.request import StagePayload

    return StagePayload(request_id=rid, request=omni, data=data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--voice-wav", required=True)
    parser.add_argument("--alignment-dir", required=True, help="artifacts/p1/alignment")
    parser.add_argument("--fixtures-dir", required=True, help="tests/fixtures/moss_speech")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    out: dict = {"gpu": torch.cuda.get_device_name(0)}
    align = Path(args.alignment_dir)
    ref = json.loads((align / "reference" / "reference_export.json").read_text())

    # ---------------- preprocessing ----------------------------------------
    pre = create_preprocessing_executor(args.model_path, voice_wav=args.voice_wav)
    text_result = pre._fn(_payload("smoke-text", _omni()))
    ts = MossSpeechState.from_dict(text_result.data)
    out["pre_text"] = {
        "modality": ts.output_modality,
        "grid_len": ts.prompt_grid_len,
        "has_codes": len(ts.audio_codes) == 0,
        "voice_absent": ts.voice_key is None,
    }

    audio_req = _omni(
        modalities=["audio"],
        messages=[{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": _b64_wav(args.voice_wav), "format": "wav"}}
        ]}],
    )
    audio_result = pre._fn(_payload("smoke-audio", audio_req))
    aus = MossSpeechState.from_dict(audio_result.data)
    ref_codes = ref["encode"]["single_cn_r0"]
    out["pre_audio"] = {
        "codes_equal_reference": aus.audio_codes[0] == ref_codes,
        "n_codes": len(aus.audio_codes[0]),
        "voice_key": aus.voice_key is not None,
        "voice_token_len": len(aus.voice_token_ids),
        "grid_len": aus.prompt_grid_len,
    }

    # ---------------- vocoder ----------------------------------------------
    voc = create_audio_vocoder_executor(args.model_path)
    # exact P1 "mixed" case codes (200) from the reference export
    cases_codes = torch.load(align / "reference" / "cases_codes.pt")
    mixed_codes = [int(c) for c in cases_codes["mixed"]]
    vstate = dict(aus.to_dict())
    vstate["output_grid"] = [[MPAD, c] for c in mixed_codes] + [[151645, 0]]
    vstate["effective_seed"] = 0

    import random

    py0, np0, t0 = random.getstate(), np.random.get_state(), torch.get_rng_state()
    vocoded = voc._fn(_payload("smoke-voc", audio_req, data=vstate))
    vs = MossSpeechState.from_dict(vocoded.data)
    py1, np1, t1 = random.getstate(), np.random.get_state(), torch.get_rng_state()
    out["voc"] = {
        "sr": vs.audio_sample_rate,
        "n_samples": len(vs.audio_samples),
        "hash_vs_p1_reference": blake2b(torch.tensor(vs.audio_samples)) == ref["decode"]["mixed_cn_s0"]["blake2b"],
        "rng_restored_py": py0 == py1,
        "rng_restored_np": np.array_equal(np0[1], np1[1]),
        "rng_restored_torch": torch.equal(t0, t1),
    }
    # same explicit seed, different request id -> identical waveform
    vstate2 = dict(vstate)
    voc2 = voc._fn(_payload("smoke-voc-2", audio_req, data=vstate2))
    vs2 = MossSpeechState.from_dict(voc2.data)
    out["voc"]["same_seed_diff_request_equal"] = (
        blake2b(torch.tensor(vs2.audio_samples)) == blake2b(torch.tensor(vs.audio_samples))
    )
    # different seed -> HiFT draws differ -> waveform differs
    vstate3 = dict(vstate)
    vstate3["effective_seed"] = 1
    voc3 = voc._fn(_payload("smoke-voc-3", audio_req, data=vstate3))
    vs3 = MossSpeechState.from_dict(voc3.data)
    out["voc"]["diff_seed_differs"] = (
        blake2b(torch.tensor(vs3.audio_samples)) != blake2b(torch.tensor(vs.audio_samples))
    )

    # ---------------- text_decode -------------------------------------------
    td = create_text_decode_executor(args.model_path)
    grid = torch.load(Path(args.fixtures_dir) / "t2t_short" / "tokens_grid.pt")
    expected = (Path(args.fixtures_dir) / "t2t_short" / "text.txt").read_text()
    tstate = MossSpeechState(output_grid=grid.tolist()).to_dict()
    tdone = td._fn(_payload("smoke-td", _omni(), data=tstate))
    tds = MossSpeechState.from_dict(tdone.data)
    out["text_decode"] = {"matches_fixture": tds.generated_text == expected,
                          "text_head": tds.generated_text[:40]}

    # ---------------- AR boundary -------------------------------------------
    try:
        create_ar_engine_executor(args.model_path, dtype="bfloat16")
        out["ar_boundary"] = {"raised": False}
    except MossSpeechARNotImplemented as exc:
        out["ar_boundary"] = {"raised": True, "message_head": str(exc)[:120]}

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    assert out["pre_audio"]["codes_equal_reference"]
    assert out["voc"]["hash_vs_p1_reference"]
    assert out["voc"]["rng_restored_py"] and out["voc"]["rng_restored_np"] and out["voc"]["rng_restored_torch"]
    assert out["voc"]["same_seed_diff_request_equal"] and out["voc"]["diff_seed_differs"]
    assert out["text_decode"]["matches_fixture"]
    assert out["ar_boundary"]["raised"]
    print("SMOKE STAGES PASSED")


if __name__ == "__main__":
    main()
