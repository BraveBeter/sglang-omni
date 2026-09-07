#!/usr/bin/env python3
"""Multi-process positive smoke with the test-only AR stub (T2.5 step 4).

Builds a config copy of the formal YAML with ONLY the ar_engine factory
replaced by the replay stub, starts the REAL MultiProcessPipelineRunner
(real processes, real queues, real routing wrapper), and drives fixed
four-mode-style cases through preprocessing -> AR stub -> terminal:

  - text request  -> text_decode terminal, text == P0 fixture output
  - audio request (seed=0, prompt-cn input) -> audio_vocoder terminal,
    waveform blake2b == P1 reference mixed_cn_s0 (bit-exact through the
    full multi-process path)
  - abort-then-submit recovery

Run on a compute node in `.venv-omni` (offline).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/remote-home1/xrluan/.cache/huggingface")

import numpy as np
import torch

# repo root must be importable in the spawned stage processes (the editable
# install maps only the sglang_omni package; the test AR stub lives under
# tests/unit_test/moss_speech/). Spawned children inherit sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang_omni.config.manager import ConfigManager
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest


def _blake(arr) -> str:
    a = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    return hashlib.blake2b(np.asarray(a, dtype=np.float32).tobytes(), digest_size=16).hexdigest()


def _chat_omni(**kwargs):
    """Dict-serializable OmniRequest (control plane uses msgpack)."""
    from sglang_omni.proto.request import OmniRequest

    gen = _build_chat_generate_request(ChatCompletionRequest(model="m", **kwargs))
    return OmniRequest(inputs=gen.to_dict())


async def main_async(args) -> dict:
    manager = ConfigManager.from_file(args.config)
    config = manager.config
    # swap ONLY the ar factory to the replay stub (test config copy)
    for stage in config.stages:
        if stage.name == "ar_engine":
            stage.factory = "tests.unit_test.moss_speech.ar_stub.create_ar_stub_executor"
            stage.factory_args = {
                "audio_grid_file": args.audio_grid,
                "text_grid_file": args.text_grid,
            }
    runner = MultiProcessPipelineRunner(config)
    out: dict = {}
    await runner.start(timeout=600.0)
    try:
        # ---- text request -----------------------------------------------------
        text_req = _chat_omni(messages=[{"role": "user", "content": "Introduce yourself in one sentence."}])
        result = await asyncio.wait_for(runner.coordinator.submit("smoke-t1", text_req), timeout=180)
        data = result.data if hasattr(result, "data") else result
        expected_text = Path(args.text_expected).read_text()
        out["text"] = {
            "generated_matches_fixture": data.get("generated_text") == expected_text,
            "audio_absent": not data.get("audio_samples"),
            "head": (data.get("generated_text") or "")[:40],
        }

        # ---- audio request (seed=0; P1 parity) ---------------------------------
        voice_b64 = base64.b64encode(Path(args.voice_wav).read_bytes()).decode()
        audio_req = _chat_omni(
            modalities=["audio"],
            seed=0,
            messages=[{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": voice_b64, "format": "wav"}}
            ]}],
        )
        result = await asyncio.wait_for(runner.coordinator.submit("smoke-a1", audio_req), timeout=600)
        data = result.data if hasattr(result, "data") else result
        ref = json.loads(Path(args.reference_json).read_text())
        out["audio"] = {
            "sr": data.get("audio_sample_rate"),
            "n_samples": len(data.get("audio_samples") or []),
            "hash_equals_p1": _blake(data.get("audio_samples")) == ref["decode"]["mixed_cn_s0"]["blake2b"],
            "text_empty": not data.get("generated_text"),
        }

        # ---- abort then recover --------------------------------------------------
        long_req = _chat_omni(messages=[{"role": "user", "content": "another"}])
        submit_task = asyncio.create_task(runner.coordinator.submit("smoke-x1", long_req))
        await asyncio.sleep(0.05)
        await runner.coordinator.abort("smoke-x1")
        try:
            await asyncio.wait_for(submit_task, timeout=60)
        except asyncio.TimeoutError:
            out["abort"] = {"submit_finalized": False}
        result2 = await asyncio.wait_for(
            runner.coordinator.submit("smoke-t2", _chat_omni(messages=[{"role": "user", "content": "again"}])),
            timeout=240,
        )
        data2 = result2.data if hasattr(result2, "data") else result2
        out["abort"] = {
            "submit_finalized": True,
            "next_request_recovered": bool(data2.get("generated_text")),
        }
    finally:
        await runner.stop()
    out["children_exited"] = all(p.returncode is not None for g in runner._groups for p in g.processes)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio-grid", required=True)
    parser.add_argument("--text-grid", required=True)
    parser.add_argument("--text-expected", required=True)
    parser.add_argument("--voice-wav", required=True)
    parser.add_argument("--reference-json", required=True, help="P1 alignment reference_export.json")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    out = asyncio.run(main_async(args))
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    assert out["text"]["generated_matches_fixture"] and out["text"]["audio_absent"]
    assert out["audio"]["hash_equals_p1"] and out["audio"]["sr"] == 24000
    assert out["abort"]["next_request_recovered"]
    assert out["children_exited"]
    print("MULTIPROC SMOKE PASSED")


if __name__ == "__main__":
    main()
