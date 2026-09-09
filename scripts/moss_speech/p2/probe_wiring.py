#!/usr/bin/env python3
"""Local CPU probe: isolate where smoke-t1 stalls in the real runner wiring.

Chains a MINIMAL preprocessing (normalize + tiny grid, no codec/GPU) ->
test AR stub -> text_decode, using the real MultiProcessPipelineRunner,
real config shape and real route_fn. If this flows, the stall is inside the
heavy preprocessing factory; if not, it is stage wiring.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from sglang_omni.config import StageConfig
from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest

_PKG = "sglang_omni.models.moss_speech"
_TINY = "tests.unit_test.moss_speech.probe_fakes"


async def main() -> None:
    config = MossSpeechPipelineConfig(
        model_path="/tmp/probe-model",
        endpoints={"base_path": "/dev/shm/p2/probe"},
        stages=[
            StageConfig(
                name="preprocessing",
                process="preproc",
                factory=f"{_TINY}.create_probe_preprocessing",
                next="ar_engine",
            ),
            StageConfig(
                name="ar_engine",
                process="ar",
                factory=f"{_TINY}.create_probe_ar",
                next=["text_decode", "audio_vocoder"],
                route_fn=f"{_PKG}.request_builders.resolve_output_terminal",
            ),
            StageConfig(
                name="text_decode",
                process="text_out",
                factory=f"{_TINY}.create_probe_text_terminal",
                terminal=True,
            ),
            StageConfig(
                name="audio_vocoder",
                process="vocoder",
                factory=f"{_TINY}.create_probe_text_terminal",
                terminal=True,
            ),
        ],
    )
    runner = MultiProcessPipelineRunner(config)
    await runner.start(timeout=120.0)
    try:
        gen = _build_chat_generate_request(
            ChatCompletionRequest(
                model="m", messages=[{"role": "user", "content": "hello probe"}]
            )
        )
        result = await asyncio.wait_for(
            runner.coordinator.submit("probe-1", OmniRequest(inputs=gen.to_dict())),
            timeout=60,
        )
        data = result.data if hasattr(result, "data") else result
        print("PROBE RESULT:", data if isinstance(data, dict) else type(data))
        print(
            "PROBE OK"
            if isinstance(data, dict) and data.get("probe") == "done"
            else "PROBE UNEXPECTED"
        )
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
