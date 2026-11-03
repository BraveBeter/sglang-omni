# SPDX-License-Identifier: Apache-2.0
"""Self-contained canonical inputs and audio for MOSS-Speech contracts."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.models.moss_speech.payload_types import (
    CanonicalTurn,
    MossSpeechState,
    OutputModality,
)
from sglang_omni.models.moss_speech.request_builders import (
    normalize_and_validate,
    terminal_result,
)
from sglang_omni.models.moss_speech.request_data import (
    MossSpeechSGLangRequestData,
    build_sglang_moss_request,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


class CanonicalCase(BaseModel):
    output_modality: OutputModality
    turns: list[CanonicalTurn]
    input_ids: list[list[int]]
    attention_mask: list[int]
    audio_codes: list[list[int]]


class CanonicalFixture(BaseModel):
    text_encodings: dict[str, list[int]]
    cases: dict[str, CanonicalCase]


def canonical_fixture() -> CanonicalFixture:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "fixtures/moss_speech/canonical_inputs.json"
    )
    return CanonicalFixture.model_validate_json(fixture_path.read_text())


class FixtureTokenizer:
    """Use frozen tokenization for golden text and UTF-8 bytes for other text."""

    pad_token_id: int = 151643

    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings: dict[str, list[int]] = encodings

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {
            "input_ids": list(
                self.encodings.get(text, [1000 + byte for byte in text.encode("utf-8")])
            )
        }


def native_request(
    request: OmniRequest, request_id: str = "request"
) -> tuple[MossSpeechState, MossSpeechSGLangRequestData]:
    state = normalize_and_validate(request, request_id=request_id, allow_streaming=True)
    state.input_grid = [[151644, 512]]
    state = MossSpeechState.from_dict(state.to_dict())
    payload = StagePayload(request_id, request, state.to_dict())
    return state, build_sglang_moss_request(state, payload=payload)


class NativeRequestClient(Client):
    """Replace coordinator execution while retaining real Client result aggregation."""

    def __init__(self) -> None:
        self.native: MossSpeechSGLangRequestData | None = None

    async def generate(
        self, request: GenerateRequest, request_id: str | None = None
    ) -> AsyncIterator[GenerateChunk]:
        request_id = request_id or "request"
        state, self.native = native_request(
            Client.build_omni_request(request), request_id
        )
        state.generated_text = "accepted"
        state.finish_reason = "stop"
        if state.output_modality == "audio":
            state.audio_samples = [0.0, 0.1, -0.1] * 80
            state.audio_sample_rate = 24000
        else:
            pass
        yield Client.default_result_builder(request_id, terminal_result(state))


def wav_bytes(
    sample_rate: int = 16000,
    sample_count: int = 1600,
    *,
    nonfinite: bool = False,
) -> bytes:
    time_seconds = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = 0.1 * np.sin(2 * np.pi * 220 * time_seconds)
    if nonfinite:
        waveform[sample_count // 2] = np.nan
    else:
        pass
    buffer = io.BytesIO()
    sf.write(
        buffer,
        waveform,
        sample_rate,
        format="WAV",
        subtype="FLOAT" if nonfinite else "PCM_16",
    )
    return buffer.getvalue()
