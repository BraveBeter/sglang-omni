# SPDX-License-Identifier: Apache-2.0
"""Typed request state transported between MOSS-Speech stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import torch

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

SOSP_TOKEN_ID = 151646
EOSP_TOKEN_ID = 16384
MODALITY_PAD_TOKEN_ID = 151667
AUDIO_PAD_TOKEN_ID = 512
TEXT_VOCAB = 151680
AUDIO_VOCAB = 16512

OutputModality = Literal["text", "audio"]
VoicePayload = dict[str, bytes | list[int] | str]


class CanonicalTurn(TypedDict):
    role: Literal["system", "user", "assistant"]
    kind: Literal["text", "audio"]
    text: str | None


@dataclass
class MossSpeechState(DeclarativeStateBase):
    """Canonical input, sampling settings and model output for one request."""

    output_modality: OutputModality = wire("text", codec="str")
    turns: list[CanonicalTurn] = wire(default_factory=list, codec="list")
    effective_seed: int = wire(0, codec="int")
    temperature: float | None = wire(None, codec="float")
    top_p: float | None = wire(None, codec="float")
    top_k: int | None = wire(None, codec="opt_int")
    repetition_penalty: float | None = wire(None, codec="float")
    max_new_tokens: int | None = wire(None, codec="opt_int")
    stop: list[str] = wire(default_factory=list, codec="list")
    audio_codes: list[list[int]] = wire(default_factory=list, codec="list")
    input_grid: list[list[int]] = wire(default_factory=list, codec="list")
    attention_mask: list[int] = wire(default_factory=list, codec="list")
    prompt_grid_len: int = wire(0, codec="int")
    voice_key: str | None = wire(None, codec="str_or")
    voice_payload: VoicePayload | None = wire(None, codec="raw")
    output_grid: list[list[int]] = wire(default_factory=list, codec="list")
    generated_text: str = wire("", codec="str")
    audio_samples: list[float] = wire(default_factory=list, codec="list")
    audio_sample_rate: int = wire(24000, codec="int")

    def __post_init__(self) -> None:
        # note (xrluan): Decoded waveforms are consumed before crossing the stage boundary.
        self.decoded_audio: list[tuple[torch.Tensor, int, str]] = []
