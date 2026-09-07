# SPDX-License: Apache-2.0
"""Typed per-request state for the MOSS-Speech pipeline (P2).

All cross-stage fields use ``wire`` codecs so the state serializes through
the framework's process transport. Tensors that cross processes are CPU
float/int tensors materialized only at the serialization boundary
(AGENT.md §4.3); device placement happens in the consuming stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

# Reference token/grid constants (MODEL_CONTRACT §3).
SOSP_TOKEN_ID = 151646
EOSP_TOKEN_ID = 16384
MODALITY_PAD_TOKEN_ID = 151667
AUDIO_PAD_TOKEN_ID = 512
TEXT_VOCAB = 151680
AUDIO_VOCAB = 16512


@dataclass
class MossSpeechState(DeclarativeStateBase):
    """Per-request state flowing preprocessing -> ar_engine -> terminals."""

    # ---- canonical request (filled by request_builders.normalize) --------
    output_modality: str = wire("text", codec="str")  # "text" | "audio"
    # conversation turns as wire-safe dicts:
    #   {"role": str, "kind": "text"|"audio", "text": str|None,
    #    "audio": {"waveform": list[float], "sr": int, "source_key": str}|None}
    turns: list[dict[str, Any]] = wire(default_factory=list, codec="list")

    # ---- parameter fidelity (P2 chat contract §3.1/§5) --------------------
    explicit_params: list[str] = wire(default_factory=list, codec="list")
    effective_seed: int = wire(0, codec="int")
    # raw sampling values as provided (endpoint fill-ins already applied by
    # the HTTP layer); the AR builder distinguishes explicit via explicit_params
    temperature: Optional[float] = wire(None, codec="float")
    top_p: Optional[float] = wire(None, codec="float")
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = wire(None, codec="float")
    max_new_tokens: Optional[int] = None
    stop: list[str] = wire(default_factory=list, codec="list")

    # ---- per-turn codec codes (preprocessing encode) ----------------------
    # one list[int] per audio turn, same order as `turns`
    audio_codes: list[list[int]] = wire(default_factory=list, codec="list")

    # ---- canonical grid (processor output, before AR) ----------------------
    input_grid: list[list[list[int]]] = wire(default_factory=list, codec="list")  # (B, L, 2)
    attention_mask: list[list[int]] = wire(default_factory=list, codec="list")
    prompt_grid_len: int = wire(0, codec="int")

    # ---- voice conditioning transport (audio requests only) ----------------
    # Immutable precomputed default voice, transported with the request
    # (P2 contract §7.1); vocoder may cache device copies keyed by voice_key.
    voice_key: Optional[str] = wire(None, codec="str_or")
    voice_prompt_token: Any = None  # tensor, set by preprocessing (tensor list)
    voice_prompt_feat: Any = None
    voice_embedding: Any = None

    # ---- AR output (stub or native) ----------------------------------------
    output_grid: list[list[list[int]]] = wire(default_factory=list, codec="list")  # (B, L_new, 2)
    generated_text: str = wire("", codec="str")

    # ---- terminal outputs ----------------------------------------------------
    audio_samples: list[float] = wire(default_factory=list, codec="list")
    audio_sample_rate: int = wire(24000, codec="int")
