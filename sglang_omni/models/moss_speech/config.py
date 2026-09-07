# SPDX-License: Apache-2.0
"""Pipeline configuration for MOSS-Speech (P2 skeleton).

Topology (P1 placement decision D-1/D-3, P2 chat contract §7):

    preprocessing (validate + lower + codec encode, Layout A)
        -> ar_engine (native SGLang AR; P3 — factory raises a tagged
           not-implemented error until then)
        -> route_fn -> text_decode (terminal, CPU)
                    |-> audio_vocoder (terminal, GPU, serial + RNG scope)

Preprocessing/vocoder consume the P1 codec adapter exclusively; no codec
logic is duplicated here. The formal ``ar_engine`` factory performs its
argument and hf_config checks and then raises
``MossSpeechARNotImplemented`` (P3 pointer) — see stages.py.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.moss_speech"


class MossSpeechPipelineConfig(PipelineConfig):
    """4-stage chat-native pipeline: preprocessing -> ar_engine -> dual terminals."""

    architecture: ClassVar[str] = "MossSpeechForCausalLM"

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "ar_engine"}

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "ar_engine"}

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # colocated edges (same process) — none in V1: three GPU stages live in
        # separate processes per the P1 placement decision D-3.
        return frozenset()

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="preproc",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"encode_batch_size": 4},
            gpu=0,
            next="ar_engine",
        ),
        StageConfig(
            name="ar_engine",
            process="ar",
            factory=f"{_PKG}.stages.create_ar_engine_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            next=["text_decode", "audio_vocoder"],
            route_fn=f"{_PKG}.request_builders.resolve_output_terminal",
        ),
        StageConfig(
            name="text_decode",
            process="text_out",
            factory=f"{_PKG}.stages.create_text_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name="audio_vocoder",
            process="vocoder",
            factory=f"{_PKG}.stages.create_audio_vocoder_executor",
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = MossSpeechPipelineConfig
