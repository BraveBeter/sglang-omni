# SPDX-License: Apache-2.0
"""Native MOSS-Speech pipeline: preprocessing -> AR -> text/audio terminal.

P1 codec components are shared by preprocessing and vocoder. The AR stage
uses the same native builder validated by the Phase 3 parity drivers.
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    StageResourceConfig,
    StageRuntimeConfig,
)

_PKG = "sglang_omni.models.moss_speech"


class MossSpeechPipelineConfig(PipelineConfig):
    """4-stage chat-native pipeline: preprocessing -> ar_engine -> dual terminals."""

    architecture: ClassVar[str] = "MossSpeechForCausalLM"
    # Per-request terminal join: V1 routes text/audio to exactly one terminal
    # (qwen3_omni precedent; without this the coordinator joins BOTH terminals
    # and single-modality requests hang).
    terminal_stages_fn: str | None = f"{_PKG}.request_builders.resolve_terminal_stages"

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
    entry_stage: str | None = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="preproc",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"encode_batch_size": 4},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.06)
            ),
            next="ar_engine",
        ),
        StageConfig(
            name="ar_engine",
            process="ar",
            factory=f"{_PKG}.stages.create_ar_engine_executor",
            factory_args={"dtype": "bfloat16"},
            gpu=0,
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.72)
            ),
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
            runtime=StageRuntimeConfig(
                resources=StageResourceConfig(total_gpu_memory_fraction=0.12)
            ),
            terminal=True,
        ),
    ]


EntryClass = MossSpeechPipelineConfig
