# SPDX-License: Apache-2.0
"""Native MOSS-Speech pipeline: preprocessing -> AR -> text/audio terminal.

P1 codec components are shared by preprocessing and vocoder. The AR stage
uses the same native builder validated by the Phase 3 parity drivers.
"""

from __future__ import annotations

from typing import Any, ClassVar

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

    @staticmethod
    def validate_chat_request(request: Any) -> None:
        """Reject unsupported chat requests before HTTP streaming headers."""
        from sglang_omni.proto.request import OmniRequest
        from sglang_omni.serve.openai_api import _build_chat_generate_request

        from .request_builders import normalize_and_validate

        lengths = [
            value
            for value in (request.max_tokens, request.max_completion_tokens)
            if value is not None
        ]
        if any(not 1 <= value <= 512 for value in lengths):
            raise ValueError("max_tokens/max_completion_tokens must be within [1, 512]")
        if len(set(lengths)) > 1:
            raise ValueError("max_tokens and max_completion_tokens conflict")
        normalize_and_validate(
            OmniRequest(inputs=_build_chat_generate_request(request)),
            request_id="preflight",
        )

    @classmethod
    def generation_admission_defaults(cls) -> dict[str, Any]:
        return {"max_running_requests": 4, "max_queued_requests": 20}

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
            factory_args={
                "dtype": "bfloat16",
                "server_args_overrides": {
                    "max_running_requests": 4,
                    "max_queued_requests": 20,
                    "max_total_tokens": 4096,
                },
            },
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
