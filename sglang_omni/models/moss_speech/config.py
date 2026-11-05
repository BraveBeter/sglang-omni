# SPDX-License-Identifier: Apache-2.0
"""Eager MOSS-Speech pipeline configuration."""

from typing import ClassVar, Literal

from pydantic import ConfigDict, Field, StrictBool, model_validator

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.moss_speech.components.flow_precision import (
    FlowPrecisionProfile,
)
from sglang_omni.models.moss_speech.hf_config import (
    KV_RESERVE_TOKENS,
    MAX_CONTEXT_TOKENS,
)

PACKAGE = "sglang_omni.models.moss_speech"


class MossSpeechPreprocessingArgs(FactoryArgs):
    model_config = ConfigDict(extra="forbid")

    codec_path: str | None = None
    voice_wav: str | None = None
    encode_batch_size: int = Field(default=4, ge=1)
    context_limit: int = Field(default=MAX_CONTEXT_TOKENS, ge=1, le=MAX_CONTEXT_TOKENS)


class MossSpeechPreprocessingStage(StageConfig):
    factory: MossSpeechPreprocessingArgs = Field(
        default_factory=MossSpeechPreprocessingArgs
    )


class MossSpeechEngineArgs(FactoryArgs):
    model_config = ConfigDict(extra="forbid")

    decode_rope_reuse: bool = False
    dtype: str = "bfloat16"
    context_length: int = Field(default=MAX_CONTEXT_TOKENS, ge=1, le=MAX_CONTEXT_TOKENS)
    native_norm_library: str | None = None
    decode_graph: StrictBool = False
    decode_graph_serial_batch: StrictBool = False

    @model_validator(mode="after")
    def validate_serial_graph(self) -> "MossSpeechEngineArgs":
        if self.decode_graph_serial_batch and not self.decode_graph:
            raise ValueError("decode_graph_serial_batch requires decode_graph")
        else:
            return self


class MossSpeechEngineStage(EngineStageConfig):
    factory: MossSpeechEngineArgs = Field(default_factory=MossSpeechEngineArgs)


class MossSpeechVocoderArgs(FactoryArgs):
    model_config = ConfigDict(extra="forbid")

    codec_path: str | None = None
    chunk_size: Literal[5, 25] = 25
    broadcast_attention_bias: StrictBool = False
    cfm_graph_mel_length: int | None = Field(
        default=None, ge=100, le=4000, multiple_of=100
    )
    flow_precision: FlowPrecisionProfile | None = None

    @model_validator(mode="after")
    def validate_flow_precision(self) -> "MossSpeechVocoderArgs":
        if self.flow_precision is not None and self.cfm_graph_mel_length is not None:
            raise ValueError("BF16 Flow precision excludes the FP32 CFM graph")
        else:
            return self


class MossSpeechVocoderStage(StageConfig):
    factory: MossSpeechVocoderArgs = Field(default_factory=MossSpeechVocoderArgs)


class MossSpeechPipelineConfig(PipelineConfig):
    """Four modalities with text and audio terminals for both response modes."""

    architecture: ClassVar[str] = "MossSpeechForCausalLM"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "preprocessing": MossSpeechPreprocessingStage,
        "ar_engine": MossSpeechEngineStage,
        "audio_vocoder": MossSpeechVocoderStage,
    }

    terminal_stages_fn: str | None = (
        f"{PACKAGE}.request_builders.resolve_terminal_stages"
    )
    entry_stage: str | None = "preprocessing"
    stages: list[StageConfig] = [
        MossSpeechPreprocessingStage(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            gpu=0,
            gpu_memory_fraction=0.06,
            next="ar_engine",
        ),
        MossSpeechEngineStage(
            name="ar_engine",
            process="ar_engine",
            factory_path=f"{PACKAGE}.stages.create_ar_engine_executor",
            engine=EngineArgs(
                max_running_requests=4,
                max_queued_requests=20,
                max_total_tokens=MAX_CONTEXT_TOKENS + KV_RESERVE_TOKENS,
                disable_cuda_graph=True,
            ),
            gpu=0,
            gpu_memory_fraction=0.72,
            next=["text_decode", "audio_vocoder"],
            route_fn=f"{PACKAGE}.request_builders.resolve_output_terminal",
            stream_to=["text_decode", "audio_vocoder"],
            stream_done_to_fn=f"{PACKAGE}.streaming.resolve_stream_terminal",
        ),
        StageConfig(
            name="text_decode",
            process="text_decode",
            factory_path=f"{PACKAGE}.stages.create_text_decode_executor",
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        MossSpeechVocoderStage(
            name="audio_vocoder",
            process="audio_vocoder",
            factory_path=f"{PACKAGE}.stages.create_audio_vocoder_executor",
            gpu=0,
            gpu_memory_fraction=0.12,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


EntryClass = MossSpeechPipelineConfig
