# SPDX-License-Identifier: Apache-2.0
"""Factories for text preprocessing, native AR generation and offline audio."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import torch
from transformers import AutoTokenizer

from sglang_omni.models.moss_speech.codec import MossSpeechCodec
from sglang_omni.models.moss_speech.concurrent_codec import MossSpeechConcurrentCodec
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
from sglang_omni.models.moss_speech.hf_config import (
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechSGLangRequestData,
    preprocess_request,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device


def create_preprocessing_executor(
    model_path: str,
    *,
    codec_path: str | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    ensure_moss_speech_config_registered()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    codec = (
        MossSpeechCodec(
            codec_path, resolve_concrete_device(device, gpu_id), load_decoder=False
        )
        if codec_path is not None
        else None
    )
    return SimpleScheduler(
        partial(preprocess_request, tokenizer=tokenizer, audio_encoder=codec),
        shutdown_callback=codec.close if codec is not None else None,
    )


def create_ar_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
    cache_repetition_history: bool = False,
) -> OmniScheduler[MossSpeechSGLangRequestData]:
    if not isinstance(cache_repetition_history, bool):
        raise ValueError("cache_repetition_history must be a boolean")
    else:
        pass
    scheduler = MossSpeechEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype="bfloat16",
        server_args_overrides=server_args_overrides,
        total_gpu_memory_fraction=(
            process_total_gpu_memory_fraction
            if process_total_gpu_memory_fraction is not None
            else total_gpu_memory_fraction
        ),
    )
    if cache_repetition_history:
        assert isinstance(scheduler.model_runner, MossSpeechModelRunner)
        scheduler.model_runner.cache_repetition_history = True
    else:
        pass
    return scheduler


def create_audio_vocoder_executor(
    model_path: str,
    *,
    codec_path: str | None,
    flow_dtype: Literal["float32", "bfloat16"],
    flow_slots: Literal[1, 2] = 1,
    terminal_flow_steps: int | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    if codec_path is None:
        raise ValueError("Set audio_vocoder.factory.codec_path")
    else:
        pass
    codec_class = MossSpeechConcurrentCodec if flow_slots == 2 else MossSpeechCodec
    if terminal_flow_steps is not None and terminal_flow_steps < 1:
        raise ValueError("terminal_flow_steps must be positive")
    else:
        pass
    step_options = (
        {"terminal_flow_steps": terminal_flow_steps}
        if terminal_flow_steps is not None
        else {}
    )
    codec = codec_class(
        codec_path,
        resolve_concrete_device(device, gpu_id),
        flow_dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[flow_dtype],
        **step_options,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    return SimpleScheduler(
        partial(codec.decode, tokenizer=tokenizer),
        max_concurrency=flow_slots,
        shutdown_callback=codec.close,
    )
