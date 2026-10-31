# SPDX-License-Identifier: Apache-2.0
"""Normalize requests and adapt canonical grids to the eager AR scheduler."""

from __future__ import annotations

import hashlib
import math
from dataclasses import fields
from typing import Protocol

import torch
from pydantic import TypeAdapter, ValidationError

from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.models.moss_speech.audio_tasks import (
    RequestValidationError,
    lower_audio_task,
    normalize_turns,
    restore_audio_task_request,
)
from sglang_omni.models.moss_speech.hf_config import MAX_CONTEXT_TOKENS
from sglang_omni.models.moss_speech.payload_types import (
    EOSP_TOKEN_ID,
    MODALITY_PAD_TOKEN_ID,
    MossSpeechState,
)
from sglang_omni.models.moss_speech.request_data import (
    MossSpeechSGLangRequestData,
    build_sglang_moss_request,
)
from sglang_omni.models.moss_speech.request_lifecycle import MossSpeechRequestLifecycle
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.types import DeferredAdmission

TEXT_TERMINAL = "text_decode"
AUDIO_TERMINAL = "audio_vocoder"


class RequestBuilder(Protocol):
    def __call__(
        self, payload: StagePayload
    ) -> (
        MossSpeechSGLangRequestData | DeferredAdmission[MossSpeechSGLangRequestData]
    ): ...


class ResultAdapter(Protocol):
    def __call__(self, request_data: MossSpeechSGLangRequestData) -> StagePayload: ...


def as_generate_request(inputs: object) -> GenerateRequest:
    if isinstance(inputs, GenerateRequest):
        return inputs
    elif not isinstance(inputs, dict):
        raise RequestValidationError(
            "requires a GenerateRequest or a serialized request object"
        )
    elif set(inputs) - {parameter.name for parameter in fields(GenerateRequest)}:
        raise RequestValidationError("serialized request contains unsupported fields")
    else:
        try:
            return TypeAdapter(GenerateRequest).validate_python(inputs)
        except ValidationError as error:
            raise RequestValidationError(str(error)) from error


def request_generate(request: OmniRequest) -> GenerateRequest:
    inputs = request.inputs
    if (
        isinstance(inputs, GenerateRequest)
        or isinstance(inputs, dict)
        and "sampling" in inputs
    ):
        return as_generate_request(inputs)
    elif request.metadata.get("task") in ("tts", "asr") and (
        isinstance(inputs, str) or isinstance(inputs, dict) and "audio_bytes" in inputs
    ):
        return restore_audio_task_request(request)
    elif isinstance(inputs, dict) and not request.params and not request.metadata:
        return as_generate_request(inputs)
    else:
        pass
    metadata = dict(request.metadata)
    if isinstance(inputs, dict):
        messages = inputs.get("messages")
        metadata.update(
            {
                name: inputs[name]
                for name in ("audios", "images", "videos")
                if name in inputs
            }
        )
    else:
        messages = inputs
    if not isinstance(messages, list):
        raise RequestValidationError("requires chat messages")
    else:
        pass
    sampling_names = {parameter.name for parameter in fields(SamplingParams)}
    parameters = request.params
    unsupported = (
        set(parameters) - sampling_names - {"stream", "stage_sampling", "stage_params"}
    )
    if unsupported:
        raise RequestValidationError(
            f"unsupported request parameters: {sorted(unsupported)}"
        )
    else:
        pass
    return as_generate_request(
        {
            "messages": messages,
            "metadata": metadata,
            "sampling": {
                name: value
                for name, value in parameters.items()
                if name in sampling_names
            },
            "output_modalities": metadata.get("output_modalities", ["text"]),
            "stream": parameters.get("stream", False),
            "max_tokens": parameters.get("max_new_tokens"),
            "stage_sampling": parameters.get("stage_sampling"),
            "stage_params": parameters.get("stage_params"),
        }
    )


def normalize_and_validate(
    request: OmniRequest, *, request_id: str, allow_streaming: bool = False
) -> MossSpeechState:
    generation = lower_audio_task(request_generate(request))
    stream = request.params.get("stream", generation.stream)
    if not isinstance(stream, bool):
        raise RequestValidationError("stream must be a boolean")
    else:
        pass
    if (
        generation.prompt is not None
        or generation.prompt_token_ids is not None
        or generation.multimodal_train_inputs is not None
    ):
        raise RequestValidationError(
            "requires chat messages instead of prompt or token inputs"
        )
    else:
        pass
    sampling = generation.sampling
    for name in ("temperature", "top_p", "repetition_penalty", "min_p"):
        value = getattr(sampling, name)
        if value is not None and not math.isfinite(value):
            raise RequestValidationError(f"{name} must be finite")
        else:
            pass
    if (
        sampling.temperature is not None
        and sampling.temperature < 0
        or sampling.top_p is not None
        and not 0 < sampling.top_p <= 1
        or sampling.repetition_penalty is not None
        and sampling.repetition_penalty <= 0
        or sampling.top_k is not None
        and sampling.top_k < -1
    ):
        raise RequestValidationError("invalid sampling parameters")
    elif sampling.min_p not in (None, 0.0):
        raise RequestValidationError("min_p is unsupported", reason="not_supported")
    elif sampling.stop or sampling.stop_token_ids:
        raise RequestValidationError(
            "custom stop strings and token IDs are unsupported", reason="not_supported"
        )
    elif sampling.seed is not None and not -(2**63) <= sampling.seed < 2**64:
        raise RequestValidationError("seed must fit torch.Generator's 64-bit range")
    else:
        pass
    if (
        generation.max_tokens is not None
        and sampling.max_new_tokens is not None
        and generation.max_tokens != sampling.max_new_tokens
    ):
        raise RequestValidationError("max_tokens and max_new_tokens conflict")
    else:
        pass
    max_new_tokens = (
        generation.max_tokens
        if generation.max_tokens is not None
        else sampling.max_new_tokens
    )
    if max_new_tokens is not None and not 1 <= max_new_tokens < MAX_CONTEXT_TOKENS:
        raise RequestValidationError(
            f"max_new_tokens must be within [1, {MAX_CONTEXT_TOKENS - 1}]"
        )
    elif stream and not allow_streaming:
        raise RequestValidationError(
            "streaming is disabled for this pipeline", reason="not_supported"
        )
    elif (
        generation.stage_sampling or generation.stage_params or generation.extra_params
    ):
        raise RequestValidationError(
            "stage overrides and extra params are unsupported", reason="not_supported"
        )
    else:
        pass
    metadata = generation.metadata
    audio_config = metadata.get("audio_config")
    if audio_config and not (
        allow_streaming
        and stream
        and isinstance(audio_config, dict)
        and set(audio_config) == {"format"}
        and audio_config["format"] in ("wav", "pcm")
    ):
        raise RequestValidationError(
            "request-level voice/audio configuration is unsupported",
            reason="not_supported",
        )
    elif metadata.get("images") or metadata.get("videos"):
        raise RequestValidationError(
            "image/video inputs are unsupported", reason="not_supported"
        )
    else:
        pass
    modalities = generation.output_modalities or ["text"]
    if modalities not in (["text"], ["audio"]):
        raise RequestValidationError("modalities must be exactly ['text'] or ['audio']")
    else:
        pass
    turns, decoded_audio = normalize_turns(generation.messages or [], metadata)
    seed = sampling.seed
    if seed is None:
        digest = hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "big") & 0x7FFFFFFF
    else:
        pass
    state = MossSpeechState(
        output_modality=modalities[0],
        turns=turns,
        effective_seed=seed,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        repetition_penalty=sampling.repetition_penalty,
        max_new_tokens=max_new_tokens,
    )
    state.decoded_audio = decoded_audio
    request.params["stream"] = stream
    return state


def pop_decoded_audio(state: MossSpeechState) -> list[tuple[torch.Tensor, int, str]]:
    decoded_audio = state.decoded_audio
    state.decoded_audio = []
    return decoded_audio


def state_from_payload(payload: StagePayload) -> MossSpeechState:
    if isinstance(payload.data, MossSpeechState):
        return payload.data
    elif isinstance(payload.data, dict):
        return MossSpeechState.from_dict(payload.data)
    else:
        raise TypeError(f"unsupported stage data {type(payload.data)!r}")


def resolve_terminal_stages(request: OmniRequest) -> list[str]:
    generation = lower_audio_task(request_generate(request))
    modalities = generation.output_modalities or ["text"]
    if modalities == ["audio"]:
        return [AUDIO_TERMINAL]
    elif modalities == ["text"]:
        return [TEXT_TERMINAL]
    else:
        raise RequestValidationError(
            f"cannot resolve terminal stages for modalities {modalities!r}"
        )


def resolve_output_terminal(request_id: str, output: StagePayload) -> str:
    state = state_from_payload(output)
    if state.output_modality == "text":
        return TEXT_TERMINAL
    elif state.output_modality == "audio":
        return AUDIO_TERMINAL
    else:
        raise RequestValidationError(
            f"request {request_id}: invalid output_modality {state.output_modality!r}"
        )


def terminal_result(state: MossSpeechState) -> dict[str, object]:
    result = state.to_dict()
    result["text"] = state.generated_text
    result["usage"] = {
        "prompt_tokens": state.prompt_grid_len,
        "completion_tokens": len(state.output_grid),
        "total_tokens": state.prompt_grid_len + len(state.output_grid),
    }
    if state.output_modality == "audio":
        result["audio_data"] = state.audio_samples
        result["sample_rate"] = state.audio_sample_rate
    else:
        pass
    return result


def extract_output_codes(state: MossSpeechState) -> list[int]:
    codes: list[int] = []
    for text_token, audio_token in state.output_grid:
        if text_token != MODALITY_PAD_TOKEN_ID:
            continue
        elif audio_token == EOSP_TOKEN_ID:
            break
        elif not 0 <= audio_token < EOSP_TOKEN_ID:
            raise RequestValidationError(f"Invalid generated audio code {audio_token}")
        else:
            codes.append(audio_token)
    return codes


def make_moss_speech_scheduler_adapters(
    *, lifecycle: MossSpeechRequestLifecycle
) -> tuple[RequestBuilder, ResultAdapter]:
    def request_builder(
        payload: StagePayload,
    ) -> MossSpeechSGLangRequestData | DeferredAdmission[MossSpeechSGLangRequestData]:
        state = state_from_payload(payload)
        request_data = build_sglang_moss_request(state, payload=payload)
        return lifecycle.admit(request_data, state.output_modality)

    def result_adapter(request_data: MossSpeechSGLangRequestData) -> StagePayload:
        assert request_data.req is not None
        try:
            payload = request_data.stage_payload
            assert payload is not None
            state = state_from_payload(payload)
            state.output_grid = [list(row) for row in request_data.output_rows]
            state.finish_reason = request_data.finish_reason or ""
            return StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data=state.to_dict(),
            )
        finally:
            lifecycle.release(request_data.req.rid)

    return request_builder, result_adapter
