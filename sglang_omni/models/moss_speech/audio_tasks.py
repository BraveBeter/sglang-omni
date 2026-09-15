# SPDX-License-Identifier: Apache-2.0
"""Lower the shared speech endpoint protocol into canonical model chat input."""
from __future__ import annotations

import base64
from dataclasses import fields, replace
from typing import Any

from sglang_omni.client import GenerateRequest, Message, SamplingParams
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY

# Use the frozen qualification prompts; endpoint support does not redefine
# the model's speech or transcription quality profile.
TTS_SYSTEM_PROMPT = "You are a helpful voice assistant. Answer the user's questions with spoken responses."
TTS_INPUT_PREFIX = "Repeat the following text verbatim, without adding anything: "
ASR_SYSTEM_PROMPT = "Transcribe the user's audio verbatim in its original language. Output only the transcription."


def restore_audio_task_request(request: Any) -> GenerateRequest:
    """Restore the shared Client's split wire form without losing explicitness."""
    params = dict(request.params or {})
    keys = {field.name for field in fields(SamplingParams)}
    sampling = SamplingParams(
        **{name: value for name, value in params.items() if name in keys}
    )
    return GenerateRequest(
        prompt=request.inputs,
        sampling=sampling,
        stream=params.pop("stream", False),
        max_tokens=params.get("max_new_tokens"),
        stage_sampling=params.pop("stage_sampling", None),
        stage_params=params.pop("stage_params", None),
        extra_params={
            name: value for name, value in params.items() if name not in keys
        },
        output_modalities=(request.metadata or {}).get("output_modalities"),
        metadata=dict(request.metadata or {}),
    )


def lower_audio_task(request: GenerateRequest) -> GenerateRequest:
    """Map qualified endpoint fields to ordinary chat; reject unhandled options."""
    from .request_builders import MAX_AUDIO_BYTES, RequestValidationError

    metadata = dict(request.metadata or {})
    task = metadata.pop("task", None)
    if task not in ("tts", "asr") or (
        request.messages is not None and request.prompt is None
    ):
        # Ordinary chat metadata is user-defined; a task label alone must not
        # replace messages with a dedicated endpoint prompt.
        return request
    if request.messages or request.prompt_token_ids is not None:
        raise RequestValidationError("Speech endpoints require a single prompt input")
    sampling = replace(request.sampling)
    extra = dict(request.extra_params or {})
    if task == "tts":
        options = dict(metadata.pop("tts_params", {}) or {})
        allowed = {
            "voice",
            "speed",
            "response_format",
            "seed",
            EXPLICIT_GENERATION_PARAMS_KEY,
        }
        if set(options) - allowed or extra:
            raise RequestValidationError(
                "MOSS-Speech does not support speech language, instructions, references or duration controls"
            )
        if (
            options.get("voice", "default") != "default"
            or options.get("speed", 1.0) != 1.0
        ):
            raise RequestValidationError(
                "MOSS-Speech requires voice='default' and speed=1"
            )
        if options.get("response_format", "wav") not in ("wav", "pcm"):
            raise RequestValidationError(
                "MOSS-Speech speech output supports wav or pcm"
            )
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise RequestValidationError("Speech input must be a non-empty string")
        metadata[EXPLICIT_GENERATION_PARAMS_KEY] = list(
            options.get(EXPLICIT_GENERATION_PARAMS_KEY, [])
        )
        messages = [
            Message(role="system", content=TTS_SYSTEM_PROMPT),
            Message(role="user", content=TTS_INPUT_PREFIX + request.prompt),
        ]
        modality = "audio"
    else:
        if extra.pop("task", "transcribe") != "transcribe" or extra:
            raise RequestValidationError(
                "MOSS-Speech supports transcription without language, prompt, translation or timestamp controls"
            )
        prompt = request.prompt
        data = prompt.get("audio_bytes") if isinstance(prompt, dict) else None
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_AUDIO_BYTES:
            raise RequestValidationError(
                "Transcription requires non-empty audio bytes within the input limit"
            )
        messages = [
            Message(role="system", content=ASR_SYSTEM_PROMPT),
            Message(
                role="user",
                content=[
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64.b64encode(data).decode("ascii")},
                    }
                ],
            ),
        ]
        explicit = list(metadata.get(EXPLICIT_GENERATION_PARAMS_KEY, []))
        if "temperature" not in explicit:
            sampling.temperature = 0.0
            explicit.append("temperature")
        metadata[EXPLICIT_GENERATION_PARAMS_KEY] = explicit
        modality = "text"
    if request.output_modalities not in (None, [modality]):
        raise RequestValidationError(
            "Speech task conflicts with requested output modality"
        )
    limit = (
        request.max_tokens
        if request.max_tokens is not None
        else sampling.max_new_tokens
    )
    return replace(
        request,
        prompt=None,
        messages=messages,
        sampling=sampling,
        extra_params={},
        metadata=metadata,
        output_modalities=[modality],
        max_tokens=limit,
    )
