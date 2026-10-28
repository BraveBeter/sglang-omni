# SPDX-License-Identifier: Apache-2.0
"""Validate encoded audio and lower speech endpoints to model chat turns."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace

import torch
from pydantic import TypeAdapter, ValidationError

from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.models.moss_speech.payload_types import CanonicalTurn
from sglang_omni.proto.request import OmniRequest
from sglang_omni.utils.audio import AudioDecodeError, decode_with_soundfile

MAX_AUDIO_DURATION_SECONDS = 30.0
MAX_TOTAL_AUDIO_DURATION_SECONDS = 120.0
MAX_AUDIO_BYTES = 50 * 1024 * 1024
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000

TTS_SYSTEM_PROMPT = "You are a helpful voice assistant. Answer the user's questions with spoken responses."
TTS_INPUT_PREFIX = "Repeat the following text verbatim, without adding anything: "
ASR_SYSTEM_PROMPT = "Transcribe the user's audio verbatim in its original language. Output only the transcription."


class RequestValidationError(ValueError):
    """A caller error carried through the shared native-chat error surface."""

    def __init__(self, message: str, *, reason: str = "invalid_request") -> None:
        super().__init__(f"Native chat request rejected: MOSS-Speech: {message}")
        self.reason = reason


def decode_audio_bytes(encoded: bytes, *, source: str) -> tuple[torch.Tensor, int]:
    if not 0 < len(encoded) <= MAX_AUDIO_BYTES:
        raise RequestValidationError(
            f"audio {source} must contain 1..{MAX_AUDIO_BYTES} bytes",
            reason="invalid_audio",
        )
    else:
        pass
    try:
        channels, sample_rate = decode_with_soundfile(encoded, source_name=source)
    except AudioDecodeError as error:
        raise RequestValidationError(
            f"audio {source} could not be decoded", reason="invalid_audio"
        ) from error
    if channels.shape[1] == 0 or not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise RequestValidationError(
            f"audio {source} requires samples at {MIN_SAMPLE_RATE}..{MAX_SAMPLE_RATE} Hz",
            reason="invalid_audio",
        )
    elif not torch.isfinite(channels).all():
        raise RequestValidationError(
            "audio contains non-finite samples", reason="invalid_audio"
        )
    elif channels.shape[1] / sample_rate > MAX_AUDIO_DURATION_SECONDS:
        raise RequestValidationError(
            f"audio duration exceeds {MAX_AUDIO_DURATION_SECONDS} seconds",
            reason="invalid_audio",
        )
    else:
        # note (xrluan): The checkpoint processor selects the first channel instead of averaging.
        return channels[0].contiguous(), sample_rate


def load_audio_source(source: str) -> tuple[torch.Tensor, int]:
    if source.startswith(("http://", "https://", "ftp://")):
        raise RequestValidationError(
            "URL audio sources are not supported; provide base64 input_audio or a local path",
            reason="not_supported",
        )
    elif source.startswith("data:"):
        encoded = source.partition(",")[2]
    elif os.path.isfile(source):
        with open(source, "rb") as audio_file:
            return decode_audio_bytes(
                audio_file.read(MAX_AUDIO_BYTES + 1), source=source
            )
    else:
        encoded = source
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise RequestValidationError(
            "invalid base64 audio", reason="invalid_audio"
        ) from error
    return decode_audio_bytes(decoded, source="input_audio")


def restore_audio_task_request(request: OmniRequest) -> GenerateRequest:
    sampling_names = {parameter.name for parameter in fields(SamplingParams)}
    parameters = request.params
    try:
        return TypeAdapter(GenerateRequest).validate_python(
            {
                "prompt": request.inputs,
                "sampling": {
                    name: value
                    for name, value in parameters.items()
                    if name in sampling_names
                },
                "stream": parameters.get("stream", False),
                "max_tokens": parameters.get("max_new_tokens"),
                "stage_sampling": parameters.get("stage_sampling"),
                "stage_params": parameters.get("stage_params"),
                "extra_params": {
                    name: value
                    for name, value in parameters.items()
                    if name
                    not in sampling_names | {"stream", "stage_sampling", "stage_params"}
                },
                "output_modalities": request.metadata.get("output_modalities"),
                "metadata": request.metadata,
            }
        )
    except ValidationError as error:
        raise RequestValidationError(str(error)) from error


def lower_audio_task(request: GenerateRequest) -> GenerateRequest:
    """Preserve chat input; lower dedicated TTS and transcription prompts."""
    metadata = dict(request.metadata)
    task = metadata.get("task")
    if (
        task not in ("tts", "asr")
        or request.messages is not None
        and request.prompt is None
    ):
        return request
    elif request.messages or request.prompt_token_ids is not None:
        raise RequestValidationError("Speech endpoints require a single prompt input")
    else:
        metadata.pop("task")
    sampling = replace(request.sampling)
    extra = dict(request.extra_params)
    if task == "tts":
        options = metadata.pop("tts_params", {})
        if (
            not isinstance(options, dict)
            or set(options) - {"voice", "speed", "response_format", "seed"}
            or extra
        ):
            raise RequestValidationError(
                "Speech language, instructions, references and duration controls are unsupported"
            )
        elif (
            options.get("voice", "default") != "default"
            or options.get("speed", 1.0) != 1.0
        ):
            raise RequestValidationError("Speech requires voice='default' and speed=1")
        elif options.get("response_format", "wav") not in ("wav", "pcm"):
            raise RequestValidationError("Speech output supports wav or pcm")
        elif not isinstance(request.prompt, str) or not request.prompt.strip():
            raise RequestValidationError("Speech input must be a non-empty string")
        else:
            pass
        option_seed = options.get("seed")
        if option_seed is not None:
            if (
                not isinstance(option_seed, int)
                or sampling.seed is not None
                and sampling.seed != option_seed
            ):
                raise RequestValidationError("Conflicting or invalid speech seed")
            else:
                sampling.seed = option_seed
        else:
            pass
        messages = [
            Message(role="system", content=TTS_SYSTEM_PROMPT),
            Message(role="user", content=TTS_INPUT_PREFIX + request.prompt),
        ]
        modality = "audio"
    else:
        if extra.pop("task", "transcribe") != "transcribe" or extra:
            raise RequestValidationError(
                "Transcription does not support language, prompt, translation or timestamp controls"
            )
        else:
            pass
        encoded = (
            request.prompt.get("audio_bytes")
            if isinstance(request.prompt, dict)
            else None
        )
        if not isinstance(encoded, bytes) or not 0 < len(encoded) <= MAX_AUDIO_BYTES:
            raise RequestValidationError(
                "Transcription requires non-empty audio bytes within the input limit"
            )
        else:
            pass
        messages = [
            Message(role="system", content=ASR_SYSTEM_PROMPT),
            Message(
                role="user",
                content=[
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(encoded).decode("ascii")
                        },
                    }
                ],
            ),
        ]
        if sampling.temperature is None:
            sampling.temperature = 0.0
        else:
            pass
        modality = "text"
    if request.output_modalities not in (None, [modality]):
        raise RequestValidationError(
            "Speech task conflicts with requested output modality"
        )
    else:
        pass
    return replace(
        request,
        prompt=None,
        messages=messages,
        sampling=sampling,
        extra_params={},
        metadata=metadata,
        output_modalities=[modality],
    )


def normalize_turns(
    messages: Sequence[Message], metadata: Mapping[str, object]
) -> tuple[list[CanonicalTurn], list[tuple[torch.Tensor, int, str]]]:
    turns: list[CanonicalTurn] = []
    audio_sources: list[str] = []
    if not messages:
        raise RequestValidationError("messages must not be empty")
    else:
        pass
    for message in messages:
        role = message.role
        if role not in ("system", "user", "assistant"):
            raise RequestValidationError(f"unsupported role {role!r}")
        else:
            pass
        content = message.content
        text_parts: list[str] = []
        audio_part: dict[str, str] | None = None
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise RequestValidationError("chat content part must be an object")
                elif part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part.get("type") == "input_audio":
                    candidate = part.get("input_audio")
                    if audio_part is not None:
                        raise RequestValidationError(
                            "message contains multiple input_audio parts"
                        )
                    elif not isinstance(candidate, dict) or not isinstance(
                        candidate.get("data", ""), str
                    ):
                        raise RequestValidationError(
                            "input_audio requires a string data field"
                        )
                    else:
                        audio_part = {"data": candidate.get("data", "")}
                else:
                    raise RequestValidationError(
                        f"unsupported chat content part {part.get('type')!r}"
                    )
        else:
            raise RequestValidationError(
                "message content must be a string or parts list"
            )
        if audio_part is not None:
            if text_parts:
                raise RequestValidationError(
                    "message mixes text and audio; single modality per turn required"
                )
            else:
                turns.append({"role": role, "kind": "audio", "text": None})
                audio_sources.append(audio_part["data"])
        else:
            turns.append({"role": role, "kind": "text", "text": "".join(text_parts)})
    alternate_sources = metadata.get("audios")
    if alternate_sources:
        if any(audio_sources):
            raise RequestValidationError(
                "provide either input_audio parts or audios[], not both"
            )
        elif not isinstance(alternate_sources, list) or not all(
            isinstance(source, str) for source in alternate_sources
        ):
            raise RequestValidationError(
                "audios[] must be a list of paths or data URIs"
            )
        elif len(alternate_sources) != len(audio_sources):
            raise RequestValidationError(
                "audios[] must match audio turns in conversation order"
            )
        else:
            audio_sources = alternate_sources
    else:
        pass
    decoded_audio: list[tuple[torch.Tensor, int, str]] = []
    duration_seconds = 0.0
    for source in audio_sources:
        waveform, sample_rate = load_audio_source(source)
        duration_seconds += waveform.numel() / sample_rate
        fingerprint = hashlib.blake2b(
            waveform.numpy().tobytes(), digest_size=16
        ).hexdigest()
        decoded_audio.append((waveform, sample_rate, fingerprint))
    if duration_seconds > MAX_TOTAL_AUDIO_DURATION_SECONDS:
        raise RequestValidationError(
            f"total audio duration exceeds {MAX_TOTAL_AUDIO_DURATION_SECONDS} seconds"
        )
    else:
        return turns, decoded_audio
