# SPDX-License-Identifier: Apache-2.0
"""Carry fixed-voice tensors as opaque typed bytes through pipeline state."""

from __future__ import annotations

import hashlib
import json
import math

import torch

from sglang_omni.models.moss_speech.payload_types import MossSpeechState, VoicePayload
from sglang_omni.scheduling.typed_tensor import decode_typed_tensor, encode_typed_tensor

VOICE_FIELDS = ("prompt_token", "prompt_feat", "embedding")
VOICE_KEYS = {
    f"{name}_{suffix}"
    for name in VOICE_FIELDS
    for suffix in ("bytes", "shape", "dtype")
}


def validate_voice_payload(payload: object) -> VoicePayload:
    """Validate the wire boundary before hashing or restoring voice tensors."""
    if not isinstance(payload, dict) or set(payload) != VOICE_KEYS:
        raise ValueError("Voice payload requires exactly three typed tensors")
    else:
        pass
    validated_payload: VoicePayload = {}
    tensor_shapes: dict[str, list[int]] = {}
    for name in VOICE_FIELDS:
        raw = payload[f"{name}_bytes"]
        shape = payload[f"{name}_shape"]
        dtype = payload[f"{name}_dtype"]
        expected_dtypes = (
            ("uint16", "int32") if name == "prompt_token" else ("float32",)
        )
        if (
            not isinstance(dtype, str)
            or dtype not in expected_dtypes
            or not isinstance(raw, bytes)
        ):
            raise ValueError(f"Invalid voice tensor bytes/dtype: {name}")
        else:
            pass
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(length) is not int or length <= 0 for length in shape)
        ):
            raise ValueError(f"Invalid voice tensor shape: {name}")
        else:
            pass
        itemsize = 2 if dtype == "uint16" else 4
        if len(raw) != math.prod(shape) * itemsize:
            raise ValueError(f"Incomplete voice tensor bytes: {name}")
        else:
            pass
        validated_payload[f"{name}_bytes"] = raw
        validated_payload[f"{name}_shape"] = shape
        validated_payload[f"{name}_dtype"] = dtype
        tensor_shapes[name] = shape
    token_shape = tensor_shapes["prompt_token"]
    if (
        len(token_shape) != 2
        or token_shape[0] != 1
        or tensor_shapes["prompt_feat"] != [1, token_shape[1] * 4, 80]
        or tensor_shapes["embedding"] != [1, 192]
    ):
        raise ValueError("Voice token/mel/embedding shape mismatch")
    else:
        pass
    return validated_payload


def pack_voice_payload(voice: dict[str, torch.Tensor]) -> VoicePayload:
    if set(voice) != set(VOICE_FIELDS):
        raise ValueError("Voice requires prompt_token, prompt_feat and embedding")
    else:
        pass
    payload: VoicePayload = {}
    for name in VOICE_FIELDS:
        value = voice[name].to(
            dtype=torch.int32 if name == "prompt_token" else torch.float32
        )
        payload.update(encode_typed_tensor(value, key=name))
    return validate_voice_payload(payload)


def unpack_voice_payload(payload: object) -> dict[str, torch.Tensor]:
    validated_payload = validate_voice_payload(payload)
    voice: dict[str, torch.Tensor] = {}
    for name in VOICE_FIELDS:
        tensor = decode_typed_tensor(validated_payload, key=name)
        assert isinstance(tensor, torch.Tensor)
        voice[name] = tensor.to(
            dtype=torch.int32 if name == "prompt_token" else torch.float32
        )
    return voice


def voice_payload_digest(payload: object) -> str:
    validated_payload = validate_voice_payload(payload)
    digest = hashlib.sha256()
    for name in VOICE_FIELDS:
        metadata = [
            name,
            validated_payload[f"{name}_dtype"],
            validated_payload[f"{name}_shape"],
        ]
        digest.update(json.dumps(metadata, separators=(",", ":")).encode())
        raw = validated_payload[f"{name}_bytes"]
        assert isinstance(raw, bytes)
        digest.update(raw)
    return digest.hexdigest()


def state_voice_payload(state: MossSpeechState) -> VoicePayload:
    return validate_voice_payload(state.voice_payload)
