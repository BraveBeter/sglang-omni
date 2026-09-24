# SPDX-License-Identifier: Apache-2.0
"""Compact fixed-voice conditioning through model-owned pipeline state.

The framework typed-tensor format stays opaque across AR and control-plane
messages. Only the consuming vocoder restores tensors. Legacy lists remain
readable for existing internal callers.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch

from sglang_omni.scheduling.typed_tensor import decode_typed_tensor, encode_typed_tensor

_FIELDS = ("prompt_token", "prompt_feat", "embedding")
_KEYS = {
    f"{name}_{suffix}" for name in _FIELDS for suffix in ("bytes", "shape", "dtype")
}


def _validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or set(payload) != _KEYS:
        raise ValueError("Voice payload requires exactly three typed tensors")
    for name in _FIELDS:
        raw, shape, dtype = (
            payload[f"{name}_{suffix}"] for suffix in ("bytes", "shape", "dtype")
        )
        expected_dtypes = (
            ("uint16", "int32") if name == "prompt_token" else ("float32",)
        )
        if dtype not in expected_dtypes or not isinstance(raw, bytes):
            raise ValueError(f"Invalid voice tensor bytes/dtype: {name}")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(n) is not int or n <= 0 for n in shape)
        ):
            raise ValueError(f"Invalid voice tensor shape: {name}")
        itemsize = 2 if dtype == "uint16" else 4
        if len(raw) != math.prod(shape) * itemsize:
            raise ValueError(f"Incomplete voice tensor bytes: {name}")
    tokens, feat, embedding = (payload[f"{name}_shape"] for name in _FIELDS)
    if (
        len(tokens) != 2
        or tokens[0] != 1
        or feat != [1, tokens[1] * 4, 80]
        or embedding != [1, 192]
    ):
        raise ValueError("Voice token/mel/embedding shape mismatch")


def pack_voice_payload(voice: dict[str, Any]) -> dict[str, Any]:
    """Use the existing typed-tensor codec with the previous consumer dtypes."""
    if not isinstance(voice, dict) or set(voice) != set(_FIELDS):
        raise ValueError("Voice requires prompt_token, prompt_feat and embedding")
    payload = {}
    for name in _FIELDS:
        value = torch.as_tensor(
            voice[name], dtype=torch.int32 if name == "prompt_token" else torch.float32
        )
        payload.update(encode_typed_tensor(value, key=name))
    _validate_payload(payload)
    return payload


def unpack_voice_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Restore caller-owned CPU tensors only at the vocoder boundary."""
    _validate_payload(payload)
    return {
        name: decode_typed_tensor(payload, key=name).to(
            dtype=torch.int32 if name == "prompt_token" else torch.float32
        )
        for name in _FIELDS
    }


def voice_payload_digest(payload: dict[str, Any]) -> str:
    """Hash exact conditioning bytes and their typed shape without decoding."""
    _validate_payload(payload)
    digest = hashlib.sha256()
    for name in _FIELDS:
        metadata = [name, payload[f"{name}_dtype"], payload[f"{name}_shape"]]
        digest.update(json.dumps(metadata, separators=(",", ":")).encode())
        digest.update(payload[f"{name}_bytes"])
    return digest.hexdigest()


def state_voice_payload(state: Any) -> dict[str, Any]:
    """Keep new state opaque; convert legacy lists only at their consumer."""
    payload = state.voice_payload
    if payload is not None:
        _validate_payload(payload)
        if state.voice_token_ids or state.voice_feat or state.voice_embedding:
            legacy = pack_voice_payload(
                {
                    "prompt_token": [state.voice_token_ids],
                    "prompt_feat": [state.voice_feat],
                    "embedding": [state.voice_embedding],
                }
            )
            if voice_payload_digest(legacy) != voice_payload_digest(payload):
                raise ValueError("Conflicting compact and legacy voice conditioning")
        return payload
    return pack_voice_payload(
        {
            "prompt_token": [state.voice_token_ids],
            "prompt_feat": [state.voice_feat],
            "embedding": [state.voice_embedding],
        }
    )
