# SPDX-License-Identifier: Apache-2.0
"""Solve-local, same-value aligned full attention bias for qualified final1492."""
from __future__ import annotations

from typing import Any

import diffusers
import torch
import torch.nn.functional as F

from ..cfm_prefix_cache import _execution_context
from .cfm_attention import FullValidMask, _device_metadata, runtime_supported

_NATIVE_MASK_MATCHES = FullValidMask.matches


def _runtime_supported(value: torch.Tensor) -> bool:
    if value.device.type != "cuda" or diffusers.__version__ != "0.37.0":
        return False
    name, capability = _device_metadata(value.device.index)
    return (
        runtime_supported(
            device_name=name,
            capability=capability,
            torch_build=torch.__version__,
            cuda_build=torch.version.cuda,
            dtype=value.dtype,
        )
        and not torch.backends.cuda.matmul.allow_tf32
        and torch.backends.cudnn.enabled
        and torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.benchmark
        and torch.backends.cudnn.allow_tf32
        and not torch.is_autocast_enabled("cuda")
        and not torch.are_deterministic_algorithms_enabled()
    )


class FinalBiasAlignment:
    """Counters retain no tensors; the caller's original solve dict owns each view."""

    def __init__(self, *, elide_zero_bias: bool = False) -> None:
        if type(elide_zero_bias) is not bool:
            raise ValueError("Explicit final zero-bias mode required")
        self._elide_zero_bias = elide_zero_bias
        self._policy = _execution_context()
        self._key = object()
        self.prepared = self.reused = self.fallback_calls = 0
        self.closed = False

    def apply(
        self,
        *,
        decoder: Any,
        x: torch.Tensor,
        mask: torch.Tensor,
        bias: torch.Tensor,
        proof: FullValidMask | None,
        streaming: bool,
        cache: dict | None,
    ) -> torch.Tensor | None:
        from . import decoder as native

        if (
            self.closed
            or streaming
            or torch.is_grad_enabled()
            or type(decoder) is not native.CausalConditionalDecoder
            or decoder.training
            or decoder.attention_heads != 8
            or decoder.static_chunk_size != 100
            or getattr(decoder._attention_bias, "__func__", None)
            is not native._NATIVE_CAUSAL_ATTENTION_BIAS
            or native.mask_to_bias is not native._NATIVE_MASK_TO_BIAS
            or native.add_optional_chunk_mask
            is not native._NATIVE_ADD_OPTIONAL_CHUNK_MASK
            or type(proof) is not FullValidMask
            or getattr(proof.matches, "__func__", None) is not _NATIVE_MASK_MATCHES
            or not proof.matches(mask)
            or type(cache) is not dict
            or mask.shape != (2, 1, 1492)
            or mask.stride() != (1492, 1492, 1)
            or x.shape != (2, 1492, 256)
            or x.dtype != torch.float32
            or bias.shape != (16, 1492, 1492)
            or bias.stride() != (1492 * 1492, 1492, 1)
            or bias.storage_offset() != 0
            or bias.dtype != x.dtype
            or mask.dtype != x.dtype
            or bias.device != x.device
            or mask.device != x.device
            or self._policy[:5] != (True, True, True, True, True)
            or _execution_context() != self._policy
            or not _runtime_supported(x)
        ):
            self.fallback_calls += 1
            return bias
        key = (self._key, id(bias))
        if key in cache:
            self.reused += 1
            return cache[key]
        if self._elide_zero_bias:
            # Only the qualified original final full-valid constructor proves
            # an all-zero additive bias. Keep every other admission check.
            cache[key] = None
            self.prepared += 1
            return None
        # Native full-valid construction proves B/H equality. Keep every query
        # row; padding only changes physical stride, never logical mask values.
        backing = F.pad(bias[:1], (0, 4))
        aligned = backing[..., :1492].expand(16, -1, -1)
        cache[key] = aligned
        self.prepared += 1
        return aligned

    def status(self) -> dict[str, Any]:
        return dict(
            enabled=not self.closed,
            elide_zero_bias=self._elide_zero_bias,
            closed=self.closed,
            prepared=self.prepared,
            reused=self.reused,
            fallback_calls=self.fallback_calls,
            calls=self.prepared + self.reused + self.fallback_calls,
            persistent_tensor_bytes=0,
            scope="cumulative_decoder_bias_calls_including_startup",
        )

    def close(self) -> None:
        self.closed = True
