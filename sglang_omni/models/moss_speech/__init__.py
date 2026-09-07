# SPDX-License: Apache-2.0
"""MOSS-Speech model package.

Static capability declaration for V1: every optional capability is disabled
until independently verified (streaming -> P6, cuda graph / compile / batched
vocoder -> later phases per plan §7). Keep this module import-light: the
registry imports ``config`` during discovery and must not touch CUDA or load
codec weights at that point.
"""

from __future__ import annotations

from sglang_omni.models.model_capabilities import ModelCapabilities

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=False,
    supports_batch_vocoder=False,
    supports_streaming_vocoder=False,
    supports_cuda_graph=False,
    supports_torch_compile=False,
    supports_breakable_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config"]
