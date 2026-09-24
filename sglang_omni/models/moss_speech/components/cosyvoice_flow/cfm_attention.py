# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conservative, empirically bit-exact FP32 CFM attention specialization.

Only the measured A800/Torch layouts and chunk lengths are enabled. QKV
projections remain whole; two aligned query ranges omit fully masked future
keys. Unknown contexts use the original Diffusers processor/backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import diffusers
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

_NATIVE_PREPARE_ATTENTION_MASK = Attention.prepare_attention_mask


@dataclass(frozen=True, eq=False)
class FullValidMask:
    """Solve-local proof from the private all-ones construction/copy path."""

    tensor: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    def matches(self, tensor: torch.Tensor) -> bool:
        return (
            tensor is self.tensor
            and tuple(tensor.shape) == self.shape
            and tensor.stride() == self.stride
            and tensor.dtype == self.dtype
            and tensor.device == self.device
        )


def _full_valid_proof(tensor: torch.Tensor) -> FullValidMask:
    return FullValidMask(
        tensor, tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device
    )


def new_full_valid_mask(
    reference: torch.Tensor, length: int
) -> tuple[torch.Tensor, FullValidMask]:
    mask = reference.new_ones((1, length)).unsqueeze(1)
    return mask, _full_valid_proof(mask)


def expand_cfg_provenance(
    proof: FullValidMask | None, source: torch.Tensor, cfg: torch.Tensor
) -> FullValidMask | None:
    """Call only immediately after the solver's original ``cfg[:] = source``."""
    if (
        proof is None
        or not proof.matches(source)
        or source.shape != (1, 1, source.shape[-1])
        or cfg.shape != (2, 1, source.shape[-1])
        or cfg.dtype != source.dtype
        or cfg.device != source.device
    ):
        return None
    return _full_valid_proof(cfg)


@dataclass(frozen=True, eq=False)
class AttentionPlan:
    bias: torch.Tensor
    chunk_size: int
    length: int


def make_attention_plan(
    mask: torch.Tensor,
    bias: torch.Tensor,
    proof: FullValidMask | None,
    *,
    streaming: bool,
    chunk_size: int,
) -> AttentionPlan | None:
    """Bind the decoder's original additive bias to its proven valid mask.

    This is metadata only. The private caller must use the unchanged causal
    bias constructor; arbitrary external masks never acquire provenance.
    """
    if (
        proof is None
        or not proof.matches(mask)
        or not streaming
        or chunk_size <= 0
        or mask.shape[:2] != (2, 1)
        or bias.ndim != 3
        or bias.shape[-2:] != (mask.shape[-1], mask.shape[-1])
        or bias.dtype != mask.dtype
        or bias.device != mask.device
    ):
        return None
    return AttentionPlan(bias, chunk_size, mask.shape[-1])


def shape_supported(length: int, chunk_size: int) -> bool:
    """Discrete shapes covered by complete codec strict-output regression."""
    return (chunk_size == 20 and length in range(1400, 3401, 20)) or (
        chunk_size == 100 and length in range(1500, 3401, 100)
    )


def runtime_supported(
    *,
    device_name: str,
    capability: tuple[int, int],
    torch_build: str,
    cuda_build: str | None,
    dtype: torch.dtype,
) -> bool:
    return (
        device_name == "NVIDIA A800-SXM4-80GB"
        and capability == (8, 0)
        and torch_build == "2.9.1+cu128"
        and cuda_build == "12.8"
        and dtype == torch.float32
    )


@lru_cache(maxsize=16)
def _device_metadata(index: int) -> tuple[str, tuple[int, int]]:
    return torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index)


def _segment_views(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    chunk_size: int,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    length = query.shape[-2]
    split = (length // 2 // 64) * 64
    allowed_end = ((split + chunk_size - 1) // chunk_size) * chunk_size
    key_end = min(length, ((allowed_end + 63) // 64) * 64)
    return (
        (
            query[..., :split, :],
            key[..., :key_end, :],
            value[..., :key_end, :],
            bias[..., :split, :key_end],
        ),
        (query[..., split:, :], key, value, bias[..., split:, :]),
    )


def _efficient_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    segments: tuple[tuple[torch.Tensor, ...], ...],
) -> bool:
    """Respect the active backend context, including user priority ordering."""
    choose = getattr(torch, "_fused_sdp_choice", None)
    if choose is None:
        return False
    expected = int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
    try:
        return all(
            choose(q, k, v, mask, 0.0, False) == expected
            for q, k, v, mask in ((query, key, value, bias), *segments)
        )
    except (RuntimeError, NotImplementedError):
        return False


def _segmented_attention(
    query: torch.Tensor, segments: tuple[tuple[torch.Tensor, ...], ...]
) -> torch.Tensor:
    pieces = [
        F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        for q, k, v, mask in segments
    ]
    batch, heads, length, dim = query.shape
    # Efficient SDPA allocates BMHD then transposes to BHMD. Restore that
    # exact layout so the following transpose/reshape remains a view.
    result = torch.empty_strided(
        (batch, heads, length, dim),
        (length * heads * dim, dim, heads * dim, 1),
        dtype=query.dtype,
        device=query.device,
    )
    return torch.cat(pieces, dim=-2, out=result)


def _projected_layout_supported(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bias: torch.Tensor
) -> bool:
    length = query.shape[-2]
    shape, stride = (2, 8, length, 64), (length * 512, 64, 512, 1)
    return all(
        tensor.shape == shape
        and tensor.stride() == stride
        and tensor.storage_offset() == 0
        and tensor.dtype == torch.float32
        and tensor.device == query.device
        for tensor in (query, key, value)
    ) and (
        bias.shape == (2, 8, length, length)
        and bias.stride() == (8 * length * length, length * length, length, 1)
        and bias.storage_offset() == 0
        and bias.dtype == query.dtype
        and bias.device == query.device
    )


class MossCfmAttnProcessor:
    """Local specialization of Diffusers 0.37.0 AttnProcessor2_0."""

    def __init__(self, original: Any, *, prefix_cache_handler: Any = None) -> None:
        self.original = original
        self._prefix_cache_handler = prefix_cache_handler

    def _shape_supported(self, length: int, chunk_size: int) -> bool:
        if self._prefix_cache_handler is not None:
            return self._prefix_cache_handler.shape_supported(length, chunk_size)
        return shape_supported(length, chunk_size)

    def _attention(
        self,
        query: torch.Tensor,
        segments: tuple,
        plan: AttentionPlan,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._prefix_cache_handler is not None:
            return self._prefix_cache_handler.attention(query, segments, plan, mask)
        return _segmented_attention(query, segments)

    def _runtime_supported(self, hidden_states: torch.Tensor) -> bool:
        if hidden_states.device.type != "cuda" or diffusers.__version__ != "0.37.0":
            return False
        name, capability = _device_metadata(hidden_states.device.index)
        return runtime_supported(
            device_name=name,
            capability=capability,
            torch_build=torch.__version__,
            cuda_build=torch.version.cuda,
            dtype=hidden_states.dtype,
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        moss_cfm_mask_plan: AttentionPlan | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        plan = moss_cfm_mask_plan
        if (
            plan is None
            or attention_mask is not plan.bias
            or not self._shape_supported(plan.length, plan.chunk_size)
            or hidden_states.ndim != 3
            or hidden_states.shape[:2] != (2, plan.length)
            or hidden_states.dtype != torch.float32
            or attn.training
            or torch.is_grad_enabled()
            or encoder_hidden_states is not None
            or type(attn) is not Attention
            or getattr(attn.prepare_attention_mask, "__func__", None)
            is not _NATIVE_PREPARE_ATTENTION_MASK
            or attn.heads != 8
            or getattr(attn.to_q, "out_features", None) != 512
            or getattr(attn.to_k, "out_features", None) != 512
            or getattr(attn.to_v, "out_features", None) != 512
            or attn.spatial_norm is not None
            or attn.group_norm is not None
            or attn.norm_q is not None
            or attn.norm_k is not None
            or args
            or kwargs
            or not self._runtime_supported(hidden_states)
        ):
            return self.original(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                *args,
                **kwargs,
            )

        residual = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        attention_mask = attention_mask.view(
            batch_size, attn.heads, -1, attention_mask.shape[-1]
        )
        # Keep the full projection dimensions and the original operation order.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        segments = _segment_views(query, key, value, attention_mask, plan.chunk_size)
        if _projected_layout_supported(query, key, value, attention_mask) and (
            _efficient_backend(query, key, value, attention_mask, segments)
        ):
            hidden_states = self._attention(query, segments, plan, attention_mask)
        else:
            # The late backend/layout fallback must not repeat QKV projections.
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_cfm_processors(module: torch.nn.Module) -> int:
    installed = 0
    for attention in module.modules():
        if (
            type(attention) is Attention
            and type(attention.processor) is AttnProcessor2_0
        ):
            attention.set_processor(MossCfmAttnProcessor(attention.processor))
            installed += 1
    return installed
