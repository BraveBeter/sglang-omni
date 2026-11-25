# SPDX-License-Identifier: Apache-2.0
"""Broadcast the live CFM attention bias across SDPA heads."""

import weakref
from types import MethodType

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from sglang_omni.models.moss_speech.flow_precision import FlowEulerDecoder


def prepare_flow_attention_mask(
    attention: Attention,
    attention_mask: torch.Tensor | None,
    target_length: int,
    batch_size: int,
    out_dim: int = 3,
) -> torch.Tensor | None:
    if (
        attention_mask is not None
        and attention_mask.ndim == 3
        and attention_mask.shape[0] == batch_size
        and attention_mask.shape[-1] == target_length
        and out_dim in (3, 4)
    ):
        # note (Codex): The stock SDPA processor immediately views this result as four dimensions.
        return attention_mask.unsqueeze(1).expand(-1, attention.heads, -1, -1)
    else:
        return Attention.prepare_attention_mask(
            attention, attention_mask, target_length, batch_size, out_dim
        )


def install_flow_attention_broadcast(decoder: torch.nn.Module) -> None:
    if isinstance(decoder, FlowEulerDecoder):
        decoder = decoder.decoder
    else:
        pass
    for attention in decoder.estimator.modules():
        if (
            type(attention) is Attention
            and type(attention.processor) is AttnProcessor2_0
        ):
            attention.prepare_attention_mask = MethodType(
                prepare_flow_attention_mask, weakref.proxy(attention)
            )
        else:
            pass
