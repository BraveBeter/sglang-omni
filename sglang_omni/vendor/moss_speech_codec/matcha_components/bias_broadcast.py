# SPDX-License-Identifier: Apache-2.0
"""Live FP32 CFM attention bias broadcasting with a stock-processor fallback."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

logger = logging.getLogger(__name__)


class LiveBiasBroadcastProcessor:
    """Preserve live FP32 bias values without allocating one copy per head."""

    def __init__(self, fallback: AttnProcessor2_0) -> None:
        self.fallback = fallback
        self.broadcast_calls: int = 0
        self.fallback_calls: int = 0
        self.last_head_stride: int | None = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: float,
        **kwargs: float,
    ) -> torch.Tensor:
        if (
            args
            or kwargs
            or type(attn) is not Attention
            or attn.training
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or hidden_states.device.type != "cuda"
            or hidden_states.dtype is not torch.float32
            or hidden_states.ndim != 3
            or hidden_states.shape[0] != 2
            or not hidden_states.is_contiguous()
            or encoder_hidden_states is not None
            or temb is not None
            or attn.spatial_norm is not None
            or attn.group_norm is not None
            or attn.norm_q is not None
            or attn.norm_k is not None
            or attn.to_k.out_features != attn.heads * 64
            or "prepare_attention_mask" in vars(attn)
            or any(
                layer._forward_hooks or layer._forward_pre_hooks
                for layer in attn.modules()
            )
            or attention_mask is None
            or attention_mask.dtype is not torch.float32
            or attention_mask.device != hidden_states.device
            or attention_mask.ndim != 3
            or attention_mask.shape[0] != hidden_states.shape[0]
            or attention_mask.shape[1] not in (1, hidden_states.shape[1])
            or attention_mask.shape[2] != hidden_states.shape[1]
            or attention_mask.stride(-1) != 1
            or attention_mask.numel() == 0
        ):
            self.fallback_calls += 1
            return self.fallback(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                *args,
                **kwargs,
            )
        else:
            residual = hidden_states
            batch_size = hidden_states.shape[0]
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, attn.heads, -1, -1
            )
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            head_channels = key.shape[-1] // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_channels).transpose(
                1, 2
            )
            key = key.view(batch_size, -1, attn.heads, head_channels).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_channels).transpose(
                1, 2
            )
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_channels
            )
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            else:
                pass
            self.last_head_stride = attention_mask.stride(1)
            if self.broadcast_calls == 0:
                logger.info(
                    f"MOSS_CODEC_BIAS_BROADCAST_ACTIVE heads={attn.heads} dtype={query.dtype}"
                )
            else:
                pass
            self.broadcast_calls += 1
            return hidden_states / attn.rescale_output_factor


def install_broadcast_bias_attention(
    estimator: nn.Module,
) -> tuple[LiveBiasBroadcastProcessor, ...]:
    processors: list[LiveBiasBroadcastProcessor] = []
    for layer in estimator.modules():
        if type(layer) is Attention and type(layer.processor) is AttnProcessor2_0:
            processor = LiveBiasBroadcastProcessor(layer.processor)
            layer.set_processor(processor)
            processors.append(processor)
        elif (
            type(layer) is Attention
            and type(layer.processor) is LiveBiasBroadcastProcessor
        ):
            processors.append(layer.processor)
        else:
            pass
    return tuple(processors)
