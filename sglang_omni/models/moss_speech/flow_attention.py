# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# SPDX-License-Identifier: Apache-2.0
"""Reuse the MOSS Flow middle attention bias within one estimator forward."""

from __future__ import annotations

import sys

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from torch import nn


def apply_flow_transformers(
    transformer_blocks: nn.ModuleList,
    hidden_states: torch.Tensor,
    attention_bias: torch.Tensor,
    timestep_embedding: torch.Tensor,
) -> torch.Tensor:
    for transformer in transformer_blocks:
        hidden_states = transformer(
            hidden_states=hidden_states,
            attention_mask=attention_bias,
            timestep=timestep_embedding,
        )
    return hidden_states.transpose(1, 2).contiguous()


def install_middle_bias_reuse(estimator: nn.Module) -> None:
    flow_decoder = sys.modules.get("cosyvoice.flow.decoder")
    transformer_module = sys.modules.get("matcha.models.components.transformer")
    if (
        flow_decoder is None
        or transformer_module is None
        or type(estimator) is not flow_decoder.CausalConditionalDecoder
    ):
        return
    else:
        original_forward = estimator.forward
        add_chunk_mask = flow_decoder.add_optional_chunk_mask
        mask_to_bias = flow_decoder.mask_to_bias

    middle_transformers = tuple(
        (transformer, transformer.attn1)
        for _, transformer_blocks in estimator.mid_blocks
        for transformer in transformer_blocks
    )
    if not all(
        type(transformer) is transformer_module.BasicTransformerBlock
        and transformer.attn2 is None
        and type(attention) is Attention
        for transformer, attention in middle_transformers
    ):
        return
    else:
        pass

    def dense_attention_bias(
        hidden_states: torch.Tensor, sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = add_chunk_mask(
            hidden_states, sequence_mask.bool(), False, False, 0, 0, -1
        ).repeat(1, hidden_states.size(1), 1)
        return mask_to_bias(attention_mask, hidden_states.dtype)

    def forward(
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> torch.Tensor:
        if (
            streaming
            or estimator.training
            or torch.is_grad_enabled()
            or vars(nn.modules.module)["_global_forward_hooks"]
            or vars(nn.modules.module)["_global_forward_pre_hooks"]
            or any(
                transformer.attn1 is not attention
                or transformer.attn2 is not None
                or type(attention.processor) is not AttnProcessor2_0
                or any(
                    vars(module)["_forward_hooks"]
                    or vars(module)["_forward_pre_hooks"]
                    or "forward" in vars(module)
                    for module in (transformer, attention)
                )
                or "prepare_attention_mask" in vars(attention)
                for transformer, attention in middle_transformers
            )
        ):
            return original_forward(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            timestep_embedding = estimator.time_mlp(
                estimator.time_embeddings(t).to(t.dtype)
            )

        hidden_states = torch.cat((x, mu), dim=1)
        if spks is not None:
            speaker_embeddings = spks.unsqueeze(-1).expand(-1, -1, x.size(-1))
            hidden_states = torch.cat((hidden_states, speaker_embeddings), dim=1)
        else:
            pass
        if cond is not None:
            hidden_states = torch.cat((hidden_states, cond), dim=1)
        else:
            pass

        skip_states = []
        sequence_masks = [mask]
        for resnet, transformer_blocks, downsample in estimator.down_blocks:
            sequence_mask = sequence_masks[-1]
            hidden_states = resnet(hidden_states, sequence_mask, timestep_embedding)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            hidden_states = apply_flow_transformers(
                transformer_blocks,
                hidden_states,
                dense_attention_bias(hidden_states, sequence_mask),
                timestep_embedding,
            )
            skip_states.append(hidden_states)
            hidden_states = downsample(hidden_states * sequence_mask)
            sequence_masks.append(sequence_mask[:, :, ::2])
        sequence_masks.pop()

        middle_mask = sequence_masks[-1]
        middle_bias = None
        for resnet, transformer_blocks in estimator.mid_blocks:
            hidden_states = resnet(hidden_states, middle_mask, timestep_embedding)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            if (
                middle_bias is None
                or middle_bias.size(1) != hidden_states.size(1)
                or middle_bias.dtype != hidden_states.dtype
            ):
                middle_bias = dense_attention_bias(hidden_states, middle_mask)
            else:
                pass
            hidden_states = apply_flow_transformers(
                transformer_blocks, hidden_states, middle_bias, timestep_embedding
            )
        del middle_bias

        for resnet, transformer_blocks, upsample in estimator.up_blocks:
            sequence_mask = sequence_masks.pop()
            skip = skip_states.pop()
            hidden_states = torch.cat(
                (hidden_states[:, :, : skip.size(-1)], skip), dim=1
            )
            hidden_states = resnet(hidden_states, sequence_mask, timestep_embedding)
            hidden_states = hidden_states.transpose(1, 2).contiguous()
            hidden_states = apply_flow_transformers(
                transformer_blocks,
                hidden_states,
                dense_attention_bias(hidden_states, sequence_mask),
                timestep_embedding,
            )
            hidden_states = upsample(hidden_states * sequence_mask)
        hidden_states = estimator.final_block(hidden_states, sequence_mask)
        return estimator.final_proj(hidden_states * sequence_mask) * mask

    estimator.forward = forward
