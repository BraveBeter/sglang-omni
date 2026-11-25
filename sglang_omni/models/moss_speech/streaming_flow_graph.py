# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
# SPDX-License-Identifier: Apache-2.0
"""Replay the chunk-trained, single-resolution CFM estimator at exact frame counts."""

import logging
import sys
from dataclasses import dataclass
from threading import get_ident

import torch
from torch import nn

logger = logging.getLogger(__name__)
MAX_GRAPH_FRAMES = 3200
MIN_CAPTURE_FREE_BYTES = 2 * 1024**3


@dataclass
class CapturedEstimator:
    inputs: tuple[torch.Tensor, ...]
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor


def apply_transformers(
    blocks: nn.ModuleList,
    hidden_states: torch.Tensor,
    attention_bias: torch.Tensor,
    timestep_embedding: torch.Tensor,
) -> torch.Tensor:
    hidden_states = hidden_states.transpose(1, 2).contiguous()
    for block in blocks:
        hidden_states = block(
            hidden_states=hidden_states,
            attention_mask=attention_bias,
            timestep=timestep_embedding,
        )
    return hidden_states.transpose(1, 2).contiguous()


class StreamingEstimatorGraphs:
    def __init__(self, estimator: nn.Module, device: torch.device) -> None:
        self.estimator = estimator
        self.owner_thread = get_ident()
        self.owner_stream = torch.cuda.current_stream(device)
        self.capture_stream = torch.cuda.Stream(device=device)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[int, CapturedEstimator] = {}
        self.capture_enabled = True

    def run_estimator(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        estimator = self.estimator
        timestep_embedding = estimator.time_mlp(
            estimator.time_embeddings(t).to(t.dtype)
        )
        # note (Codex): All blocks in this single-resolution UNet see the same chunk grid and valid keys.
        positions = torch.arange(x.size(2), device=x.device)
        chunk_frames = estimator.static_chunk_size
        chunk_ends = (positions // chunk_frames + 1) * chunk_frames
        visible = (positions[None, :] < chunk_ends[:, None]) & mask.bool()
        visible = visible | ~visible.any(dim=-1, keepdim=True)
        attention_bias = (1.0 - visible.to(x.dtype)) * -1.0e10
        speakers = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        hidden_states = torch.cat((x, mu, speakers, cond), dim=1)
        resnet, blocks, downsample = estimator.down_blocks[0]
        skip = apply_transformers(
            blocks,
            resnet(hidden_states, mask, timestep_embedding),
            attention_bias,
            timestep_embedding,
        )
        hidden_states = downsample(skip * mask)
        for resnet, blocks in estimator.mid_blocks:
            hidden_states = apply_transformers(
                blocks,
                resnet(hidden_states, mask, timestep_embedding),
                attention_bias,
                timestep_embedding,
            )
        resnet, blocks, upsample = estimator.up_blocks[0]
        hidden_states = torch.cat((hidden_states[..., : skip.shape[-1]], skip), dim=1)
        hidden_states = apply_transformers(
            blocks,
            resnet(hidden_states, mask, timestep_embedding),
            attention_bias,
            timestep_embedding,
        )
        hidden_states = upsample(hidden_states * mask)
        hidden_states = estimator.final_block(hidden_states, mask)
        return estimator.final_proj(hidden_states * mask) * mask

    def capture(self, inputs: tuple[torch.Tensor, ...]) -> CapturedEstimator:
        static = tuple(tensor.clone() for tensor in inputs)
        self.capture_stream.wait_stream(self.owner_stream)
        try:
            with torch.cuda.stream(self.capture_stream):
                self.run_estimator(*static)
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                try:
                    output = self.run_estimator(*static)
                finally:
                    graph.capture_end()
        finally:
            self.owner_stream.wait_stream(self.capture_stream)
        return CapturedEstimator(static, graph, output)

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        frames = x.size(-1)
        if (
            not streaming
            or self.estimator.training
            or torch.is_grad_enabled()
            or x.dtype != torch.bfloat16
            or x.device != self.owner_stream.device
            or x.shape[:2] != (2, 80)
            or frames > MAX_GRAPH_FRAMES
            or frames % self.estimator.static_chunk_size
            or get_ident() != self.owner_thread
            or torch.cuda.current_stream(x.device) != self.owner_stream
        ):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            inputs = (x, mask, mu, t, spks, cond)
        captured = self.graphs.get(frames)
        if captured is None and self.capture_enabled:
            free_bytes, _ = torch.cuda.mem_get_info(x.device)
            if free_bytes < MIN_CAPTURE_FREE_BYTES:
                self.capture_enabled = False
            else:
                try:
                    captured = self.capture(inputs)
                except RuntimeError:
                    logger.exception(
                        "Disabling MOSS streaming estimator graph capture at %d frames",
                        frames,
                    )
                    self.capture_enabled = False
                    self.graphs.clear()
                else:
                    self.graphs[frames] = captured
                    logger.info(
                        "Captured MOSS streaming estimator at %d frames", frames
                    )
        else:
            pass
        if captured is None:
            return self.estimator(*inputs, streaming=streaming)
        else:
            for destination, source in zip(captured.inputs, inputs, strict=True):
                destination.copy_(source)
            captured.graph.replay()
            return captured.output.clone()


def install_streaming_estimator_graphs(
    decoder: nn.Module, device: torch.device
) -> None:
    flow_decoder = sys.modules.get("cosyvoice.flow.decoder")
    estimator = decoder.estimator
    if (
        flow_decoder is not None
        and type(estimator) is flow_decoder.CausalConditionalDecoder
        and len(estimator.down_blocks) == len(estimator.up_blocks) == 1
        and estimator.static_chunk_size == 100
        and estimator.num_decoding_left_chunks == -1
    ):
        decoder.forward_estimator = StreamingEstimatorGraphs(estimator, device)
    else:
        pass
