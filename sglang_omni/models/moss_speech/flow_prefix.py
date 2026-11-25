# SPDX-License-Identifier: Apache-2.0
"""Prefill the complete reference chunks of a streaming conditional Flow."""

import hashlib
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from sglang_omni.models.moss_speech.codec import VoiceConditioning
from sglang_omni.models.moss_speech.flow_attention import prepare_flow_attention_mask

MAX_REFERENCE_PREFIX_FRAMES = 1300


class EulerSolver(Protocol):
    def __call__(
        self,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        *,
        streaming: bool,
    ) -> torch.Tensor: ...


class StreamingFlow(Protocol):
    pre_lookahead_len: int
    token_mel_ratio: int

    def inference(
        self,
        *,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        streaming: bool,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def reference_prefix_key(voice: VoiceConditioning) -> bytes:
    identity = hashlib.sha256()
    for tensor in (voice["prompt_token"], voice["prompt_feat"], voice["embedding"]):
        identity.update(tensor.contiguous().view(torch.uint8).cpu().numpy())
    return identity.digest()


@dataclass(kw_only=True)
class ReferencePrefix:
    reference_key: bytes
    mel_frames: int
    key_values: list[torch.Tensor] = field(default_factory=list)
    convolution_tails: list[torch.Tensor] = field(default_factory=list)


@dataclass(kw_only=True)
class PrefixSolve:
    prefix: ReferencePrefix
    prefill: bool
    projection_index: int = 0
    convolution_index: int = 0
    attention_bias: torch.Tensor | None = None


class StreamingReferencePrefill:
    def __init__(self, decoder: torch.nn.Module) -> None:
        self.cached: ReferencePrefix | None = None
        self.current: ContextVar[PrefixSolve | None] = ContextVar(
            "moss_streaming_reference_prefix", default=None
        )
        estimator = decoder.estimator
        assert estimator.num_decoding_left_chunks == -1
        causal_convolution_type = type(estimator.down_blocks[0][0].block1.block[0])
        for layer in estimator.modules():
            if type(layer) is Attention and type(layer.processor) is AttnProcessor2_0:
                for projection in (layer.to_k, layer.to_v):
                    assert type(projection) is torch.nn.Linear
                    projection.forward = partial(
                        self.project, weakref.proxy(projection)
                    )
                layer.prepare_attention_mask = partial(self.mask, weakref.proxy(layer))
            elif type(layer) is causal_convolution_type:
                layer.forward = partial(self.convolve, weakref.proxy(layer))
            else:
                pass
        # note (Codex): Slicing here leaves the checkpoint's absolute-position noise and Euler grid intact.
        decoder.solve_euler = partial(
            self.solve_euler, type(decoder).solve_euler.__get__(weakref.proxy(decoder))
        )

    def project(
        self, layer: torch.nn.Linear, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        projected = F.linear(hidden_states, layer.weight, layer.bias)
        call = self.current.get()
        if call is None:
            return projected
        elif call.prefill:
            call.prefix.key_values.append(projected)
            return projected
        else:
            previous = call.prefix.key_values[call.projection_index]
            call.projection_index += 1
            return torch.cat((previous, projected), dim=1)

    def convolve(self, layer: torch.nn.Conv1d, channels: torch.Tensor) -> torch.Tensor:
        call = self.current.get()
        if call is None:
            return layer.__class__.forward(layer, channels)
        elif call.prefill:
            call.prefix.convolution_tails.append(
                channels[..., -layer.causal_padding :].clone()
            )
            return layer.__class__.forward(layer, channels)
        else:
            tail = call.prefix.convolution_tails[call.convolution_index]
            call.convolution_index += 1
            return F.conv1d(
                torch.cat((tail, channels), dim=-1),
                layer.weight,
                layer.bias,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
            )

    def mask(
        self,
        attention: Attention,
        attention_mask: torch.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> torch.Tensor | None:
        call = self.current.get()
        if call is None or call.prefill:
            pass
        else:
            if call.attention_bias is None:
                # note (Codex): Every query can attend to all earlier complete chunks.
                call.attention_bias = F.pad(attention_mask, (call.prefix.mel_frames, 0))
            else:
                pass
            attention_mask = call.attention_bias
            target_length += call.prefix.mel_frames
        return prepare_flow_attention_mask(
            attention, attention_mask, target_length, batch_size, out_dim
        )

    def solve_euler(
        self,
        full_solver: EulerSolver,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool,
    ) -> torch.Tensor:
        call = self.current.get()
        if call is None:
            return full_solver(
                noise, timesteps, mu, mask, spks, cond, streaming=streaming
            )
        else:
            frames = mu.size(-1)
            offset, end = (
                (0, call.prefix.mel_frames)
                if call.prefill
                else (call.prefix.mel_frames, frames)
            )
            suffix = full_solver(
                noise[..., offset:end].contiguous(),
                timesteps,
                mu[..., offset:end].contiguous(),
                mask[..., offset:end].contiguous(),
                spks,
                cond[..., offset:end].contiguous(),
                streaming=streaming,
            )
            if not call.prefill:
                assert call.projection_index == len(call.prefix.key_values)
                assert call.convolution_index == len(call.prefix.convolution_tails)
            else:
                pass
            return F.pad(suffix, (offset, frames - end))

    @contextmanager
    def read_step(self, prefix: ReferencePrefix) -> Iterator[None]:
        token = self.current.set(PrefixSolve(prefix=prefix, prefill=False))
        try:
            yield
        finally:
            self.current.reset(token)

    @contextmanager
    def for_voice(
        self,
        flow: StreamingFlow,
        voice: VoiceConditioning,
        reference_key: bytes,
        chunk_codes: int,
    ) -> Iterator[None]:
        tokens = voice["prompt_token"]
        # note (Codex): The encoder's right context must also end inside the reference.
        closed_chunks = (tokens.size(1) - flow.pre_lookahead_len) // chunk_codes
        closed_frames = closed_chunks * chunk_codes * flow.token_mel_ratio
        if closed_frames <= 0 or closed_frames > MAX_REFERENCE_PREFIX_FRAMES:
            yield
        else:
            prefix = self.cached
            if prefix is None or prefix.reference_key != reference_key:
                self.cached = None
                prefix = ReferencePrefix(
                    reference_key=reference_key,
                    mel_frames=closed_frames,
                )
                token = self.current.set(PrefixSolve(prefix=prefix, prefill=True))
                try:
                    flow.inference(
                        token=tokens[:, :0],
                        token_len=tokens.new_tensor([0]),
                        prompt_token_len=tokens.new_tensor([tokens.size(1)]),
                        prompt_feat_len=tokens.new_tensor(
                            [voice["prompt_feat"].size(1)]
                        ),
                        **voice,
                        streaming=True,
                        finalize=True,
                    )
                finally:
                    self.current.reset(token)
                self.cached = prefix
            else:
                pass
            with self.read_step(prefix):
                yield
