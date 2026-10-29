# SPDX-License-Identifier: Apache-2.0
"""Startup graphs for one native BF16 decode projection profile."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from sglang.srt.utils import DynamicGradMode

import sglang_omni.models.moss_speech.model_layers as model_layers
from sglang_omni.models.moss_speech.model_layers import (
    MossSpeechDecoderLayer,
    MossSpeechRMSNorm,
)

ProjectionSection = Literal["prepare", "finish"]
ProjectionInputs = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
ProjectionOutputs = (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)


@dataclass(frozen=True, kw_only=True)
class TensorState:
    identity: int
    pointer: int
    version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, kw_only=True)
class LayerState:
    identity: int
    geometry: tuple[int | float, ...]
    norm_epsilon: tuple[float, ...]
    functions: tuple[int, ...]
    tensors: tuple[TensorState, ...]


@dataclass(frozen=True, kw_only=True)
class NumericalContext:
    gradients: bool
    inference_mode: bool
    autocast: bool
    float32_matmul_precision: str
    matmul_tf32: bool
    bf16_reduced_precision: bool
    blas_library: str
    deterministic_algorithms: bool
    deterministic_warnings: bool


@dataclass(kw_only=True)
class PreparedSection:
    graph: torch.cuda.CUDAGraph
    inputs: ProjectionInputs | None
    outputs: ProjectionOutputs | None = None


class MossSpeechProjectionGraphs:
    """Own two graphs per layer; preserve eager attention and request state."""

    def __init__(
        self, layers: Sequence[MossSpeechDecoderLayer], *, batch_size: int
    ) -> None:
        if (
            type(batch_size) is not int
            or not 1 <= batch_size <= 4
            or not layers
            or len({id(layer) for layer in layers}) != len(layers)
        ):
            raise ValueError("Projection graph batch size must be an integer in 1..4")
        else:
            pass
        self.layers = tuple(layers)
        self.batch_size = batch_size
        self.device = self.layers[0].input_layernorm.weight.device
        if self.device.type != "cuda":
            raise ValueError("Projection graphs require CUDA BF16 decoder layers")
        else:
            pass
        self.lock = threading.RLock()
        self.owner_stream = torch.cuda.current_stream(self.device)
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.sections: dict[
            tuple[MossSpeechDecoderLayer, ProjectionSection], PreparedSection
        ] = {}
        self.inverse_frequencies: dict[MossSpeechDecoderLayer, torch.Tensor] = {}
        self.closed = False
        self.replays = 0
        self.fallbacks = 0
        self.prepared_model_state: tuple[LayerState, ...] | None = None
        self.prepared_numerical_context: NumericalContext | None = None
        try:
            with (
                torch.cuda.device(self.device),
                torch.inference_mode(False),
                torch.no_grad(),
            ):
                for layer in self.layers:
                    self.inverse_frequencies[layer] = (
                        model_layers.reference_rope_frequencies(
                            layer.self_attn.head_dim, layer.self_attn.rope_theta
                        ).to(self.device)
                    )
            if self.model_state() is None:
                raise ValueError(
                    "Projection graphs require native eval BF16 versioned weights and unmodified norms"
                )
            else:
                pass
            with (
                torch.cuda.device(self.device),
                torch.random.fork_rng(devices=[self.device.index]),
                DynamicGradMode(),
            ):
                if torch.is_autocast_enabled("cuda"):
                    raise ValueError("Projection graphs do not support autocast")
                else:
                    pass
                for i, layer in enumerate(self.layers):
                    for section in ("prepare", "finish"):
                        hidden_states = torch.zeros(
                            batch_size,
                            (
                                layer.input_layernorm.weight.numel()
                                if section == "prepare"
                                else layer.self_attn.q_size
                            ),
                            device=self.device,
                            dtype=torch.bfloat16,
                        )
                        residual = (
                            None
                            if section == "prepare" and i == 0
                            else torch.zeros(
                                batch_size,
                                layer.input_layernorm.weight.numel(),
                                device=self.device,
                                dtype=torch.bfloat16,
                            )
                        )
                        positions = (
                            torch.zeros(
                                batch_size, device=self.device, dtype=torch.long
                            )
                            if section == "prepare"
                            else None
                        )
                        prepared = PreparedSection(
                            graph=torch.cuda.CUDAGraph(),
                            inputs=(hidden_states, residual, positions),
                        )
                        self.sections[layer, section] = prepared
                        self.capture_stream.wait_stream(self.owner_stream)
                        with torch.cuda.stream(self.capture_stream):
                            for _ in range(2):
                                self.evaluate(layer, section, prepared.inputs)
                        with torch.cuda.graph(
                            prepared.graph, stream=self.capture_stream
                        ):
                            prepared.outputs = self.evaluate(
                                layer, section, prepared.inputs
                            )
                self.capture_stream.synchronize()
                self.prepared_model_state = self.model_state()
                self.prepared_numerical_context = self.numerical_context()
                if self.prepared_model_state is None:
                    raise ValueError(
                        "Projection graph state changed during preparation"
                    )
                else:
                    pass
        except BaseException as original_error:
            try:
                self.close()
            except BaseException as close_error:
                raise original_error from close_error
            raise

    def model_state(self) -> tuple[LayerState, ...] | None:
        states = []
        for layer in self.layers:
            attention = layer.self_attn
            norms = (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                attention.q_norm,
                attention.k_norm,
            )
            if (
                type(layer) is not MossSpeechDecoderLayer
                or layer.training
                or any(
                    type(norm) is not MossSpeechRMSNorm
                    or norm.training
                    or "forward" in vars(norm)
                    or vars(norm).get("_forward_hooks")
                    or vars(norm).get("_forward_pre_hooks")
                    or not math.isfinite(norm.variance_epsilon)
                    or norm.variance_epsilon <= 0
                    for norm in norms
                )
                or "prepare_projection" in vars(layer)
                or "finish_projection" in vars(layer)
            ):
                return None
            else:
                pass
            tensors = []
            weights = (
                *(norm.weight for norm in norms),
                attention.qkv_proj.weight,
                attention.o_proj.weight,
                layer.mlp.gate_up_proj.weight,
                layer.mlp.down_proj.weight,
            )
            for tensor in (*weights, self.inverse_frequencies[layer]):
                if (
                    tensor.is_inference()
                    or tensor.device != self.device
                    or (
                        tensor is not self.inverse_frequencies[layer]
                        and tensor.dtype != torch.bfloat16
                    )
                ):
                    return None
                else:
                    tensors.append(
                        TensorState(
                            identity=id(tensor),
                            pointer=tensor.data_ptr(),
                            version=tensor._version,  # note (Codex): Mutation tracking is required only for captured weights and constants; noqa: leading-underscore.
                            shape=tuple(tensor.shape),
                            stride=tuple(tensor.stride()),
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                    )
            states.append(
                LayerState(
                    identity=id(layer),
                    geometry=(
                        attention.num_heads,
                        attention.num_kv_heads,
                        attention.head_dim,
                        attention.q_size,
                        attention.kv_size,
                        attention.rope_theta,
                    ),
                    norm_epsilon=tuple(norm.variance_epsilon for norm in norms),
                    functions=(
                        id(type(layer).prepare_projection),
                        id(type(layer).finish_projection),
                        *(id(type(norm).forward) for norm in norms),
                        id(model_layers.request_linear),
                        id(model_layers.request_norm),
                        id(model_layers.reference_rope_frequencies),
                        id(torch.nn.functional.linear),
                        id(torch.nn.functional.silu),
                    ),
                    tensors=tuple(tensors),
                )
            )
        return tuple(states)

    def numerical_context(self) -> NumericalContext:
        return NumericalContext(
            gradients=torch.is_grad_enabled(),
            inference_mode=torch.is_inference_mode_enabled(),
            autocast=torch.is_autocast_enabled("cuda"),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            blas_library=str(torch.backends.cuda.preferred_blas_library()),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warnings=torch.is_deterministic_algorithms_warn_only_enabled(),
        )

    def can_replay(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, *, is_decode: bool
    ) -> bool:
        with self.lock:
            if self.closed:
                return False
            elif self.model_state() != self.prepared_model_state:
                self.closed = True
                return False
            else:
                return (
                    is_decode
                    and not torch.is_grad_enabled()
                    and not torch.is_autocast_enabled("cuda")
                    and tuple(hidden_states.shape)
                    == (self.batch_size, self.layers[0].input_layernorm.weight.numel())
                    and hidden_states.dtype == torch.bfloat16
                    and hidden_states.device == self.device
                    and hidden_states.stride()
                    == (self.layers[0].input_layernorm.weight.numel(), 1)
                    and positions.shape == (self.batch_size,)
                    and positions.dtype == torch.long
                    and positions.device == self.device
                    and positions.stride() == (1,)
                    and self.numerical_context() == self.prepared_numerical_context
                    and torch.cuda.current_stream(self.device).cuda_stream
                    == self.owner_stream.cuda_stream
                )

    def evaluate(
        self,
        layer: MossSpeechDecoderLayer,
        section: ProjectionSection,
        inputs: ProjectionInputs,
    ) -> ProjectionOutputs:
        hidden_states, residual, positions = inputs
        if section == "prepare":
            assert positions is not None
            return layer.prepare_projection(
                hidden_states,
                residual,
                positions,
                [1] * self.batch_size,
                self.inverse_frequencies[layer],
            )
        else:
            assert residual is not None
            return layer.finish_projection(
                hidden_states, residual, [1] * self.batch_size
            )

    def replay(
        self,
        layer: MossSpeechDecoderLayer,
        section: ProjectionSection,
        inputs: ProjectionInputs,
    ) -> ProjectionOutputs | None:
        with self.lock:
            prepared = self.sections.get((layer, section))
            if self.closed or prepared is None or prepared.inputs is None:
                self.fallbacks += 1
                return None
            elif any(
                (actual is None) != (static is None)
                or (
                    actual is not None
                    and static is not None
                    and (
                        actual.shape != static.shape
                        or actual.dtype != static.dtype
                        or actual.device != static.device
                        or actual.stride() != static.stride()
                    )
                )
                for actual, static in zip(inputs, prepared.inputs, strict=True)
            ):
                self.fallbacks += 1
                return None
            else:
                assert prepared.outputs is not None
            try:
                with torch.cuda.device(self.device):
                    for actual, static in zip(inputs, prepared.inputs, strict=True):
                        if actual is None or static is None:
                            pass
                        else:
                            static.copy_(actual)
                    prepared.graph.replay()
                    self.replays += 1
                    if section == "prepare":
                        assert len(prepared.outputs) == 4
                        return (
                            prepared.outputs[0].clone(),
                            prepared.outputs[1].clone(),
                            prepared.outputs[2].clone(),
                            (
                                inputs[0]
                                if inputs[1] is None
                                else prepared.outputs[3].clone()
                            ),
                        )
                    else:
                        return prepared.outputs[0].clone(), prepared.outputs[1].clone()
            except BaseException:
                self.closed = True
                raise

    def prepare(
        self,
        layer: MossSpeechDecoderLayer,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        outputs = self.replay(layer, "prepare", (hidden_states, residual, positions))
        if outputs is None:
            return None
        else:
            assert len(outputs) == 4
            return outputs[0], outputs[1], outputs[2], outputs[3]

    def finish(
        self,
        layer: MossSpeechDecoderLayer,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        outputs = self.replay(layer, "finish", (hidden_states, residual, None))
        if outputs is None:
            return None
        else:
            assert len(outputs) == 2
            return outputs[0], outputs[1]

    def close(self) -> None:
        with self.lock:
            if (
                self.closed
                and not self.sections
                and not self.inverse_frequencies
                and not self.layers
            ):
                return
            else:
                pass
            self.closed = True
            first_error: BaseException | None = None
            for stream in (self.owner_stream, self.capture_stream):
                try:
                    stream.synchronize()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                    else:
                        pass
            for prepared in self.sections.values():
                try:
                    prepared.graph.reset()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                    else:
                        pass
                finally:
                    prepared.inputs = None
                    prepared.outputs = None
            self.sections.clear()
            self.inverse_frequencies.clear()
            self.layers = ()
            self.prepared_model_state = None
            self.prepared_numerical_context = None
            if first_error is not None:
                raise first_error
            else:
                pass
