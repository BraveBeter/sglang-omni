# SPDX-License-Identifier: Apache-2.0
"""One prepared CUDA graph for the native streaming CFM Euler solver."""

import threading
from dataclasses import dataclass
from typing import Protocol

import torch

from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.decoder import (
    CausalConditionalDecoder,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow_matching import (
    CausalConditionalCFM,
)

SolverInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class EulerSolver(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor | None,
        cond: torch.Tensor | None,
        streaming: bool = False,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, kw_only=True)
class TensorState:
    name: str
    identity: int
    pointer: int
    version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, kw_only=True)
class ModelState:
    modules: tuple[tuple[str, int, bool, int, int], ...]
    tensors: tuple[TensorState, ...]
    guidance_rate: float
    chunk_mel_frames: int
    left_chunks: int


@dataclass(frozen=True, kw_only=True)
class NumericalContext:
    float32_matmul_precision: str
    cuda_matmul_tf32: bool
    cudnn_enabled: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cudnn_tf32: bool
    deterministic_algorithms: bool
    deterministic_warnings: bool
    flash_attention: bool
    memory_efficient_attention: bool
    math_attention: bool
    cudnn_attention: bool


class MossSpeechCFMGraph:
    """Capture once at startup; replay matching calls on one owned stream."""

    def __init__(self, decoder: CausalConditionalCFM, *, mel_length: int) -> None:
        if (
            type(decoder) is not CausalConditionalCFM
            or type(decoder.estimator) is not CausalConditionalDecoder
            or "solve_euler" in vars(decoder)
            or isinstance(mel_length, bool)
            or not 100 <= mel_length <= 4000
            or mel_length % 100
        ):
            raise ValueError(
                "CFM graphs require a plain native solver and 100..4000 mel frames in steps of 100"
            )
        else:
            pass
        parameters = tuple(decoder.parameters())
        if (
            not parameters
            or parameters[0].device.type != "cuda"
            or any(
                parameter.device != parameters[0].device
                or parameter.dtype != torch.float32
                or parameter.is_inference()
                for parameter in parameters
            )
            or any(module.training for module in decoder.modules())
            or decoder.inference_cfg_rate != 0.7
            or decoder.estimator.static_chunk_size != 100
            or decoder.estimator.num_decoding_left_chunks != -1
            or torch.is_autocast_enabled("cuda")
        ):
            raise ValueError(
                "CFM graphs require CUDA FP32 eval weights, native chunk25 guidance, and no autocast"
            )
        else:
            pass
        self.decoder = decoder
        self.device = parameters[0].device
        self.mel_length = mel_length
        self.lock = threading.RLock()
        self.original_solver: EulerSolver = decoder.solve_euler
        self.installed_solver: EulerSolver | None = None
        self.owner_stream = torch.cuda.current_stream(self.device)
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.inputs: SolverInputs | None = None
        self.output: torch.Tensor | None = None
        self.closed = False
        self.replays = 0
        self.fallbacks = 0
        try:
            with (
                torch.cuda.device(self.device),
                torch.random.fork_rng(devices=[self.device.index]),
                torch.inference_mode(),
                torch.backends.cudnn.flags(
                    enabled=torch.backends.cudnn.enabled,
                    benchmark=False,
                    deterministic=True,
                    allow_tf32=torch.backends.cudnn.allow_tf32,
                ),
            ):
                x = torch.zeros(
                    1, 80, mel_length, device=self.device, dtype=torch.float32
                )
                t_span = torch.linspace(
                    0, 1, 11, device=self.device, dtype=torch.float32
                )
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
                self.inputs = (
                    x,
                    t_span,
                    torch.zeros_like(x),
                    torch.ones(
                        1, 1, mel_length, device=self.device, dtype=torch.float32
                    ),
                    torch.zeros(1, 80, device=self.device, dtype=torch.float32),
                    torch.zeros(
                        1, mel_length, 80, device=self.device, dtype=torch.float32
                    ).transpose(1, 2),
                )
                self.capture_stream.wait_stream(self.owner_stream)
                with torch.cuda.stream(self.capture_stream):
                    for _ in range(3):
                        self.original_solver(*self.inputs, streaming=True)
                self.owner_stream.wait_stream(self.capture_stream)
                self.prepared_model_state = self.model_state()
                self.prepared_numerical_context = self.numerical_context()
                if self.prepared_model_state is None:
                    raise ValueError(
                        "CFM graph weights and buffers must track tensor versions"
                    )
                else:
                    pass
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.capture_stream):
                    self.output = self.original_solver(*self.inputs, streaming=True)
                self.capture_stream.synchronize()
        except BaseException:
            self.close()
            raise
        self.installed_solver = self.solve_euler
        decoder.solve_euler = self.installed_solver

    def model_state(self) -> ModelState | None:
        tensors = []
        for name, tensor in (
            *self.decoder.named_parameters(),
            *self.decoder.named_buffers(),
        ):
            if tensor.is_inference():
                return None
            else:
                tensors.append(
                    TensorState(
                        name=name,
                        identity=id(tensor),
                        pointer=tensor.data_ptr(),
                        version=tensor._version,  # note (Codex): PyTorch mutation counter; noqa: leading-underscore.
                        shape=tuple(tensor.shape),
                        stride=tuple(tensor.stride()),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )
        return ModelState(
            modules=tuple(
                (
                    name,
                    id(module),
                    module.training,
                    id(getattr(module.forward, "__func__", module.forward)),
                    id(vars(module).get("processor")),
                )
                for name, module in self.decoder.named_modules()
            ),
            tensors=tuple(tensors),
            guidance_rate=self.decoder.inference_cfg_rate,
            chunk_mel_frames=self.decoder.estimator.static_chunk_size,
            left_chunks=self.decoder.estimator.num_decoding_left_chunks,
        )

    def numerical_context(self) -> NumericalContext:
        return NumericalContext(
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_enabled=torch.backends.cudnn.enabled,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warnings=torch.is_deterministic_algorithms_warn_only_enabled(),
            flash_attention=torch.backends.cuda.flash_sdp_enabled(),
            memory_efficient_attention=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_attention=torch.backends.cuda.math_sdp_enabled(),
            cudnn_attention=torch.backends.cuda.cudnn_sdp_enabled(),
        )

    def can_replay(self, inputs: SolverInputs, *, streaming: bool) -> bool:
        if (
            self.closed
            or self.graph is None
            or self.output is None
            or self.inputs is None
            or not streaming
            or self.device.type != "cuda"
            or not torch.is_inference_mode_enabled()
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or any(
                actual.device != prepared.device
                or actual.dtype != prepared.dtype
                or actual.shape != prepared.shape
                or actual.stride() != prepared.stride()
                for actual, prepared in zip(inputs, self.inputs, strict=True)
            )
            or any(module.training for module in self.decoder.modules())
            or self.model_state() != self.prepared_model_state
            or self.numerical_context() != self.prepared_numerical_context
        ):
            return False
        else:
            return (
                torch.cuda.current_stream(self.device).cuda_stream
                == self.owner_stream.cuda_stream
            )

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor | None,
        cond: torch.Tensor | None,
        streaming: bool = False,
    ) -> torch.Tensor:
        with self.lock:
            if spks is None or cond is None:
                self.fallbacks += 1
                return self.original_solver(x, t_span, mu, mask, spks, cond, streaming)
            else:
                inputs = (x, t_span, mu, mask, spks, cond)
            if not self.can_replay(inputs, streaming=streaming):
                self.fallbacks += 1
                return self.original_solver(*inputs, streaming=streaming)
            else:
                assert self.inputs is not None
                assert self.graph is not None
                assert self.output is not None
            try:
                with torch.cuda.device(self.device):
                    for prepared, actual in zip(self.inputs, inputs, strict=True):
                        prepared.copy_(actual)
                    self.graph.replay()
                    self.replays += 1
                    # note (Codex): A session keeps a mel-tail view beyond the next replay.
                    return self.output.clone()
            except BaseException:
                self.closed = True
                raise

    def close(self) -> None:
        with self.lock:
            if (
                self.closed
                and self.graph is None
                and self.inputs is None
                and self.installed_solver is None
            ):
                return
            else:
                pass
            if (
                self.installed_solver is not None
                and vars(self.decoder).get("solve_euler") is self.installed_solver
            ):
                vars(self.decoder).pop("solve_euler")
            else:
                pass
            self.installed_solver = None
            self.closed = True
            try:
                self.owner_stream.synchronize()
                self.capture_stream.synchronize()
            finally:
                try:
                    if self.graph is not None:
                        self.graph.reset()
                    else:
                        pass
                finally:
                    self.graph = None
                    self.inputs = None
                    self.output = None
