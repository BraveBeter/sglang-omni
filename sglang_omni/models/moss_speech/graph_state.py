# SPDX-License-Identifier: Apache-2.0
"""Captured execution policy and normalization ownership for AR graph families."""

import ctypes
from dataclasses import dataclass

import torch

from sglang_omni.models.moss_speech.model_layers import MossSpeechRMSNorm
from sglang_omni.models.moss_speech.native_norm import NativeNorm


@dataclass(frozen=True, kw_only=True)
class NormalizationState:
    identity: int
    implementation: int
    instance_implementation: int
    epsilon: float
    training: bool
    native_norm: NativeNorm | None
    native_library: ctypes.CDLL | None


def normalization_state(model: torch.nn.Module) -> tuple[NormalizationState, ...]:
    return tuple(
        NormalizationState(
            identity=id(module),
            implementation=id(type(module).forward),
            instance_implementation=id(vars(module).get("forward")),
            epsilon=module.variance_epsilon,
            training=module.training,
            native_norm=module.native_norm,
            native_library=(
                module.native_norm.library if module.native_norm is not None else None
            ),
        )
        for module in model.modules()
        if isinstance(module, MossSpeechRMSNorm)
    )


@dataclass(frozen=True, kw_only=True)
class ExecutionPolicy:
    sdpa: tuple[bool, bool, bool, bool]
    sdpa_priority: tuple[int, ...]
    float32_matmul_precision: str
    matmul_tf32: bool
    bf16_reduced_precision: bool
    blas_library: str
    autocast: bool
    deterministic_algorithms: bool
    deterministic_warnings: bool


def execution_policy() -> ExecutionPolicy:
    cuda = torch.backends.cuda
    return ExecutionPolicy(
        sdpa=(
            cuda.flash_sdp_enabled(),
            cuda.mem_efficient_sdp_enabled(),
            cuda.math_sdp_enabled(),
            cuda.cudnn_sdp_enabled(),
        ),
        sdpa_priority=tuple(
            torch._C._get_sdp_priority_order()
        ),  # noqa: leading-underscore
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=cuda.matmul.allow_tf32,
        bf16_reduced_precision=cuda.matmul.allow_bf16_reduced_precision_reduction,
        blas_library=str(cuda.preferred_blas_library()),
        autocast=torch.is_autocast_enabled("cuda"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warnings=torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def native_norm_calls(model: torch.nn.Module) -> int:
    owners = {
        module.native_norm
        for module in model.modules()
        if isinstance(module, MossSpeechRMSNorm) and module.native_norm is not None
    }
    return sum(owner.calls for owner in owners)


def reset_graph(graph: torch.cuda.CUDAGraph | None, device: torch.device) -> None:
    first_error: BaseException | None = None
    try:
        torch.cuda.synchronize(device)
    except BaseException as error:
        first_error = error
    if graph is not None:
        try:
            graph.reset()
        except BaseException as error:
            if first_error is None:
                first_error = error
            else:
                pass
    else:
        pass
    if first_error is not None:
        raise first_error
    else:
        pass
