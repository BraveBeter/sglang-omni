# SPDX-License-Identifier: Apache-2.0
"""Finite full-affine Lt plans with owned outputs and no prefix or graph cache."""

import ctypes
import logging
import os
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
FULL_LENGTHS = (1392, 1400, 1424, 1500)
PROJECTIONS = ((512, 256), (1024, 256), (256, 1024), (256, 512))


@dataclass(kw_only=True)
class PreparedProjection:
    context: ctypes.c_void_p
    pointer: ctypes.c_void_p
    weight: torch.Tensor
    bias: torch.Tensor | None
    weight_pointer: int
    bias_pointer: int


class DirectLtOwner:
    """One device/default stream and workspace, closed after all launches finish."""

    def __init__(self, library_path: str) -> None:
        if (
            os.environ.get("LD_PRELOAD")
            or torch.version.cuda is None
            or torch.version.cuda.split(".")[0] != "13"
        ):
            raise ValueError("Direct Lt requires CUDA13 without an interposed runtime")
        else:
            pass
        self.library = ctypes.CDLL(str(Path(library_path).resolve(strict=True)))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        for name in ("lt_abi_version", "lt_header_version", "lt_loaded_version"):
            function = getattr(self.library, name)
            function.argtypes, function.restype = [], integer
        if (
            self.library.lt_abi_version() != 1
            or self.library.lt_header_version() != 130101
            or self.library.lt_loaded_version() != 130101
        ):
            raise ValueError(
                "Direct Lt requires the qualified 13.1.1 headers and runtime"
            )
        else:
            pass
        self.library.lt_context_create.argtypes = [ctypes.POINTER(pointer), pointer]
        self.library.lt_context_create.restype = integer
        self.library.lt_context_destroy.argtypes = [pointer]
        self.library.lt_context_destroy.restype = integer
        self.library.lt_projection_create.argtypes = [
            pointer,
            pointer,
            pointer,
            integer,
            integer,
            integer,
            ctypes.POINTER(pointer),
        ]
        self.library.lt_projection_create.restype = integer
        self.library.lt_projection_destroy.argtypes = [pointer, pointer]
        self.library.lt_projection_destroy.restype = integer
        self.library.lt_run.argtypes = [pointer, pointer, pointer, pointer, pointer]
        self.library.lt_run.restype = integer
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Direct Lt must be initialized outside capture")
        else:
            pass
        self.device = torch.cuda.current_device()
        self.cuda_stream = torch.cuda.current_stream()
        self.stream = self.cuda_stream.cuda_stream
        self.lock = threading.RLock()
        self.runtime_thread: int | None = None
        self.closed = False
        self.calls = 0
        self.installed = 0
        self.pointer = pointer()
        status = self.library.lt_context_create(ctypes.byref(self.pointer), self.stream)
        if status != 0:
            raise RuntimeError(f"Direct Lt initialization failed: {status}")
        else:
            self.finalizer = weakref.finalize(
                self, self.library.lt_context_destroy, self.pointer
            )

    def apply(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        cache: dict[int, PreparedProjection],
    ) -> torch.Tensor | None:
        if (
            torch.is_grad_enabled()
            or type(value) is not torch.Tensor
            or value.device.type != "cuda"
            or value.dtype != torch.float32
            or value.device.index != self.device
            or type(weight) is not torch.nn.Parameter
            or weight.dtype != torch.float32
            or weight.device != value.device
            or weight.ndim != 2
            or (weight.shape[1], weight.shape[0]) not in PROJECTIONS
            or not weight.is_contiguous()
            or weight.data_ptr() % 256
            or value.ndim != 3
            or value.shape[0] != 2
            or value.shape[1] not in FULL_LENGTHS
            or value.shape[2] != weight.shape[1]
            or not value.is_contiguous()
            or value.data_ptr() % 256
            or (bias is None and weight.shape[0] != 512)
            or (
                bias is not None
                and (
                    type(bias) is not torch.nn.Parameter
                    or weight.shape[0] == 512
                    or bias.dtype != torch.float32
                    or bias.device != value.device
                    or tuple(bias.shape) != (weight.shape[0],)
                    or not bias.is_contiguous()
                    or bias.data_ptr() % 16
                )
            )
            or torch.is_autocast_enabled("cuda")
            or torch.backends.cuda.matmul.allow_tf32
            or torch.cuda.current_device() != self.device
            or torch.cuda.current_stream().cuda_stream != self.stream
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        else:
            pass
        with self.lock:
            if self.closed:
                return None
            else:
                pass
            thread_id = threading.get_ident()
            if self.runtime_thread is None:
                self.runtime_thread = thread_id
            elif self.runtime_thread != thread_id:
                return None
            else:
                pass
            length = value.shape[1]
            bias_pointer = bias.data_ptr() if bias is not None else 0
            prepared = cache.get(length)
            if prepared is not None and prepared.context is not self.pointer:
                cache.pop(length)
                prepared = None
            else:
                pass
            if prepared is not None and (
                prepared.weight is not weight
                or prepared.bias is not bias
                or prepared.weight_pointer != weight.data_ptr()
                or prepared.bias_pointer != bias_pointer
            ):
                status = self.library.lt_projection_destroy(
                    self.pointer, prepared.pointer
                )
                if status != 0:
                    raise RuntimeError(f"Direct Lt plan replacement failed: {status}")
                else:
                    cache.pop(length)
                    prepared = None
            else:
                pass
            if prepared is None:
                pointer = ctypes.c_void_p()
                status = self.library.lt_projection_create(
                    self.pointer,
                    weight.data_ptr(),
                    bias_pointer,
                    length * 2,
                    weight.shape[1],
                    weight.shape[0],
                    ctypes.byref(pointer),
                )
                if status != 0:
                    raise RuntimeError(f"Direct Lt plan preparation failed: {status}")
                else:
                    prepared = PreparedProjection(
                        context=self.pointer,
                        pointer=pointer,
                        weight=weight,
                        bias=bias,
                        weight_pointer=weight.data_ptr(),
                        bias_pointer=bias_pointer,
                    )
                    cache[length] = prepared
            else:
                pass
            output = torch.empty(
                (*value.shape[:-1], weight.shape[0]),
                dtype=value.dtype,
                device=value.device,
            )
            status = self.library.lt_run(
                self.pointer,
                prepared.pointer,
                value.data_ptr(),
                output.data_ptr(),
                self.stream,
            )
            if status != 0:
                raise RuntimeError(f"Direct Lt launch failed: {status}")
            else:
                self.calls += 1
            return output

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            else:
                self.closed = True
            status = self.finalizer()
            if status != 0:
                raise RuntimeError(f"Direct Lt shutdown failed: {status}")
            else:
                logger.info(
                    f"MOSS CFM DirectLt closed after {self.calls} launches across {self.installed} projections"
                )


class DirectLtLinear(torch.nn.Linear):
    def __init__(self, source: torch.nn.Linear, owner: DirectLtOwner) -> None:
        torch.nn.Module.__init__(self)
        self.training = source.training
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight
        self.bias = source.bias
        self.owner = owner
        self.plans: dict[int, PreparedProjection] = {}

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training:
            output = self.owner.apply(value, self.weight, self.bias, self.plans)
            if output is not None:
                return output
            else:
                pass
        else:
            pass
        return torch.nn.functional.linear(value, self.weight, self.bias)


def install_direct_lt(estimator: torch.nn.Module, library_path: str) -> DirectLtOwner:
    device = next(estimator.parameters()).device
    with torch.cuda.device(device):
        owner = DirectLtOwner(library_path)
    try:
        for parent in estimator.modules():
            for name, module in list(parent.named_children()):
                if (
                    type(module) is torch.nn.Linear
                    and (module.in_features, module.out_features) in PROJECTIONS
                    and not module._forward_hooks  # noqa: leading-underscore  # PyTorch hook registry.
                    and not module._forward_pre_hooks  # noqa: leading-underscore  # PyTorch hook registry.
                ):
                    setattr(parent, name, DirectLtLinear(module, owner))
                    owner.installed += 1
                else:
                    pass
        if owner.installed == 0:
            raise ValueError("No plain CFM projections qualify for Direct Lt")
        else:
            return owner
    except BaseException:
        owner.close()
        raise
