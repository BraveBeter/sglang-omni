# SPDX-License-Identifier: Apache-2.0
"""Explicit CUDA pointwise library with the original Torch mean reduction."""

import ctypes
import math
from pathlib import Path

import torch


class NativeNorm:
    """Keep a plain C ABI library alive for request-local BF16 outputs."""

    def __init__(self, library_path: str) -> None:
        self.library = ctypes.CDLL(str(Path(library_path).resolve(strict=True)))
        self.library.norm_abi_version.argtypes = []
        self.library.norm_abi_version.restype = ctypes.c_int
        self.library.norm_cuda_version.argtypes = []
        self.library.norm_cuda_version.restype = ctypes.c_int
        compiled_cuda = self.library.norm_cuda_version()
        actual_cuda = torch.version.cuda
        if (
            self.library.norm_abi_version() != 1
            or actual_cuda is None
            or tuple(map(int, actual_cuda.split(".")[:2]))
            != (compiled_cuda // 1000, (compiled_cuda % 1000) // 10)
            or torch.cuda.get_device_capability() != (8, 0)
        ):
            raise ValueError(
                "Native norm requires matching CUDA compiler/runtime and SM80"
            )
        else:
            pass
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.library.norm_square.argtypes = [
            pointer,
            pointer,
            integer,
            integer,
            integer,
            integer,
            pointer,
        ]
        self.library.norm_square.restype = integer
        self.library.norm_finish.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            integer,
            integer,
            integer,
            integer,
            ctypes.c_float,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
        ]
        self.library.norm_finish.restype = integer
        self.calls = 0

    def apply(
        self, hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
    ) -> torch.Tensor | None:
        if (
            torch.is_grad_enabled()
            or hidden_states.device.type != "cuda"
            or weight.device != hidden_states.device
            or hidden_states.device.index != torch.cuda.current_device()
            or hidden_states.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or tuple(hidden_states.shape) not in ((1, 4096), (1, 32, 128), (1, 8, 128))
            or not hidden_states.is_contiguous()
            or tuple(weight.shape) != (hidden_states.shape[-1],)
            or not weight.is_contiguous()
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            return None
        else:
            pass
        width = hidden_states.shape[-1]
        groups = hidden_states.numel() // width
        square = torch.empty_like(hidden_states, dtype=torch.float32)
        output = torch.empty_like(hidden_states)
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        result = self.library.norm_square(
            hidden_states.data_ptr(),
            square.data_ptr(),
            groups,
            width,
            width,
            width,
            stream,
        )
        if result != 0:
            raise RuntimeError(f"Native norm square launch failed: {result}")
        else:
            pass
        mean = square.mean(-1, keepdim=True)
        result = self.library.norm_finish(
            hidden_states.data_ptr(),
            mean.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            groups,
            width,
            width,
            width,
            epsilon,
            stream,
            None,
            None,
            None,
            None,
        )
        if result != 0:
            raise RuntimeError(f"Native norm finish launch failed: {result}")
        else:
            self.calls += 1
        return output
