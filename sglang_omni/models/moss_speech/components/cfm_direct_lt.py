"""Private FP32 affine provider used only while preparing CFM CUDA graphs.

This uses the public cuBLASLt 12.8.4 ABI, without interposing on PyTorch. The
restricted algorithm still requires native full/tail numerical qualification by
the graph owner. Neither API success nor an AlgoCheck proves bitwise equality.

A projection returns borrowed storage, overwritten by its next invocation. Its
caller must consume it immediately, and return owned final CFM outputs. Graphs
retain every projection/context until their last replay has finished. Parameter
version, training, source and backend invalidation belongs to the graph owner.
"""

from __future__ import annotations

import ctypes as C
import os
import threading
from pathlib import Path
from typing import Any

_CONFIG = {0: 0, 1: 14, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
_FF_FIRST_CONFIG = {0: 0, 1: 20, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
_FF_FIRST_TAIL_CONFIG = {0: 0, 1: 15, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
_QKV_1392_CONFIG = {0: 0, 1: 16, 2: 3, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
_QKV_1424_CONFIG = {0: 0, 1: 16, 2: 3, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
_QKV_1400_CONFIG = {0: 0, 1: 16, 2: 3, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
_QKV_1500_CONFIG = {0: 0, 1: 16, 2: 2, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
_WORKSPACE_BYTES = 1048576
_TAIL_ROWS = (224, 240, 288, 440)
_ROWS = (*_TAIL_ROWS, 2784, 2800, 2848, 3000)


class _Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class _HeuristicResult(C.Structure):
    _fields_ = [
        ("algo", _Algo),
        ("workspaceSize", C.c_size_t),
        ("state", C.c_int),
        ("wavesCount", C.c_float),
        ("reserved", C.c_int * 4),
    ]


def _get_torch() -> Any:
    import torch

    return torch


def _checked(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation}: status {status}")


def _scalar_get(function: Any, owner: Any, attribute: int, ctype: Any) -> Any:
    value, written = ctype(), C.c_size_t()
    _checked(
        function(owner, attribute, C.byref(value), C.sizeof(value), C.byref(written)),
        "GetAttribute",
    )
    if written.value != C.sizeof(value):
        raise RuntimeError("GetAttribute returned an unexpected size")
    return value.value


def _find_loaded_library(maps: str) -> Path:
    paths = set()
    for line in maps.splitlines():
        if "libcublasLt.so." not in line:
            continue
        parts = line.split(maxsplit=5)
        if (
            len(parts) != 6
            or not parts[5].startswith("/")
            or parts[5].endswith(" (deleted)")
        ):
            raise RuntimeError("Invalid loaded cuBLASLt path")
        paths.add(Path(parts[5]).resolve())
    if len(paths) != 1:
        raise RuntimeError("Exactly one already-loaded cuBLASLt library is required")
    return paths.pop()


def _open_loaded_library() -> tuple[Any, Path]:
    if os.environ.get("LD_PRELOAD"):
        raise RuntimeError("Direct Lt cannot run with LD_PRELOAD")
    maps = Path("/proc/self/maps").read_text()
    if any(
        name in maps for name in ("interposer.so", "lt_interposer", "liblt_capture")
    ):
        raise RuntimeError("An Lt interposer is loaded")
    path = _find_loaded_library(maps)
    return C.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_NOLOAD | os.RTLD_LOCAL), path


def _bind_library(lib: Any) -> None:
    p, i, s = C.c_void_p, C.c_int, C.c_size_t
    pp = C.POINTER(p)
    signatures = {
        "cublasLtGetVersion": ([], s),
        "cublasLtCreate": ([pp], i),
        "cublasLtDestroy": ([p], i),
        "cublasLtMatmulDescCreate": ([pp, i, i], i),
        "cublasLtMatmulDescDestroy": ([p], i),
        "cublasLtMatmulDescSetAttribute": ([p, i, p, s], i),
        "cublasLtMatmulDescGetAttribute": ([p, i, p, s, C.POINTER(s)], i),
        "cublasLtMatrixLayoutCreate": ([pp, i, C.c_uint64, C.c_uint64, C.c_int64], i),
        "cublasLtMatrixLayoutDestroy": ([p], i),
        "cublasLtMatrixLayoutGetAttribute": ([p, i, p, s, C.POINTER(s)], i),
        "cublasLtMatmulAlgoInit": ([p, i, i, i, i, i, i, i, C.POINTER(_Algo)], i),
        "cublasLtMatmulAlgoConfigSetAttribute": ([C.POINTER(_Algo), i, p, s], i),
        "cublasLtMatmulAlgoConfigGetAttribute": (
            [C.POINTER(_Algo), i, p, s, C.POINTER(s)],
            i,
        ),
        "cublasLtMatmulAlgoCapGetAttribute": (
            [C.POINTER(_Algo), i, p, s, C.POINTER(s)],
            i,
        ),
        "cublasLtMatmulAlgoCheck": (
            [p, p, p, p, p, p, C.POINTER(_Algo), C.POINTER(_HeuristicResult)],
            i,
        ),
        "cublasLtMatmul": (
            [p, p, p, p, p, p, p, p, p, p, p, p, C.POINTER(_Algo), p, s, p],
            i,
        ),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result


class DirectLtContext:
    """One device, handle and workspace; all launches share a prepared stream.

    The owner serializes graph preparation/replay. ``prepare_stream`` is an
    explicit startup handoff, outside capture; it waits for the previous stream.
    CPU threads may change. Replay contains no Python/Lt provider calls.
    """

    def __init__(self, device: Any) -> None:
        self.torch = _get_torch()
        self.device = self.torch.device(device)
        self.device_index = self.device.index
        if self.device.type != "cuda" or self.device_index is None:
            raise ValueError("An explicit CUDA device index is required")
        self.lock = threading.RLock()
        self.closed = False
        self.failed = False
        self.plans: list[DirectLtProjection] = []
        self.workspace = None
        self.handle = C.c_void_p()
        self.stream = None
        self._check_device()
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot initialize direct Lt during capture")
        if (
            self.torch.cuda.get_device_capability(self.device) != (8, 0)
            or self.torch.cuda.get_device_name(self.device) != "NVIDIA A800-SXM4-80GB"
        ):
            raise RuntimeError("Direct Lt is restricted to the tested A800 device")
        if C.sizeof(_Algo) != 64 or C.sizeof(_HeuristicResult) != 96:
            raise RuntimeError("Unsupported cuBLASLt ABI sizes")
        self.lib, self.path = _open_loaded_library()
        _bind_library(self.lib)
        self.version = int(self.lib.cublasLtGetVersion())
        if self.version != 120804:
            raise RuntimeError("cuBLASLt 12.8.4 is required")
        try:
            _checked(self.lib.cublasLtCreate(C.byref(self.handle)), "Create")
            if not self.handle.value:
                raise RuntimeError("Null Lt handle")
            self.workspace = self.torch.empty(
                _WORKSPACE_BYTES, dtype=self.torch.uint8, device=self.device
            )
            self.stream = self.torch.cuda.current_stream(self.device)
        except BaseException:
            if self.handle.value:
                _checked(self.lib.cublasLtDestroy(self.handle), "Destroy")
                self.handle = C.c_void_p()
            self.workspace = None
            self.closed = True
            raise

    def _check_device(self) -> None:
        if self.torch.cuda.current_device() != self.device_index:
            raise RuntimeError("Changed current CUDA device")

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("Direct Lt context is closed")
        if self.failed:
            raise RuntimeError("Direct Lt context previously failed")
        self._check_device()

    def _check_stream(self) -> int:
        self._check_open()
        pointer = int(self.torch.cuda.current_stream(self.device).cuda_stream)
        if self.stream is None or pointer != int(self.stream.cuda_stream):
            raise RuntimeError("Unprepared current CUDA stream")
        return pointer

    def prepare_stream(self) -> None:
        """Bind the current stream after a startup-only serial handoff."""
        with self.lock:
            self._check_open()
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Cannot prepare a stream during capture")
            current = self.torch.cuda.current_stream(self.device)
            if self.stream is not None and int(current.cuda_stream) != int(
                self.stream.cuda_stream
            ):
                self.stream.synchronize()
            self.stream = current

    @property
    def owners(self) -> tuple[Any, ...]:
        return () if self.closed or self.workspace is None else (self.workspace,)

    def describe(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "library": str(self.path),
            "version": self.version,
            "workspace_bytes": _WORKSPACE_BYTES,
            "workspace_pointer": (
                self.workspace.data_ptr() if self.workspace is not None else None
            ),
            "prepared_stream": (
                int(self.stream.cuda_stream) if self.stream is not None else None
            ),
        }

    def status(self) -> dict[str, Any]:
        return {
            "closed": self.closed,
            "failed": self.failed,
            "projections": len(self.plans),
            "calls": sum(plan.calls for plan in self.plans),
        }

    def close(self) -> None:
        """Destroy only after all graph users have stopped and synchronized."""
        with self.lock:
            if self.closed:
                return
            self._check_device()
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Cannot close direct Lt during capture")
            # Replay may use a different stream from the last capture. Once-only
            # device synchronization is intentional, and never occurs in run().
            self.torch.cuda.synchronize(self.device)
            errors = []
            for plan in reversed(self.plans):
                try:
                    plan._close_synced()
                except Exception as exc:
                    errors.append(str(exc))
            self.plans.clear()
            if self.handle.value:
                try:
                    _checked(self.lib.cublasLtDestroy(self.handle), "Destroy")
                except Exception as exc:
                    errors.append(str(exc))
            self.handle = C.c_void_p()
            self.workspace = None
            self.closed = True
            if errors:
                raise RuntimeError("; ".join(errors))


class DirectLtProjection:
    """Fixed biased affine projection with persistent input and borrowed output.

    ``rows`` counts flattened rows for the explicitly admitted startup shapes.
    Every full/tail pair still requires native numerical qualification. Parameters must retain their
    bits and storage until the graph owner invalidates and closes its graphs.
    The default ``output`` profile keeps K512/1024 -> 256 with tile14. Only an
    explicit ``ff_first`` profile admits K256 -> 1024: tile20 for native-full
    startup controls and tile15 for the finite tail-row set.
    """

    def __init__(
        self,
        weight: Any,
        bias: Any,
        rows: int,
        *,
        context: DirectLtContext | None = None,
        profile: str = "output",
    ) -> None:
        torch = _get_torch()
        allowed_rows = _ROWS
        if profile == "output":
            output_features, allowed_k, config = 256, (512, 1024), _CONFIG
        elif profile == "ff_first":
            output_features, allowed_k = 1024, (256,)
            config = _FF_FIRST_TAIL_CONFIG if rows in _TAIL_ROWS else _FF_FIRST_CONFIG
        elif profile == "qkv_1392":
            output_features, allowed_k, config = 512, (256,), _QKV_1392_CONFIG
            allowed_rows = (224, 2784)
        elif profile == "qkv_1424":
            output_features, allowed_k, config = 512, (256,), _QKV_1424_CONFIG
            allowed_rows = (288, 2848)
        elif profile == "qkv_1400":
            output_features, allowed_k, config = 512, (256,), _QKV_1400_CONFIG
            allowed_rows = (240, 2800)
        elif profile == "qkv_1500":
            output_features, allowed_k, config = 512, (256,), _QKV_1500_CONFIG
            allowed_rows = (440, 3000)
        else:
            raise ValueError("Unsupported direct Lt profile")
        requires_bias = profile in ("output", "ff_first")
        if requires_bias:
            if (
                bias is None
                or tuple(bias.shape) != (output_features,)
                or bias.dtype != torch.float32
                or bias.device != weight.device
                or tuple(bias.stride()) != (1,)
                or bias.data_ptr() <= 0
                or bias.data_ptr() % 4
            ):
                raise ValueError("Unsupported FP32 Lt bias")
        elif bias is not None:
            raise ValueError("QKV profiles require absent bias")
        shape = tuple(weight.shape)
        if (
            len(shape) != 2
            or shape[0] != output_features
            or shape[1] not in allowed_k
            or rows not in allowed_rows
            or weight.dtype != torch.float32
            or weight.device.type != "cuda"
            or tuple(weight.stride()) != (shape[1], 1)
            or weight.data_ptr() <= 0
        ):
            raise ValueError("Unsupported FP32 Lt parameters or rows")
        self.weight, self.bias, self.rows, self.k = weight, bias, rows, shape[1]
        self.profile, self.output_features = profile, output_features
        self._algorithm_config = dict(config)
        self.weight_pointer = weight.data_ptr()
        self.bias_pointer = bias.data_ptr() if bias is not None else None
        self.capabilities: dict[str, int] = {}
        self.context = (
            context if context is not None else DirectLtContext(weight.device)
        )
        self.owns_context = context is None
        self.closed = False
        self.calls = 0
        self.desc = C.c_void_p()
        self.layouts: list[Any] = []
        self.input = None
        self.output = None
        self.algo = _Algo()
        self.alpha, self.beta = C.c_float(1.0), C.c_float(0.0)
        with self.context.lock:
            try:
                self.context._check_open()
                if self.context.device != weight.device:
                    raise ValueError("Projection/context device mismatch")
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Cannot initialize a projection during capture")
                self._prepare()
                self.context.plans.append(self)
            except BaseException:
                self._close_synced()  # No launch has occurred for this projection.
                if self.owns_context:
                    self.context.close()
                raise

    def _prepare(self) -> None:
        context, k, rows = self.context, self.k, self.rows
        output_features = self.output_features
        lib, torch = context.lib, context.torch
        # POINTER_MODE_HOST, TRANSA=T, TRANSB/TRANSC=N, zero bias batch
        # stride and SM target. QKV uses DEFAULT and never installs a bias.
        attributes = {2: 0, 3: 1, 4: 0, 5: 0, 7: 1, 10: 0, 15: 0}
        if self.bias is not None:
            attributes.update({7: 4, 8: self.bias_pointer})
        _checked(
            lib.cublasLtMatmulDescCreate(C.byref(self.desc), 68, 0), "MatmulDescCreate"
        )
        for attribute, value in attributes.items():
            ctype = (
                C.c_void_p
                if attribute == 8
                else C.c_int64 if attribute == 10 else C.c_int
            )
            scalar = ctype(value)
            _checked(
                lib.cublasLtMatmulDescSetAttribute(
                    self.desc, attribute, C.byref(scalar), C.sizeof(scalar)
                ),
                "MatmulDescSetAttribute",
            )
        expected = {0: 68, 1: 0, **attributes, 8: self.bias_pointer, 26: -1}
        for attribute, value in expected.items():
            ctype = (
                C.c_void_p
                if attribute == 8
                else C.c_int64 if attribute == 10 else C.c_int
            )
            if (
                _scalar_get(
                    lib.cublasLtMatmulDescGetAttribute, self.desc, attribute, ctype
                )
                != value
            ):
                raise RuntimeError("FP32/epilogue descriptor readback changed")
        self.descriptor = {str(key): value for key, value in expected.items()}
        self.layout_values = []
        for height, width, ld in (
            (k, output_features, k),
            (k, rows, k),
            (output_features, rows, output_features),
        ):
            layout = C.c_void_p()
            _checked(
                lib.cublasLtMatrixLayoutCreate(C.byref(layout), 0, height, width, ld),
                "MatrixLayoutCreate",
            )
            self.layouts.append(layout)
            values = {0: 0, 1: 0, 2: height, 3: width, 4: ld, 5: 1, 6: 0}
            for attribute, value in values.items():
                ctype = (
                    C.c_uint64
                    if attribute in (2, 3)
                    else C.c_int64 if attribute in (4, 6) else C.c_int
                )
                if (
                    _scalar_get(
                        lib.cublasLtMatrixLayoutGetAttribute, layout, attribute, ctype
                    )
                    != value
                ):
                    raise RuntimeError("Matrix layout readback changed")
            self.layout_values.append(
                {str(key): value for key, value in values.items()}
            )
        _checked(
            lib.cublasLtMatmulAlgoInit(
                context.handle, 68, 0, 0, 0, 0, 0, 0, C.byref(self.algo)
            ),
            "AlgoInit",
        )
        if self.profile in ("qkv_1392", "qkv_1400", "qkv_1424", "qkv_1500"):
            self.capabilities = {
                "split_k_support": _scalar_get(
                    lib.cublasLtMatmulAlgoCapGetAttribute,
                    C.byref(self.algo),
                    0,
                    C.c_int32,
                ),
                "reduction_scheme_mask": _scalar_get(
                    lib.cublasLtMatmulAlgoCapGetAttribute,
                    C.byref(self.algo),
                    1,
                    C.c_uint32,
                ),
            }
            if (
                not self.capabilities["split_k_support"]
                or not self.capabilities["reduction_scheme_mask"] & 1
            ):
                raise RuntimeError("Fixed QKV split-K/INPLACE capabilities unavailable")
        for attribute, value in self._algorithm_config.items():
            if attribute:
                scalar = (C.c_uint16 if attribute >= 7 else C.c_uint32)(value)
                _checked(
                    lib.cublasLtMatmulAlgoConfigSetAttribute(
                        C.byref(self.algo), attribute, C.byref(scalar), C.sizeof(scalar)
                    ),
                    "AlgoConfigSetAttribute",
                )
        self.config = {}
        for attribute, value in self._algorithm_config.items():
            ctype = C.c_uint16 if attribute >= 7 else C.c_uint32
            actual = _scalar_get(
                lib.cublasLtMatmulAlgoConfigGetAttribute,
                C.byref(self.algo),
                attribute,
                ctype,
            )
            if actual != value:
                raise RuntimeError("Fixed algorithm configuration changed")
            self.config[str(attribute)] = actual
        a, b, c = self.layouts
        result = _HeuristicResult()
        _checked(
            lib.cublasLtMatmulAlgoCheck(
                context.handle,
                self.desc,
                a,
                b,
                c,
                c,
                C.byref(self.algo),
                C.byref(result),
            ),
            "AlgoCheck",
        )
        _checked(result.state, "AlgoCheck state")
        self.required_workspace = int(result.workspaceSize)
        if not 0 <= self.required_workspace <= _WORKSPACE_BYTES:
            raise RuntimeError("Required workspace exceeds the fixed 1 MiB allocation")
        self.alignment = [
            _scalar_get(
                lib.cublasLtMatmulAlgoCapGetAttribute,
                C.byref(self.algo),
                16 + attr,
                C.c_uint32,
            )
            for attr in range(4)
        ]
        self.input = torch.empty(
            (2, rows // 2, k), dtype=torch.float32, device=context.device
        )
        self.output = torch.empty(
            (2, rows // 2, output_features), dtype=torch.float32, device=context.device
        )
        for pointer, alignment in zip(
            (
                self.weight_pointer,
                self.input.data_ptr(),
                self.output.data_ptr(),
                self.output.data_ptr(),
            ),
            self.alignment,
        ):
            if alignment <= 0 or pointer <= 0 or pointer % alignment:
                raise RuntimeError("Misaligned direct Lt operand")
        self.input_pointer, self.output_pointer = (
            self.input.data_ptr(),
            self.output.data_ptr(),
        )
        self.workspace_pointer = context.workspace.data_ptr()

    def run(self, value: Any) -> Any:
        """Copy input and launch on the prepared current stream; return borrowed storage."""
        context = self.context
        with context.lock:
            if self.closed:
                raise RuntimeError("Direct Lt projection is closed")
            stream_pointer = context._check_stream()
            source_strides = ((self.rows // 2 * self.k, self.k, 1),)
            if self.rows in _TAIL_ROWS and self.k in (256, 512):
                # The caller may pass an admitted original full-stride tail view.
                # copy_ gathers directly into the unchanged compact Lt input.
                source_strides += (((self.rows // 2 + 1280) * self.k, self.k, 1),)
            if (
                tuple(value.shape) != (2, self.rows // 2, self.k)
                or value.dtype != context.torch.float32
                or value.device != context.device
                or tuple(value.stride()) not in source_strides
            ):
                raise ValueError(
                    "Changed direct Lt input shape, layout, precision or device"
                )
            if (
                self.weight.data_ptr() != self.weight_pointer
                or (self.bias.data_ptr() if self.bias is not None else None)
                != self.bias_pointer
            ):
                raise RuntimeError("Changed direct Lt parameter storage")
            if (
                self.input.data_ptr() != self.input_pointer
                or self.output.data_ptr() != self.output_pointer
                or context.workspace.data_ptr() != self.workspace_pointer
            ):
                raise RuntimeError("Changed direct Lt buffer storage")
            self.input.copy_(value)
            a, b, c = self.layouts
            p = C.c_void_p
            try:
                _checked(
                    context.lib.cublasLtMatmul(
                        context.handle,
                        self.desc,
                        C.byref(self.alpha),
                        p(self.weight_pointer),
                        a,
                        p(self.input_pointer),
                        b,
                        C.byref(self.beta),
                        p(self.output_pointer),
                        c,
                        p(self.output_pointer),
                        c,
                        C.byref(self.algo),
                        p(self.workspace_pointer),
                        _WORKSPACE_BYTES,
                        p(stream_pointer),
                    ),
                    "Matmul",
                )
            except BaseException:
                context.failed = True
                raise
            self.calls += 1
            return self.output

    __call__ = run

    @property
    def owners(self) -> tuple[Any, ...]:
        if self.closed:
            return ()
        result = (self.input, self.output, self.weight)
        return result + (self.bias,) if self.bias is not None else result

    def describe(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "output_features": self.output_features,
            "k": self.k,
            "rows": self.rows,
            "config": dict(self.config),
            "capabilities": dict(self.capabilities),
            "descriptor": dict(self.descriptor),
            "layouts": [dict(value) for value in self.layout_values],
            "required_workspace": self.required_workspace,
            "alignment": list(self.alignment),
            "algo_hex": bytes(self.algo).hex(),
            "borrowed_output": True,
        }

    def status(self) -> dict[str, Any]:
        return {
            "closed": self.closed,
            "calls": self.calls,
            "owns_context": self.owns_context,
        }

    def _close_synced(self) -> None:
        if self.closed:
            return
        errors = []
        for layout in reversed(self.layouts):
            try:
                _checked(
                    self.context.lib.cublasLtMatrixLayoutDestroy(layout),
                    "MatrixLayoutDestroy",
                )
            except Exception as exc:
                errors.append(str(exc))
        if self.desc.value:
            try:
                _checked(
                    self.context.lib.cublasLtMatmulDescDestroy(self.desc),
                    "MatmulDescDestroy",
                )
            except Exception as exc:
                errors.append(str(exc))
        self.layouts.clear()
        self.desc = C.c_void_p()
        self.input = self.output = self.weight = self.bias = None
        self.closed = True
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self) -> None:
        with self.context.lock:
            if self.closed:
                return
            if self.owns_context:
                self.context.close()
                return
            self.context._check_device()
            if self.context.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Cannot close a projection during capture")
            self.context.torch.cuda.synchronize(self.context.device)
            try:
                self._close_synced()
            finally:
                self.context.plans.remove(self)
