# SPDX-License-Identifier: Apache-2.0
"""Explicit, relocatable package loader for qualified capture-only norm kernels."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SHAPES = {(1, 4096): 82, (1, 32, 128): 40, (1, 8, 128): 40}
QUALIFIED_FILES = {
    "native.cu": "73071c84be350ba5c61f51dd5bbb24bb9fedde5a0823fe9697938a8411f3a168",
    "native.so": "2ee910b25b9a570d58a0513762be4872fa729e892f922b3dc508701c8fa30d03",
    "native.ptx": "c7f6c4e20f914d9f3d65b53e361dab282d91f0cd87d23adfa5e550cb428e2ec4",
    "native.sass": "665c1f5825d6030739396bbe9bbd2cb9979ecb299af67a6182cf970305ce87bd",
}
MANIFEST_FIELDS = {
    "schema_version": 1,
    "abi_version": 1,
    "compiler": {
        "version": "12.8.61",
        "sha256": "3aadf006f53dd288a372b2c4fabd46df1caa25aacd2e60964d7a26af25e91517",
        "flags": ["-std=c++17", "-O3", "-arch=sm_80", "--fmad=false"],
        "ptx_no_fma": True,
        "host_null_rejected": True,
    },
    "environment": {
        "torch": "2.9.1+cu128",
        "cuda": "12.8",
        "compute_capability": [8, 0],
    },
    "qualification": {
        "scope": "kernel_only_not_public_runtime",
        "capture_sha256": "4dce631e6efb02aa785c4d7bd41ae52fd6636623ca2c7eb842980e2382a79740",
        "micro_sha256": "2ba6e725d9d56affdd6df1f22fbb5f35585c1ce114125b9db573612855321830",
        "complete_ar_sha256": "d47b7d235f4a6e2ee6c39114a567383b12f6da80ff90c1257cb2c3015a184eb2",
        "normal_ar_sha256": "c3b309815a132f636aa6e7d595a21da6f66797c9a2f4a904f9a55b6c9ca7fa8e",
    },
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def validate_manifest(path: Path | str) -> dict:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") == 2:
        from .native_build import NORM_FLAGS, validate_local_package

        manifest = validate_local_package(path, "norm")
        require(
            type(manifest.get("abi_version")) is int and manifest["abi_version"] == 1,
            "Unexpected norm ABI version",
        )
        require(
            manifest["compiler"]["flags"] == NORM_FLAGS
            and manifest["compiler"]["ptx_no_fma"] is True
            and manifest["compiler"]["host_null_rejected"] is True,
            "Missing norm compiler controls",
        )
        validate_environment(MANIFEST_FIELDS, manifest["environment"])
        return dict(
            manifest=manifest,
            manifest_sha256=sha(path),
            library=str(path.parent / "native.so"),
        )
    require(
        set(manifest) == set(MANIFEST_FIELDS) | {"files"},
        "Unknown or missing norm package field",
    )
    for key, value in MANIFEST_FIELDS.items():
        require(
            json.dumps(manifest[key], sort_keys=True)
            == json.dumps(value, sort_keys=True),
            "Unqualified norm package field: " + key,
        )
    require(
        type(manifest["schema_version"]) is int
        and type(manifest["abi_version"]) is int,
        "Strict norm package versions required",
    )
    require(
        manifest["compiler"]["ptx_no_fma"] is True
        and manifest["compiler"]["host_null_rejected"] is True,
        "Missing original compiler/ABI controls",
    )
    require(
        manifest["files"] == QUALIFIED_FILES, "Unqualified or relocated file identity"
    )
    for name, digest in QUALIFIED_FILES.items():
        target = (path.parent / name).resolve()
        require(
            not Path(name).is_absolute() and target.is_relative_to(path.parent),
            "Norm package file escapes its directory",
        )
        require(
            target.is_file() and sha(target) == digest,
            "Norm package bytes changed: " + name,
        )
    return dict(
        manifest=manifest,
        manifest_sha256=sha(path),
        library=str((path.parent / "native.so").resolve()),
    )


def validate_environment(manifest: dict, actual: dict) -> None:
    require(
        json.dumps(actual, sort_keys=True)
        == json.dumps(manifest["environment"], sort_keys=True),
        "Unqualified norm Torch/CUDA/device environment",
    )


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_call(spec: dict) -> None:
    shape = tuple(spec["shape"])
    stride = spec["stride"]
    require(shape in SHAPES and len(stride) == len(shape), "Unsupported finite shape")
    require(
        stride[-1] == 1
        and all(type(v) is int and v > 0 for v in stride)
        and (len(shape) == 2 or stride[-2] == shape[-1]),
        "Unsupported original layout",
    )
    require(
        spec["dtype"] == spec["weight_dtype"] == "torch.bfloat16",
        "Only original BF16 supported",
    )
    require(spec["weight_shape"] == [shape[-1]], "Unexpected norm weight")
    require(
        str(spec["device"]).startswith("cuda:")
        and spec["device"] == spec["weight_device"],
        "Cross-device input",
    )
    require(
        type(spec["epsilon"]) in (int, float)
        and math.isfinite(spec["epsilon"])
        and spec["epsilon"] > 0,
        "Invalid epsilon",
    )


def configure_abi(lib: object) -> None:
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    lib.norm_abi_version.argtypes = []
    lib.norm_abi_version.restype = integer
    lib.norm_square.argtypes = [
        pointer,
        pointer,
        integer,
        integer,
        integer,
        integer,
        pointer,
    ]
    lib.norm_square.restype = integer
    lib.norm_finish.argtypes = [
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
    lib.norm_finish.restype = integer


class Pointwise:
    def __init__(self, torch: Any, manifest_path: Path | str):
        self.torch = torch
        bound = validate_manifest(manifest_path)
        validate_environment(
            bound["manifest"],
            dict(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                compute_capability=list(torch.cuda.get_device_capability()),
            ),
        )
        self.identity = dict(
            manifest_sha256=bound["manifest_sha256"],
            library_sha256=bound["manifest"]["files"]["native.so"],
            qualification=bound["manifest"]["qualification"],
        )
        self.lib = ctypes.CDLL(bound["library"])
        configure_abi(self.lib)
        require(self.lib.norm_abi_version() == 1, "Unexpected norm library ABI")
        require(
            self.lib.norm_square(None, None, 1, 4096, 4096, 4096, None) == -1,
            "Norm library null ABI control failed",
        )

    def __call__(
        self,
        x: object,
        weight: object,
        epsilon: float,
        layout: dict,
        debug: bool = False,
    ) -> dict:
        t = self.torch
        validate_call(
            dict(
                shape=list(x.shape),
                stride=list(x.stride()),
                dtype=str(x.dtype),
                weight_shape=list(weight.shape),
                weight_dtype=str(weight.dtype),
                device=str(x.device),
                weight_device=str(weight.device),
                epsilon=epsilon,
            )
        )
        require(
            t.is_inference_mode_enabled(),
            "Local numerical experiment requires inference mode; no production autograd claim",
        )
        require(
            t.cuda.current_device() == x.device.index, "Current CUDA device differs"
        )
        require(weight.stride() == (1,), "Weight stride differs")
        groups = 1 if x.ndim == 2 else x.shape[-2]
        width = x.shape[-1]
        xs = width if x.ndim == 2 else x.stride(-2)
        square = t.empty_strided(
            tuple(x.shape),
            tuple(layout["square_stride"]),
            dtype=t.float32,
            device=x.device,
        )
        output = t.empty_strided(
            tuple(x.shape),
            tuple(layout["output_stride"]),
            dtype=t.bfloat16,
            device=x.device,
        )
        ss = width if square.ndim == 2 else square.stride(-2)
        os = width if output.ndim == 2 else output.stride(-2)
        stream = t.cuda.current_stream(x.device).cuda_stream
        result = self.lib.norm_square(
            x.data_ptr(), square.data_ptr(), groups, width, xs, ss, stream
        )
        require(result == 0, "CUDA square launch failed: " + str(result))
        mean = square.mean(-1, keepdim=True)
        require(mean.is_contiguous(), "Native mean output is not contiguous")
        added = inverse = scaled = rounded = None
        if debug:
            added = t.empty_like(mean)
            inverse = t.empty_like(mean)
            scaled = t.empty_like(square)
            rounded = t.empty_like(output)

        def ptr(value):
            return None if value is None else value.data_ptr()

        result = self.lib.norm_finish(
            x.data_ptr(),
            mean.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            groups,
            width,
            xs,
            os,
            epsilon,
            stream,
            ptr(added),
            ptr(inverse),
            ptr(scaled),
            ptr(rounded),
        )
        require(result == 0, "CUDA finish launch failed: " + str(result))
        return dict(
            square=square,
            mean=mean,
            added=added,
            inverse=inverse,
            scaled=scaled,
            rounded=rounded,
            output=output,
        )


class Engine:
    def __init__(self, torch: Any, manifest_path: Path | str):
        self.pointwise = Pointwise(torch, manifest_path)
        self.identity = self.pointwise.identity

    def __call__(self, x: Any, weight: Any, epsilon: float) -> dict:
        strides = []
        size = 1
        for dim in reversed(tuple(x.shape)):
            strides.append(size)
            size *= dim
        stride = list(reversed(strides))
        return self.pointwise(
            x,
            weight,
            epsilon,
            dict(square_stride=stride, output_stride=stride),
            debug=False,
        )
