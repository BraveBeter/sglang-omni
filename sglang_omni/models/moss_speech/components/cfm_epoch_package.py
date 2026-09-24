"""Load only the explicit qualified CPU metadata binary; never compile at runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import platform
import sys
import sysconfig
from abc import ABCMeta
from pathlib import Path

QUALIFIED_MANIFEST_SHA256 = (
    "802fc3351e7a97c4c324603894218f3c461fa721cbafe53bf1cf48c0f3e7bac5"
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def loaded_library(maps: str, name: str) -> Path:
    paths = set()
    for line in maps.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        path = parts[5]
        base = Path(path.removesuffix(" (deleted)")).name
        if base == name or base.startswith(name + "."):
            require(
                path.startswith("/") and not path.endswith(" (deleted)"),
                "Invalid loaded library",
            )
            paths.add(Path(path).resolve())
    require(len(paths) == 1, "Missing or ambiguous loaded library: " + name)
    return paths.pop()


def validate_abi(expected, actual):
    require(
        json.dumps(expected, sort_keys=True) == json.dumps(actual, sort_keys=True),
        "Unqualified Python/Torch ABI",
    )


def validate_manifest(path):
    path = Path(path).resolve()
    candidate = json.loads(path.read_text())
    if candidate.get("schema_version") == 2:
        from ..native_build import validate_local_package

        manifest = validate_local_package(path, "epoch")
        require(
            manifest["abi"]["implementation"] == "cpython"
            and manifest["abi"]["python_version"][:2] == [3, 10]
            and manifest["abi"]["torch"] == "2.9.1+cu128",
            "Unsupported native epoch build environment",
        )
        return dict(manifest=manifest, library=path.parent / "native.so")
    require(sha(path) == QUALIFIED_MANIFEST_SHA256, "Unqualified native epoch manifest")
    manifest = json.loads(path.read_text())
    require(
        set(manifest["files"]) == {"epoch.cpp", "native.so"},
        "Unexpected package members",
    )
    for name, digest in manifest["files"].items():
        target = (path.parent / name).resolve()
        require(
            not Path(name).is_absolute() and target.is_relative_to(path.parent),
            "Package path escapes directory",
        )
        require(
            target.is_file() and sha(target) == digest,
            "Changed native epoch package file: " + name,
        )
    return dict(manifest=manifest, library=path.parent / "native.so")


_MODULE_METHODS = (
    "named_parameters",
    "named_buffers",
    "_named_members",
    "named_modules",
    "__dict__",
    "__getattribute__",
    "__getattr__",
    "__hash__",
    "__eq__",
)
_TENSOR_KEYS = (
    "__dict__",
    "__getattribute__",
    "__getattr__",
    "__hash__",
    "__eq__",
    "data_ptr",
    "shape",
    "stride",
    "storage_offset",
    "dtype",
    "device",
    "_version",
    "__torch_function__",
    "__torch_dispatch__",
)


def _raw(cls, names):
    return {name: inspect.getattr_static(cls, name, None) for name in names}


def make_binding(torch):
    return dict(
        module_type=torch.nn.Module,
        tensor_type=torch.Tensor,
        parameter_type=torch.nn.Parameter,
        type_meta=type,
        abc_meta=ABCMeta,
        meta_methods=_raw(
            type, ("__getattribute__", "__getattr__", "__hash__", "__eq__")
        ),
        module_methods=_raw(torch.nn.Module, _MODULE_METHODS),
        method_keys=_MODULE_METHODS,
        reserved=("_parameters", "_buffers", "_modules", "training", "rand_noise"),
        tensor_keys=_TENSOR_KEYS,
        tensor_methods=_raw(torch.Tensor, _TENSOR_KEYS),
        parameter_methods=_raw(torch.nn.Parameter, _TENSOR_KEYS),
        tensor_meta_methods=_raw(
            type(torch.Tensor), ("__instancecheck__", "__subclasscheck__")
        ),
        dtype_type=type(torch.float32),
        device_type=torch.device,
        dtype_methods=_raw(type(torch.float32), ("__str__",)),
        device_methods=_raw(torch.device, ("__str__",)),
        dot=".",
        training_name="training",
        noise_suffix=".rand_noise",
    )


class NativeEpochPackage:
    def __init__(self, path):
        require(not os.getenv("LD_PRELOAD"), "LD_PRELOAD is unsupported")
        package = validate_manifest(path)
        manifest = package["manifest"]
        import torch

        actual = dict(
            implementation=sys.implementation.name,
            python_version=list(sys.version_info[:3]),
            soabi=sysconfig.get_config_var("SOABI"),
            machine=platform.machine(),
            torch=torch.__version__,
            cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        )
        validate_abi(manifest["abi"], actual)
        runtime = manifest["runtime"]
        require(
            sha(Path(sys.executable).resolve()) == runtime["python_executable_sha256"],
            "Python executable differs",
        )
        maps = Path("/proc/self/maps").read_text()
        for name, digest in runtime["torch_libraries"].items():
            target = (Path(torch.__file__).resolve().parent / "lib" / name).resolve()
            require(
                loaded_library(maps, name) == target and sha(target) == digest,
                "Active Torch library differs: " + name,
            )
        for name, digest in runtime["system_libraries"].items():
            require(
                sha(loaded_library(maps, name)) == digest,
                "Active system library differs: " + name,
            )
        spec = importlib.util.spec_from_file_location(
            "_epoch_native", package["library"]
        )
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        self.native = native
        self.binding = make_binding(torch)
        self.identity = dict(
            manifest_sha256=sha(Path(path)),
            library_sha256=manifest["files"]["native.so"],
            source_sha256=manifest["files"]["epoch.cpp"],
            abi=actual,
            qualification=manifest["qualification"],
            retains_original_rpath=manifest["build"]["retains_original_rpath"],
            runtime_compilation=False,
        )

    def epoch(self, module, original, *, stats=None, probe=None):
        if self.native is None:
            raise RuntimeError("Native epoch package is closed")
        if stats is None:
            stats = {}
        try:
            result, actual = self.native.snapshot(module, self.binding, probe)
        except self.native.Unsupported as error:
            stats.update(route="original", reason=str(error))
            return original(module)
        stats.update(route="native", **actual)
        return result

    def close(self):
        self.native = self.binding = None
