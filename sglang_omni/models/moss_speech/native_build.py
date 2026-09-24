# SPDX-License-Identifier: Apache-2.0
"""Explicit offline builds and local numerical qualification of native packages.

No compilation occurs during serving. Packages bind the installed source and
validation procedure, their actual binary and the environment used to test it.
"""
from __future__ import annotations

import ctypes
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from .decode_norm_kernel import configure_abi, require, sha

SOURCE = Path(__file__).parent / "kernels"
NORM_FLAGS = ["-std=c++17", "-O3", "-arch=sm_80", "--fmad=false"]
MEMBERS = {
    "norm": ("native.cu", "native.so", "native.ptx"),
    "epoch": ("epoch.cpp", "native.so"),
}


def validate_local_package(path: Path, kind: str) -> dict:
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    require(
        type(value.get("schema_version")) is int and value["schema_version"] == 2,
        "Unsupported local package version",
    )
    require(value.get("kind") == kind, "Wrong native package kind")
    require(
        set(value["files"]) == set(MEMBERS[kind]) | {"validation.json"},
        "Unexpected native package members",
    )
    for name, digest in value["files"].items():
        member = (path.parent / name).resolve()
        require(
            member.is_relative_to(path.parent)
            and member.is_file()
            and sha(member) == digest,
            "Changed native package file: " + name,
        )
    source = "decode_norm_native.cu" if kind == "norm" else "epoch.cpp"
    require(
        value["files"][MEMBERS[kind][0]] == sha(SOURCE / source),
        "Native package source differs from installed source",
    )
    report = json.loads((path.parent / "validation.json").read_text())
    require(
        report.get("validator_sha256") == sha(Path(__file__)),
        "Native package validation procedure changed; rebuild",
    )
    require(
        report.get("kind") == kind
        and report.get("pass") is True
        and report.get("cases") == (36 if kind == "norm" else 8),
        "Missing complete local numerical validation",
    )
    require(
        report.get("files") == {n: value["files"][n] for n in MEMBERS[kind]},
        "Validation does not describe this native binary",
    )
    require(
        value.get("qualification")
        == {
            "scope": "local-numerical-v1",
            "report_sha256": value["files"]["validation.json"],
        },
        "Missing native validation receipt",
    )
    return value


def abi(torch) -> dict:
    return dict(
        implementation=sys.implementation.name,
        python_version=list(sys.version_info[:3]),
        soabi=sysconfig.get_config_var("SOABI"),
        machine=platform.machine(),
        torch=str(torch.__version__),
        cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    )


def epoch_runtime(torch) -> dict:
    from .components.cfm_epoch_package import loaded_library

    maps = Path("/proc/self/maps").read_text()
    return dict(
        python_executable_sha256=sha(Path(sys.executable).resolve()),
        torch_libraries={
            n: sha(loaded_library(maps, n))
            for n in ("libtorch_python.so", "libtorch_cpu.so", "libc10.so")
        },
        system_libraries={
            n: sha(loaded_library(maps, n))
            for n in (
                "libstdc++.so.6",
                "libgcc_s.so.1",
                "libc.so.6",
                "libm.so.6",
                "ld-linux-x86-64.so.2",
            )
        },
    )


def _write_package(out: Path, kind: str, metadata: dict, cases: int) -> Path:
    files = {name: sha(out / name) for name in MEMBERS[kind]}
    report = dict(
        kind=kind, cases=cases, files=files, validator_sha256=sha(Path(__file__))
    )
    report["pass"] = True
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    files["validation.json"] = sha(out / "validation.json")
    manifest = dict(
        schema_version=2,
        kind=kind,
        files=files,
        **metadata,
        qualification=dict(
            scope="local-numerical-v1", report_sha256=files["validation.json"]
        ),
    )
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate_local_package(path, kind)
    return path


def _tool(name: str) -> Path:
    found = shutil.which(name)
    if found is None:
        raise ValueError(f"Compiler not found: {name}")
    return Path(found).resolve()


def qualify_norm(torch, library: Path) -> int:
    """Compare every intermediate and graph replay with independent Torch math."""
    from .decode_norm_kernel import SHAPES, Pointwise

    pointwise = Pointwise.__new__(Pointwise)
    pointwise.torch = torch
    pointwise.lib = ctypes.CDLL(str(library))
    configure_abi(pointwise.lib)
    require(pointwise.lib.norm_abi_version() == 1, "Wrong norm ABI")
    require(
        pointwise.lib.norm_square(None, None, 1, 4096, 4096, 4096, None) == -1,
        "Norm null-pointer control failed",
    )
    cases = 0
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.inference_mode():
        for shape in SHAPES:
            for scale in (0.0, 1e-4, 1.0, 100.0):
                x = (
                    torch.randn(shape, device="cuda", generator=generator) * scale
                ).bfloat16()
                weight = torch.randn(
                    shape[-1], device="cuda", generator=generator
                ).bfloat16()
                for epsilon in (1e-6, 1e-5, 1e-4):

                    def reference():
                        square = x.float().pow(2)
                        mean = square.mean(-1, keepdim=True)
                        added = mean + epsilon
                        inverse = torch.rsqrt(added)
                        scaled = x.float() * inverse
                        rounded = scaled.bfloat16()
                        return dict(
                            square=square,
                            mean=mean,
                            added=added,
                            inverse=inverse,
                            scaled=scaled,
                            rounded=rounded,
                            output=weight * rounded,
                        )

                    layout = dict(square_stride=x.stride(), output_stride=x.stride())
                    expected = reference()
                    actual = pointwise(x, weight, epsilon, layout, debug=True)
                    for key in expected:
                        require(
                            torch.equal(
                                expected[key].view(torch.uint8),
                                actual[key].view(torch.uint8),
                            ),
                            f"Norm numerical mismatch: {shape}, {scale}, {epsilon}, {key}",
                        )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        pointwise(x, weight, epsilon, layout)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            replay = pointwise(x, weight, epsilon, layout)["output"]
                    torch.cuda.current_stream().wait_stream(stream)
                    for delta in (0.0, 0.25):
                        x.add_(delta)
                        graph.replay()
                        require(
                            torch.equal(
                                reference()["output"].view(torch.uint8),
                                replay.view(torch.uint8),
                            ),
                            "Norm graph replay differs after input update",
                        )
                    graph.reset()
                    cases += 1
    return cases


def build_norm(out: Path, nvcc: str = "nvcc") -> Path:
    import torch

    from .decode_norm_kernel import MANIFEST_FIELDS, validate_environment

    validate_environment(
        MANIFEST_FIELDS,
        dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            compute_capability=list(torch.cuda.get_device_capability()),
        ),
    )
    compiler = _tool(nvcc)
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    require(
        re.search(r"\brelease 12\.8\b", version) is not None, "CUDA 12.8 nvcc required"
    )
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE / "decode_norm_native.cu", out / "native.cu")
    # Stable basenames avoid embedding the author's source/build paths.
    for flags, target in [
        (["-shared", "-Xcompiler=-fPIC"], "native.so"),
        (["--ptx"], "native.ptx"),
    ]:
        subprocess.run(
            [str(compiler), *NORM_FLAGS, *flags, "native.cu", "-o", target],
            cwd=out,
            check=True,
        )
    require(
        re.search(r"\bfma\.", (out / "native.ptx").read_text()) is None,
        "Norm compiler introduced fused multiply-add",
    )
    cases = qualify_norm(torch, out / "native.so")
    return _write_package(
        out,
        "norm",
        dict(
            abi_version=1,
            compiler=dict(
                version=version,
                sha256=sha(compiler),
                flags=NORM_FLAGS,
                ptx_no_fma=True,
                host_null_rejected=True,
            ),
            environment=MANIFEST_FIELDS["environment"],
        ),
        cases,
    )


def _epoch_module(library: Path):
    spec = importlib.util.spec_from_file_location("_epoch_native", library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualify_epoch(torch, library: Path) -> int:
    from .components.cfm_epoch_package import make_binding
    from .components.cfm_prefix_cache import tensor_epoch

    native = _epoch_module(library)
    binding = make_binding(torch)
    root = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
    root.register_buffer("scratch", torch.arange(8).reshape(2, 4))
    root.rand_noise = torch.zeros(2)
    root.alias = root[0]
    cases = 0
    for change in (
        lambda: None,
        lambda: root.scratch.add_(1),
        lambda: setattr(root, "rand_noise", torch.ones(3)),
        lambda: root.train(),
        lambda: root.register_buffer("extra", torch.ones(1)),
        lambda: root.__delattr__("alias"),
    ):
        change()
        before = tensor_epoch(root)
        value, stats = native.snapshot(root, binding, None)
        require(
            value == before == tensor_epoch(root) and stats["gil_held"] is True,
            "Native epoch does not match the complete live Python snapshot",
        )
        cases += 1

    class Override(torch.nn.Module):
        def named_parameters(self, *args, **kwargs):
            return super().named_parameters(*args, **kwargs)

    for unsupported in (Override(), root):
        if unsupported is root:
            root.named_buffers = lambda *a, **kw: iter(())
        try:
            native.snapshot(unsupported, binding, None)
        except native.Unsupported:
            cases += 1
        else:
            raise ValueError("Native epoch failed to reject overridden traversal")
    return cases


def build_epoch(out: Path, cxx: str = "g++") -> Path:
    import torch

    require(
        sys.implementation.name == "cpython"
        and sys.version_info[:2] == (3, 10)
        and str(torch.__version__) == "2.9.1+cu128"
        and platform.machine() == "x86_64",
        "Native epoch requires CPython 3.10, Torch 2.9.1+cu128 and x86_64",
    )
    compiler = _tool(cxx)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SOURCE / "epoch.cpp", out / "epoch.cpp")
    base = Path(torch.__file__).resolve().parent
    flags = [
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
    ]
    subprocess.run(
        [
            str(compiler),
            *flags,
            "-I" + sysconfig.get_path("include"),
            "-I" + str(base / "include"),
            "-I" + str(base / "include/torch/csrc/api/include"),
            "epoch.cpp",
            "-L" + str(base / "lib"),
            "-ltorch_python",
            "-ltorch_cpu",
            "-lc10",
            "-o",
            "native.so",
        ],
        cwd=out,
        check=True,
    )
    cases = qualify_epoch(torch, out / "native.so")
    return _write_package(
        out,
        "epoch",
        dict(
            abi=abi(torch),
            runtime=epoch_runtime(torch),
            build=dict(
                compiler_sha256=sha(compiler),
                compiler_version=subprocess.check_output(
                    [str(compiler), "--version"], text=True
                ).splitlines()[0],
                options=flags,
                retains_original_rpath=False,
                runtime_compilation=False,
            ),
        ),
        cases,
    )
