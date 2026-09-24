# SPDX-License-Identifier: Apache-2.0
"""A rebuilt package needs current sources, complete validation and intact bytes."""
import json
import shutil

import pytest

from sglang_omni.models.moss_speech import native_build as build
from sglang_omni.models.moss_speech.decode_norm_kernel import sha


@pytest.fixture(params=["norm", "epoch"])
def package(tmp_path, request):
    kind = request.param
    root = tmp_path / "package"
    root.mkdir()
    source = build.SOURCE / ("decode_norm_native.cu" if kind == "norm" else "epoch.cpp")
    for name in build.MEMBERS[kind]:
        (root / name).write_bytes(
            source.read_bytes() if name == build.MEMBERS[kind][0] else name.encode()
        )
    path = build._write_package(root, kind, {}, 36 if kind == "norm" else 8)
    return path, kind


def test_relocation_preserves_package_identity(package, tmp_path):
    path, kind = package
    other = tmp_path / "relocated"
    shutil.copytree(path.parent, other)
    assert build.validate_local_package(path, kind) == build.validate_local_package(
        other / path.name, kind
    )
    assert str(tmp_path) not in path.read_text()


@pytest.mark.parametrize("member", ["native.so", "validation.json"])
def test_changed_binary_or_validation_is_rejected(package, member):
    path, kind = package
    (path.parent / member).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        build.validate_local_package(path, kind)


def test_unqualified_build_or_stale_validator_is_rejected(package):
    path, kind = package
    report = path.parent / "validation.json"
    value = json.loads(report.read_text())
    value["validator_sha256"] = "0" * 64
    report.write_text(json.dumps(value))
    manifest = json.loads(path.read_text())
    manifest["files"][report.name] = sha(report)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="procedure changed"):
        build.validate_local_package(path, kind)


def test_package_cannot_escape_via_symlink(package, tmp_path):
    path, kind = package
    binary = path.parent / "native.so"
    other = tmp_path / "outside.so"
    binary.rename(other)
    binary.symlink_to(other)
    with pytest.raises(ValueError, match="Changed"):
        build.validate_local_package(path, kind)


def test_source_and_validation_must_match_installed_procedure(
    package, monkeypatch, tmp_path
):
    path, kind = package
    source = tmp_path / "changed-source"
    source.mkdir()
    (source / ("decode_norm_native.cu" if kind == "norm" else "epoch.cpp")).write_bytes(
        b"different"
    )
    monkeypatch.setattr(build, "SOURCE", source)
    with pytest.raises(ValueError, match="source differs"):
        build.validate_local_package(path, kind)


@pytest.mark.parametrize("field,value", [("pass", False), ("cases", 0), ("files", {})])
def test_failed_or_incomplete_numerical_report_cannot_be_admitted(
    package, field, value
):
    path, kind = package
    report_path = path.parent / "validation.json"
    report = json.loads(report_path.read_text())
    report[field] = value
    report_path.write_text(json.dumps(report))
    manifest = json.loads(path.read_text())
    manifest["files"]["validation.json"] = sha(report_path)
    manifest["qualification"]["report_sha256"] = sha(report_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="numerical validation|does not describe"):
        build.validate_local_package(path, kind)
