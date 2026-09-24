# SPDX-License-Identifier: Apache-2.0
"""GPU startup mutexes follow physical identity across visibility namespaces."""
from __future__ import annotations

import fcntl
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.utils import gpu_memory


def install_nvml(
    monkeypatch: pytest.MonkeyPatch, identity: str | bytes
) -> SimpleNamespace:
    state = SimpleNamespace(
        identity=identity, handles=[], stopped=0, failure=None, init_failure=None
    )

    def initialize() -> None:
        if state.init_failure:
            raise state.init_failure

    def handle(device: int | str) -> int | str:
        state.handles.append(device)
        return device

    def uuid(device: int | str) -> str | bytes:
        del device
        if state.failure:
            raise state.failure
        return state.identity

    def shutdown() -> None:
        state.stopped += 1

    fake = SimpleNamespace(
        nvmlInit=initialize,
        nvmlShutdown=shutdown,
        nvmlDeviceGetHandleByIndex=handle,
        nvmlDeviceGetHandleByUUID=handle,
        nvmlDeviceGetUUID=uuid,
    )
    monkeypatch.setattr(gpu_memory, "_try_import_pynvml", lambda: fake)
    return state


def test_separate_allocations_with_visible_zero_use_distinct_locks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = install_nvml(monkeypatch, "GPU-first")
    first = gpu_memory.get_gpu_startup_lock_path(
        0, env={"CUDA_VISIBLE_DEVICES": "0"}, base_dir=tmp_path
    )
    state.identity = "GPU-second"
    second = gpu_memory.get_gpu_startup_lock_path(
        0, env={"CUDA_VISIBLE_DEVICES": "0"}, base_dir=tmp_path
    )
    assert first != second
    assert state.stopped == 2


@pytest.mark.parametrize("identity", ["GPU-same", b"GPU-same", "MIG-GPU-parent/1/2"])
def test_device_index_and_uuid_aliases_share_a_lock(
    identity: str | bytes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_nvml(monkeypatch, identity)
    text = identity.decode() if isinstance(identity, bytes) else identity
    by_index = gpu_memory.get_gpu_startup_lock_path(
        0, env={"CUDA_VISIBLE_DEVICES": "3"}, base_dir=tmp_path
    )
    by_uuid = gpu_memory.get_gpu_startup_lock_path(
        0, env={"CUDA_VISIBLE_DEVICES": text}, base_dir=tmp_path
    )
    assert by_index == by_uuid
    assert by_uuid.parent == tmp_path


@pytest.mark.parametrize("failure", ["missing", "init", "query", "empty", "unknown"])
def test_missing_identity_preserves_visible_device_fallback(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = install_nvml(monkeypatch, "GPU-valid")
    if failure == "missing":
        monkeypatch.setattr(gpu_memory, "_try_import_pynvml", lambda: None)
    elif failure == "init":
        state.init_failure = RuntimeError("NVML init unavailable")
    elif failure == "query":
        state.failure = RuntimeError("NVML unavailable")
    elif failure == "empty":
        state.identity = ""
    else:
        state.identity = "unknown"
    path = gpu_memory.get_gpu_startup_lock_path(
        0, env={"CUDA_VISIBLE_DEVICES": "4"}, base_dir=tmp_path
    )
    assert path.name == "sglang_omni_gpu_4_startup.lock"


def test_canonical_mutex_is_exclusive_and_released_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_nvml(monkeypatch, "GPU-mutex")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(gpu_memory.tempfile, "gettempdir", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="factory failed"):
        with gpu_memory.gpu_startup_lock(0) as path:
            assert "GPU-mutex" in path.name
            with path.open("a+") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise RuntimeError("factory failed")
    with path.open("a+") as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(other.fileno(), fcntl.LOCK_UN)


def test_invalid_visible_mapping_fails_before_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = install_nvml(monkeypatch, "GPU-valid")
    with pytest.raises(RuntimeError, match="Invalid GPU"):
        gpu_memory.get_gpu_startup_lock_path(
            1, env={"CUDA_VISIBLE_DEVICES": "0"}, base_dir=tmp_path
        )
    assert not state.handles
