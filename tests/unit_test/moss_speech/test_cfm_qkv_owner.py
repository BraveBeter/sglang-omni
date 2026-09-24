"""Default-off production owner admission and cleanup contracts."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as Obj

import pytest

from sglang_omni.models.moss_speech.components import cfm_full_graph as api


def make(enabled=False):
    cfm = Obj(solve_euler=lambda *a, **k: "native")
    bank = Obj(
        flow=Obj(decoder=cfm),
        binding=Obj(shapes=(1392, 1424, 1500), valid=True),
        active_length=None,
        mode="cached",
        entries={},
    )
    return api.FullCfmGraphBank(cfm, bank, qkv_multistream=enabled)


def test_explicit_boolean_default_off_has_no_branch_resources():
    owner = make()
    assert owner._qkv_branches is None
    assert owner._branch_plans == {}
    assert not owner._multistream(1392)
    owner.close()
    owner.close()
    assert owner._closed
    with pytest.raises((ValueError, RuntimeError)):
        make(1)


def test_only_independently_qualified_shapes_select_parallel():
    owner = make(True)
    assert all(owner._multistream(n) for n in (1392, 1424))
    assert not any(owner._multistream(n) for n in (1384, 1400, 1492, 1500))
    assert owner._qkv_branches is None
    owner.close()


def test_runtime_plans_select_branches_without_mutating_serial_controls():
    owner = make(True)
    original, branch = object(), object()
    owner._tail_plans[(1392, "layer", "k")] = original
    owner._branch_plans[(1392, "layer", "k")] = branch
    owner._tail_plans[(1500, "layer", "k")] = original
    assert owner._runtime_plans(1392)[(1392, "layer", "k")] is branch
    assert owner._tail_plans[(1392, "layer", "k")] is original
    assert owner._runtime_plans(1500)[(1500, "layer", "k")] is original


def test_close_graphs_before_branches_before_main_context(monkeypatch):
    from contextlib import nullcontext

    import torch

    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    order = []
    owner = make(True)
    owner._graphs[1392] = Obj(close=lambda: order.append("graph"))
    owner._qkv_branches = Obj(close=lambda: order.append("branches"))
    owner._lt_context = Obj(device="cuda:0", close=lambda: order.append("main"))
    owner.close()
    assert order == ["graph", "branches", "main"]
    assert not owner._branch_plans and owner._qkv_branches is None


def test_failed_branch_sync_keeps_context_and_owners(monkeypatch):
    from contextlib import nullcontext

    import torch

    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    owner = make(True)
    owner._lt_context = Obj(device="cuda:0", close=lambda: None)
    marker = object()
    owner._branch_plans[(1392, "layer", "k")] = marker

    def fail():
        raise RuntimeError("sync failed")

    owner._qkv_branches = Obj(close=fail)
    with pytest.raises(RuntimeError, match="sync failed"):
        owner.close()
    assert owner._branch_plans[(1392, "layer", "k")] is marker
    assert owner._qkv_branches is not None and not owner._closed


def test_actual_replay_admission_remains_thread_local_without_rebuild():
    owner = make(True)
    owner._prepared = True
    owner.bank.active_length = 1392
    called = []
    owner._graphs[1392] = Obj(
        matches=lambda *a: True, run=lambda values: called.append(values) or "graph"
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert (
            executor.submit(owner.solve_euler, *range(6), streaming=True).result()
            == "graph"
        )
    assert len(called) == 1 and owner._qkv_branches is None


def test_branch_startup_rejects_different_plan_or_original_tail(monkeypatch):
    from contextlib import nullcontext

    import torch

    from sglang_omni.models.moss_speech.components import cfm_direct_lt as lt

    monkeypatch.setattr(torch.cuda, "stream", lambda *a: nullcontext())
    owner = make(True)
    branch_context = object()
    owner._qkv_branches = Obj(
        streams={"k": object()},
        contexts={"k": branch_context},
        call=lambda kind, plan, value: plan(value),
        bind=lambda: Obj(calls=0),
    )
    key = (1392, 0, "layer", "k")
    owner._tail_plans[(1392, "layer", "k")] = Obj(
        describe=lambda: {"algorithm": "original"}
    )
    value = torch.zeros(2, 1392, 256)
    output = torch.zeros(2, 1392, 512)
    created = []

    class Plan:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

        def describe(self):
            return {"algorithm": "wrong"}

        def __call__(self, value):
            return output[:, 1280:]

    monkeypatch.setattr(lt, "DirectLtProjection", Plan)
    with pytest.raises(RuntimeError, match="algorithm"):
        owner._collect_branch(key, Obj(weight=object(), bias=None), value, output)
    assert not owner._branch_keys and created[0]["context"] is branch_context
    owner._branch_plans.clear()
    Plan.describe = lambda self: {"algorithm": "original"}
    Plan.__call__ = lambda self, value: torch.ones(2, 112, 512)
    with pytest.raises(RuntimeError, match="tail projection"):
        owner._collect_branch(key, Obj(weight=object(), bias=None), value, output)
    assert not owner._branch_keys
    owner._branch_plans.clear()
    Plan.__call__ = lambda self, value: output[:, 1280:]
    owner._collect_branch(key, Obj(weight=object(), bias=None), value, output)
    assert owner._branch_keys == {key}
    with pytest.raises(RuntimeError, match="Duplicate"):
        owner._collect_branch(key, Obj(), value, output)


def test_default_off_sync_failure_keeps_main_plan_and_open_state(monkeypatch):
    from contextlib import nullcontext

    import torch

    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    owner = make(False)
    marker = object()

    def fail():
        raise RuntimeError("main sync failed")

    owner._lt_context = Obj(device="cuda:0", close=fail)
    owner._tail_plans[(1500, "layer", "q")] = marker
    with pytest.raises(RuntimeError, match="sync failed"):
        owner.close()
    assert not owner._closed and owner._tail_plans[(1500, "layer", "q")] is marker
    assert owner._qkv_branches is None


def test_independent_body_counts_and_capture_refs_are_actual():
    owner = make(True)
    workspace, normalized = object(), object()
    owner._qkv_branches = Obj(owners=(workspace,))
    owner._tail_blocks[(1392, 0, "layer")] = Obj(capture_owners=[normalized])
    assert owner._branch_owners(1392) == (workspace, normalized)
    assert owner._branch_owners(1500) == ()
    assert owner._expected_body(1392)["qkv_fork_join_calls"] == 560
    assert owner._expected_body(1500)["qkv_fork_join_calls"] == 0
    assert "qkv_fork_join_calls" not in make(False)._expected_body(1392)


def test_branch_close_explicitly_selects_owner_device_before_provider(monkeypatch):
    from contextlib import contextmanager

    import torch

    current = ["cuda:0"]

    @contextmanager
    def device(value):
        before = current[0]
        current[0] = value
        try:
            yield
        finally:
            current[0] = before

    monkeypatch.setattr(torch.cuda, "device", device)
    owner = make(True)

    def check():
        assert current[0] == "cuda:1"

    owner._lt_context = Obj(device="cuda:1", close=check)
    owner._qkv_branches = Obj(close=check)
    owner.close()
    assert owner._closed and current[0] == "cuda:0"
