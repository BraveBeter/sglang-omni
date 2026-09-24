"""Concrete proof/layout and inference-tensor replay boundary regressions."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang_omni.models.moss_speech.components import cfm_full_graph as api
from sglang_omni.models.moss_speech.components.cfm_graph_lifetime import ReplayLease
from sglang_omni.models.moss_speech.components.cosyvoice_flow.cfm_attention import (
    new_full_valid_mask,
)


def values():
    return dict(
        x=torch.ones(1, 80, 16),
        mask=torch.ones(1, 1, 16),
        mu=torch.ones(1, 80, 16),
        t_span=torch.linspace(0, 1, 11),
        spks=torch.ones(1, 80),
        cond=torch.ones(1, 80, 16),
    )


def entry(monkeypatch):
    tensors = values()
    tensors["mask"], proof = new_full_valid_mask(tensors["mask"], 16)
    result = api._ShapeGraph.__new__(api._ShapeGraph)
    prefix = torch.zeros(2, 8, 8, 64)
    result.parent_entries = (((16, 0, "layer"), prefix),)
    result.owner = SimpleNamespace(
        cfm=object(), bank=SimpleNamespace(entries=dict(result.parent_entries))
    )
    result.closed = False
    result.context = ("qualified",)
    result.device = tensors["x"].device
    result.stream = 7
    result.layouts = {k: api.tensor_layout(t) for k, t in tensors.items()}
    monkeypatch.setattr(api, "execution_context", lambda cfm: ("qualified",))
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7)
    )
    return result, tensors, proof


def test_real_mask_provenance_shape_backend_and_prefix_identity(monkeypatch):
    graph, tensors, proof = entry(monkeypatch)
    assert graph.matches(tensors, True, proof)
    assert not graph.matches(tensors, False, proof)
    assert not graph.matches(tensors, True, None)
    assert not graph.matches(tensors, True, SimpleNamespace(matches=lambda t: True))
    changed = dict(tensors, mask=tensors["mask"].clone())
    assert not graph.matches(changed, True, proof)
    for key, value in (
        ("x", torch.zeros(2, 80, 16)),
        ("t_span", torch.zeros(10)),
        ("mu", tensors["mu"].double()),
    ):
        assert not graph.matches(dict(tensors, **{key: value}), True, proof)
    monkeypatch.setattr(api, "execution_context", lambda cfm: ("changed",))
    assert not graph.matches(tensors, True, proof)
    monkeypatch.setattr(api, "execution_context", lambda cfm: ("qualified",))
    key, prefix = graph.parent_entries[0]
    graph.owner.bank.entries[key] = prefix.clone()
    assert not graph.matches(tensors, True, proof)


def test_changed_caller_stream_falls_back_before_replay(monkeypatch):
    graph, tensors, proof = entry(monkeypatch)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=8)
    )
    assert not graph.matches(tensors, True, proof)


def test_inference_input_copies_owned_writable_output_and_held_samples(monkeypatch):
    graph = api._ShapeGraph.__new__(api._ShapeGraph)
    source = values()
    graph.device = torch.device("cpu")
    with torch.inference_mode():
        graph.inputs = {k: v.clone() for k, v in source.items()}
        graph.output = torch.zeros_like(source["x"])
    graph.lease = ReplayLease(
        tuple(graph.inputs.values()) + (graph.output,), api.tensor_storage, stream=7
    )
    calls = []

    def replay():
        calls.append(True)
        graph.output.copy_(graph.inputs["x"] + graph.inputs["mu"])

    graph.graph = SimpleNamespace(replay=replay)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7)
    )
    first = graph.run(source)
    assert not first.is_inference()
    first.add_(3)
    source["x"].add_(1)
    second = graph.run(source)
    assert torch.equal(first, torch.full_like(first, 5))
    assert torch.equal(second, torch.full_like(second, 3))
    assert first.data_ptr() != second.data_ptr() != graph.output.data_ptr()
    assert calls == [True, True]


def test_public_and_observer_positional_order_matches_native_solver():
    import inspect

    from sglang_omni.models.moss_speech.components.cosyvoice_flow.flow_matching import (
        CausalConditionalCFM,
    )

    native = tuple(inspect.signature(CausalConditionalCFM.solve_euler).parameters)
    assert (
        tuple(inspect.signature(api.FullCfmGraphBank.solve_euler).parameters) == native
    )
    assert tuple(inspect.signature(api.FullCfmGraphBank._observe).parameters) == native

    class Solver:
        def solve_euler(
            self, x, t_span, mu, mask, spks, cond, streaming=False, full_valid_mask=None
        ):
            return (x, t_span, mu, mask, spks, cond)

    cfm = Solver()
    bank = SimpleNamespace(
        flow=SimpleNamespace(decoder=cfm),
        binding=SimpleNamespace(shapes=(1500,), valid=True),
        active_length=None,
        mode="cached",
        entries={},
    )
    graph = api.FullCfmGraphBank(cfm, bank)
    expected = tuple(object() for _ in range(6))
    assert graph.solve_euler(*expected) == expected
    assert graph._observe(*expected) == expected


def test_close_uses_direct_lt_device_even_with_different_caller_device(monkeypatch):
    class Solver:
        def solve_euler(self, *args, **kwargs):
            return None

    cfm = Solver()
    bank = SimpleNamespace(flow=SimpleNamespace(decoder=cfm))
    graph = api.FullCfmGraphBank(cfm, bank)
    active = []
    from contextlib import contextmanager

    @contextmanager
    def device_scope(device):
        active.append(device)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(torch.cuda, "device", device_scope)
    selected = torch.device("cuda:1")
    closed = []

    def close():
        assert active == [selected]
        closed.append(True)

    graph._lt_context = SimpleNamespace(device=selected, close=close)
    graph.close()
    assert closed == [True] and graph._closed


def test_close_cannot_overtake_pending_replay_admission():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    class Solver:
        def solve_euler(self, *args, **kwargs):
            return "eager"

    cfm = Solver()
    bank = SimpleNamespace(
        flow=SimpleNamespace(decoder=cfm),
        binding=SimpleNamespace(valid=True),
        active_length=1500,
        mode="cached",
    )
    graph = api.FullCfmGraphBank(cfm, bank)
    graph._prepared = True
    entered, release = threading.Event(), threading.Event()

    def matches(*args):
        entered.set()
        assert release.wait(5)
        return True

    item = SimpleNamespace(
        matches=matches, run=lambda values: "graph", close=lambda: None
    )
    graph._graphs[1500] = item
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(graph.solve_euler, *[object() for _ in range(6)])
        assert entered.wait(5)
        try:
            with pytest.raises(RuntimeError, match="busy"):
                graph.close()
        finally:
            release.set()
            future.result(timeout=5)
    assert not graph._closed
