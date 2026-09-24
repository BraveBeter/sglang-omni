"""Model-local full-CFM owner contracts before GPU qualification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_speech.components import cfm_full_graph as api
from sglang_omni.models.moss_speech.components.cosyvoice_flow.flow_matching import (
    CausalConditionalCFM,
)


class Cfm:
    def __init__(self):
        self.calls = []

    def solve_euler(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "eager"


def owner():
    cfm = Cfm()
    bank = SimpleNamespace(
        flow=SimpleNamespace(decoder=cfm),
        binding=SimpleNamespace(shapes=(1500,), valid=True),
        active_length=None,
        mode="cached",
        entries={},
    )
    return api.FullCfmGraphBank(cfm, bank), cfm, bank


def test_temporary_instance_method_restored_on_failure_and_preserves_inheritance():
    model = Cfm()
    original = model.solve_euler
    with pytest.raises(RuntimeError, match="failed"):
        with api.temporary_attribute(model, "solve_euler", lambda: "temporary"):
            assert model.solve_euler() == "temporary"
            raise RuntimeError("failed")
    assert "solve_euler" not in model.__dict__
    assert model.solve_euler == original

    def marker():
        return "owned"

    model.solve_euler = marker
    with api.temporary_attribute(model, "solve_euler", lambda: "replacement"):
        pass
    assert model.solve_euler is marker


def test_unprepared_unknown_requests_fall_back_without_cuda_or_capture():
    graph, cfm, bank = owner()
    values = [object() for _ in range(6)]
    assert graph.solve_euler(*values, streaming=False) == "eager"
    assert len(cfm.calls) == 1
    assert graph.status()["graph_replays"] == 0
    assert graph.status()["fallback_calls"] == 1
    assert not graph.status()["enabled"]
    with pytest.raises(RuntimeError, match="prepared"):
        graph.install()
    assert "solve_euler" not in cfm.__dict__


def test_install_close_restores_original_and_keeps_parent_bank(monkeypatch):
    graph, cfm, bank = owner()
    graph._prepared = True
    graph.install()
    assert cfm.solve_euler == graph.solve_euler
    graph.close()
    graph.close()
    assert "solve_euler" not in cfm.__dict__ and bank.binding.valid
    assert cfm.solve_euler(1) == "eager"
    assert graph.solve_euler(1, 2, 3, 4, 5, 6) == "eager"


def test_close_does_not_overwrite_a_later_instance_override():
    graph, cfm, _ = owner()
    graph._prepared = True
    graph.install()

    def later():
        return "later"

    cfm.solve_euler = later
    graph.close()
    assert cfm.solve_euler is later


def test_close_rejects_overlap_without_dropping_owners():
    graph, _, bank = owner()
    assert graph._lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="busy"):
            graph.close()
        assert not graph._closed and bank.binding.valid
    finally:
        graph._lock.release()
    graph.close()


def test_admitted_dispatch_counts_only_real_replay_across_cpu_threads():
    graph, cfm, bank = owner()
    bank.active_length = 1500
    graph._prepared = True
    seen = []
    entry = SimpleNamespace(
        matches=lambda values, streaming, proof: streaming and proof is token,
        run=lambda values: seen.append(values) or "graph",
        bytes=0,
        shared_bytes=0,
    )
    graph._graphs[1500] = entry
    token = object()
    values = [object() for _ in range(6)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert (
            executor.submit(
                graph.solve_euler, *values, streaming=True, full_valid_mask=token
            ).result(timeout=5)
            == "graph"
        )
    assert tuple(seen[0]) == api.INPUT_KEYS
    assert list(seen[0].values()) == values
    assert graph.status()["graph_replays"] == 1 and not cfm.calls
    assert graph.solve_euler(*values, streaming=True, full_valid_mask=None) == "eager"
    assert graph.status()["graph_replays"] == 1


def test_prepare_refuses_active_scope_and_does_not_run_calibration():
    graph, _, bank = owner()
    bank.active_length = 1500
    called = []
    with pytest.raises(RuntimeError, match="active"):
        graph.prepare(lambda: called.append(True))
    assert not called and bank.binding.valid


def test_euler_body_matches_original_operations_for_dynamic_six_inputs():
    class Solver:
        inference_optimizations = True
        inference_cfg_rate = 0.7
        calls = 0

        def forward_estimator(self, x, mask, mu, t, spks, cond, streaming, **kwargs):
            self.calls += 1
            return (
                x * 0.125
                + mu * 0.25
                + cond * 0.375
                + spks.unsqueeze(-1) * 0.01
                + t[:, None, None] * mask
            )

    generator = torch.Generator().manual_seed(7)
    values = dict(
        x=torch.randn(1, 80, 16, generator=generator),
        mask=torch.ones(1, 1, 16),
        mu=torch.randn(1, 80, 16, generator=generator),
        t_span=1 - torch.cos(torch.linspace(0, 1, 11) * 0.5 * torch.pi),
        spks=torch.randn(1, 80, generator=generator),
        cond=torch.randn(1, 80, 16, generator=generator),
    )
    solver = Solver()
    body = api.EulerBody(solver, values, proof=None)
    for _ in range(2):
        expected = CausalConditionalCFM.solve_euler(solver, **values, streaming=True)
        result = body()
        assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))
        assert body.last_estimator_calls == 10
        values["x"].mul_(0.5)
        values["mu"].add_(0.03125)


def test_raw_gate_preserves_signed_zero_and_owned_input_layout():
    assert not api.raw_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert api.raw_equal(torch.tensor([0.0]), torch.tensor([0.0]))
    first = torch.ones(2, 3)
    assert api.tensor_layout(first) == api.tensor_layout(first.clone())
    assert api.tensor_storage(first) != api.tensor_storage(first.clone())


@pytest.mark.parametrize("length", [1392, 1400, 1424, 1500])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_ff_first_tail_uses_explicit_tail_projection_then_native_gelu(
    length, approximate
):
    from diffusers.models.activations import GELU

    module = GELU(256, 1024, approximate=approximate).eval()
    value = torch.linspace(-3, 3, 2 * length * 256).reshape(2, length, 256)
    projection_inputs = []
    gelu_inputs = []
    native_gelu = module.gelu

    def projection(source):
        projection_inputs.append(source)
        return source.repeat_interleave(4, dim=-1)

    def gelu(source):
        gelu_inputs.append(source)
        return native_gelu(source)

    def forbidden(source):
        raise AssertionError("Native full Linear reached graph replay")

    module.proj.forward = forbidden
    module.gelu = gelu
    # CPU GELU dispatch is not the qualified CUDA kernel. Verify routing here;
    # startup separately checks every actual CUDA tail against full-module bits.
    expected = native_gelu(projection(value[:, 1280:, :].contiguous()))
    projection_inputs.clear()
    actual = api.ff_first_tail(module, value, projection)
    assert len(projection_inputs) == 1
    assert projection_inputs[0].shape == (2, length - 1280, 256)
    assert not projection_inputs[0].is_contiguous()
    assert projection_inputs[0].stride() == (length * 256, 256, 1)
    assert projection_inputs[0].storage_offset() == 1280 * 256
    assert projection_inputs[0].data_ptr() == value[:, 1280:, :].data_ptr()
    assert len(gelu_inputs) == 1
    assert gelu_inputs[0].shape == (2, length - 1280, 1024)
    assert api.raw_equal(actual, expected)
    assert actual.data_ptr() != expected.data_ptr()


@pytest.mark.parametrize("unsupported", ["subclass", "mode", "projection"])
def test_ff_first_tail_rejects_unqualified_module(unsupported):
    from diffusers.models.activations import GELU

    class OtherGelu(GELU):
        pass

    module = (OtherGelu if unsupported == "subclass" else GELU)(256, 1024)
    if unsupported == "mode":
        module.approximate = "changed"
    if unsupported == "projection":
        module.proj = torch.nn.Linear(256, 512)
    with pytest.raises(RuntimeError, match="FF first"):
        api.ff_first_tail(module, torch.zeros(2, 1500, 256), lambda value: value)


def test_ff_first_startup_checks_actual_full_output_and_counts_only_passes():
    from diffusers.models.activations import GELU

    graph, _, _ = owner()
    graph._collecting = 1500
    graph._key = lambda name: (1500, 0, name)
    module = GELU(256, 1024).eval()
    value = torch.zeros(2, 1500, 256)
    calls = []

    def projection(source):
        calls.append(source.shape)
        return source.repeat_interleave(4, dim=-1)

    module.proj.forward = projection
    expected = module(value)
    graph._tail_plans[(1500, "layer", "ff_first")] = projection
    hook = module.register_forward_hook(
        lambda mod, args, output: graph._collect_ff_first("layer", mod, args, output)
    )
    try:
        module(value)
    finally:
        hook.remove()
    assert graph._ff_first_checks == 1
    assert calls == [(2, 1500, 256)] * 2 + [(2, 220, 256)]
    wrong = expected.clone()
    wrong[:, 1280:, :] = -0.0
    with pytest.raises(RuntimeError, match="FF first/GELU"):
        graph._collect_ff_first("layer", module, (value,), wrong)
    assert graph._ff_first_checks == 1


def test_whole_block_scope_uses_current_input_and_restores_on_failure():
    from sglang_omni.models.moss_speech.components.cfm_prefix_cache import (
        PrefixCallLedger,
    )

    graph, _, bank = owner()
    bank.names = ("layer.attn1",)
    graph._tail_ledger = PrefixCallLedger(bank.names)
    hidden, mask = object(), object()

    def original(**kw):
        return "native"

    block = SimpleNamespace(forward=original)
    graph._blocks = (("layer.attn1", block),)
    seen = []

    def run(value, bias, *, key):
        seen.append((value, bias, key))
        return "tail"

    graph._tail_blocks[(1400, 0, "layer.attn1")] = SimpleNamespace(run=run)
    with pytest.raises(RuntimeError, match="injected"):
        with graph.block_scope(1400):
            assert block.forward(hidden_states=hidden, attention_mask=mask) == "tail"
            raise RuntimeError("injected")
    assert block.forward is original
    assert seen == [(hidden, mask, (1400, 0, "layer.attn1"))]


@pytest.mark.parametrize(
    "kind,k,out_features",
    [("ff_first", 256, 1024), ("ff", 1024, 256), ("to_out", 512, 256)],
)
def test_projection_calibration_selects_qualified_profile_and_rejects_bad_tail(
    monkeypatch, kind, k, out_features
):
    from contextlib import nullcontext

    from sglang_omni.models.moss_speech.components import cfm_direct_lt

    graph, _, _ = owner()
    graph._collecting = 1400
    graph._key = lambda name: (1400, 0, name)
    graph._lt_context = object()
    module = torch.nn.Linear(k, out_features)
    value = torch.zeros(2, 1400, k)
    output = torch.zeros(2, 1400, out_features)
    plans = []

    class Plan:
        bad = False

        def __init__(self, weight, bias, rows, *, context, profile):
            assert weight is module.weight and bias is module.bias
            assert context is graph._lt_context
            self.rows, self.profile = rows, profile
            plans.append(self)

        def __call__(self, source):
            assert source.shape == (2, self.rows // 2, k)
            if self.rows == 240 and kind != "ff":
                assert source.stride() == (1400 * k, k, 1)
                assert source.storage_offset() == 1280 * k
                assert source.data_ptr() == value[:, 1280:, :].data_ptr()
            else:
                assert source.is_contiguous()
            result = torch.zeros(2, self.rows // 2, out_features)
            if self.bad:
                result[0, 0, 0] = -0.0
            return result

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(cfm_direct_lt, "DirectLtProjection", Plan)
    graph._collect_projection("layer", kind, module, (value,), output)
    assert [(p.rows, p.profile) for p in plans] == [
        (240, "ff_first" if kind == "ff_first" else "output"),
        (2800, "ff_first" if kind == "ff_first" else "output"),
    ]
    if kind == "ff_first":
        assert not graph._prefixes and graph._raw_checks == 0
        assert graph._ff_first_keys == {(1400, 0, "layer", kind)}
    else:
        assert not graph._prefixes and len(graph._raw_keys) == graph._raw_checks == 1
        assert not graph._ff_first_keys
    with pytest.raises(RuntimeError, match="Duplicate"):
        graph._collect_projection("layer", kind, module, (value,), output)
    graph._key = lambda name: (1400, 1, name)
    plans[0].bad = True
    before = (len(graph._prefixes), graph._raw_checks, len(graph._ff_first_keys))
    with pytest.raises(RuntimeError, match="tail direct-Lt"):
        graph._collect_projection("layer", kind, module, (value,), output)
    assert before == (
        len(graph._prefixes),
        graph._raw_checks,
        len(graph._ff_first_keys),
    )


@pytest.mark.parametrize("count", [0, 55, 56, 559])
def test_qkv_key_precedes_attention_while_old_projections_follow_it(count):
    graph, _, bank = owner()
    bank.active_length = 1400
    bank.names = tuple(f"layer{i}" for i in range(56))
    bank.ledger = SimpleNamespace(count=count)
    label = bank.names[count % 56]
    assert graph._key(label, before_attention=True) == (1400, count // 56, label)
    bank.ledger.count += 1
    assert graph._key(label) == (1400, count // 56, label)
    with pytest.raises(RuntimeError, match="order"):
        graph._key(bank.names[(count + 1) % 56])
    bank.ledger.count = 560
    with pytest.raises(RuntimeError, match="order"):
        graph._key(label, before_attention=True)


@pytest.mark.parametrize("length", [1392, 1400, 1424, 1500])
@pytest.mark.parametrize("kind", ["q", "k", "v"])
def test_qkv_calibration_uses_pre_attention_profile_and_retains_only_kv_prefixes(
    monkeypatch, length, kind
):
    from sglang_omni.models.moss_speech.components import cfm_direct_lt

    graph, _, bank = owner()
    graph._collecting = bank.active_length = length
    bank.names = tuple(f"layer{i}" for i in range(56))
    bank.ledger = SimpleNamespace(count=0)
    graph._lt_context = object()
    linear = torch.nn.Linear(256, 512, bias=False)
    value = torch.ones(2, length, 256)
    output = torch.ones(2, length, 512)
    plans = []

    class Plan:
        def __init__(self, weight, bias, rows, *, context, profile):
            assert weight is linear.weight and bias is None
            assert context is graph._lt_context
            self.rows, self.profile = rows, profile
            plans.append(self)

        def __call__(self, source):
            assert source.shape == (2, self.rows // 2, 256)
            if self.rows == 2 * length:
                assert source.is_contiguous()
            else:
                assert source.stride() == (length * 256, 256, 1)
                assert source.storage_offset() == 1280 * 256
                assert source.data_ptr() == value[:, 1280:, :].data_ptr()
            return source.repeat_interleave(2, -1)

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(cfm_direct_lt, "DirectLtProjection", Plan)
    graph._collect_projection("layer0", kind, linear, (value,), output)
    assert [(p.rows, p.profile) for p in plans] == [
        (2 * (length - 1280), f"qkv_{length}"),
        (2 * length, f"qkv_{length}"),
    ]
    key = (length, 0, "layer0", kind)
    assert graph._qkv_keys == {key}
    assert graph._raw_checks == 0 and not graph._ff_first_keys
    if kind == "q":
        assert graph._prefixes == {}
        assert set(graph._query_slots) == {length}
        zeros = graph._query_slots[length]
        assert zeros.shape == (2, length, 512) and zeros.is_contiguous()
        assert torch.count_nonzero(zeros) == 0
        bank.ledger.count = 1
        graph._collect_projection("layer1", kind, linear, (value,), output)
        assert graph._query_slots[length] is zeros
    else:
        assert not graph._prefixes and set(graph._kv_slots) == {key}
        prefix = graph._kv_slots[key]
        assert api.raw_equal(prefix, output)
        assert prefix.data_ptr() != output.data_ptr()
        assert not graph._query_slots
    with pytest.raises(RuntimeError, match="Duplicate"):
        graph._collect_projection(
            bank.names[bank.ledger.count], kind, linear, (value,), output
        )


@pytest.mark.parametrize("length", [1392, 1400, 1424, 1500])
@pytest.mark.parametrize("failure", ["full", "tail"])
def test_qkv_calibration_rejects_full_or_tail_before_adding_cache(
    monkeypatch, length, failure
):
    from sglang_omni.models.moss_speech.components import cfm_direct_lt

    graph, _, bank = owner()
    graph._collecting = bank.active_length = length
    bank.names = tuple(f"layer{i}" for i in range(56))
    bank.ledger = SimpleNamespace(count=0)
    graph._lt_context = object()
    linear = torch.nn.Linear(256, 512, bias=False)
    calls = []

    class Plan:
        def __init__(self, weight, bias, rows, *, context, profile):
            self.rows = rows

        def __call__(self, source):
            which = "full" if self.rows == 2 * length else "tail"
            calls.append(which)
            result = torch.zeros(2, self.rows // 2, 512)
            if which == failure:
                result[0, 0, 0] = -0.0
            return result

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(cfm_direct_lt, "DirectLtProjection", Plan)
    with pytest.raises(RuntimeError, match=f"{failure} direct-Lt"):
        graph._collect_projection(
            "layer0",
            "k",
            linear,
            (torch.zeros(2, length, 256),),
            torch.zeros(2, length, 512),
        )
    assert calls == (["full"] if failure == "full" else ["full", "tail"])
    assert (
        not graph._qkv_keys
        and not graph._prefixes
        and not graph._query_slots
        and not graph._kv_slots
    )


def test_whole_block_ledger_is_independent_of_parent_attention_ledger():
    from sglang_omni.models.moss_speech.components.cfm_prefix_cache import (
        PrefixCallLedger,
    )

    graph, _, bank = owner()
    bank.names = ("layer.attn1",)
    bank.ledger = SimpleNamespace(count=0)
    graph._tail_ledger = PrefixCallLedger(bank.names)
    block = SimpleNamespace(forward=lambda **kwargs: "native")
    graph._blocks = (("layer.attn1", block),)
    keys = []
    for step in range(10):
        graph._tail_blocks[(1500, step, "layer.attn1")] = SimpleNamespace(
            run=lambda value, mask, *, key: keys.append(key)
        )
    with graph.block_scope(1500):
        for _ in range(10):
            block.forward(hidden_states=object(), attention_mask=object())
    assert keys == [(1500, step, "layer.attn1") for step in range(10)]
    assert graph._tail_ledger.count == 10 and bank.ledger.count == 0


@pytest.mark.parametrize(
    "blocks,sdpa,projections,hits,fallbacks,accepted",
    [
        (560, 560, 3360, 0, 0, True),
        (559, 560, 3360, 0, 0, False),
        (560, 559, 3360, 0, 0, False),
        (560, 560, 3359, 0, 0, False),
        (560, 560, 3360, 560, 0, False),
        (560, 560, 3360, 0, 1, False),
    ],
)
def test_graph_body_requires_actual_tail_counters_and_zero_parent_hits(
    blocks, sdpa, projections, hits, fallbacks, accepted
):
    graph, _, bank = owner()
    bank.binding.voice_digest, bank.binding.environment = "voice", "environment"
    bank.names = tuple(f"layer{i}" for i in range(56))
    bank.hits, bank.fallbacks = 11, 3
    bank.ledger = SimpleNamespace(count=0)
    stats = [0, 0, 0]
    graph._tail_blocks[(1400, 0, "layer0")] = SimpleNamespace(
        counts=lambda: tuple(stats)
    )

    @contextmanager
    def scope(*args):
        bank.active_length = 1400
        try:
            yield
        finally:
            bank.active_length = None

    bank.scope = scope

    class Body:
        last_estimator_calls = 10

        def __call__(self):
            stats[:] = [blocks, sdpa, projections]
            graph._tail_ledger.count = blocks
            bank.hits += hits
            bank.fallbacks += fallbacks
            return "output"

    entry = api._ShapeGraph.__new__(api._ShapeGraph)
    entry.owner, entry.length, entry.body = graph, 1400, Body()
    entry.actual_bodies = []
    if accepted:
        assert entry._body() == "output"
        assert entry.actual_bodies == [
            dict(
                blocks=560,
                estimator_calls=10,
                attention_hits=0,
                attention_fallbacks=0,
                block_tail_calls=560,
                tail_sdpa_calls=560,
                direct_projection_calls=3360,
            )
        ]
    else:
        with pytest.raises(RuntimeError, match="block tail"):
            entry._body()
        assert entry.actual_bodies == []
    assert bank.active_length is None


def test_close_releases_qkv_owned_caches_without_releasing_parent():
    graph, _, bank = owner()
    parent = torch.ones(2, 1280, 512)
    bank.entries[(1500, 0, "layer")] = parent
    graph._qkv_keys.add((1500, 0, "layer", "k"))
    graph._prefixes[(1500, 0, "layer", "k")] = parent.clone()
    graph._query_slots[1500] = torch.zeros_like(parent)
    graph.close()
    assert (
        not graph._qkv_keys
        and not graph._prefixes
        and not graph._query_slots
        and not graph._kv_slots
    )
    assert bank.entries[(1500, 0, "layer")] is parent


def test_whole_block_scope_rejects_unknown_kwargs_before_execution():
    from sglang_omni.models.moss_speech.components.cfm_prefix_cache import (
        PrefixCallLedger,
    )

    graph, _, bank = owner()
    bank.names = ("layer.attn1",)
    graph._tail_ledger = PrefixCallLedger(bank.names)
    block = SimpleNamespace(forward=lambda **kwargs: "native")
    graph._blocks = (("layer.attn1", block),)
    calls = []
    graph._tail_blocks[(1400, 0, "layer.attn1")] = SimpleNamespace(
        run=lambda *a, **k: calls.append(True)
    )
    with graph.block_scope(1400):
        with pytest.raises(RuntimeError, match="block arguments"):
            block.forward(hidden_states=object(), attention_mask=object(), unknown=1)
    assert calls == [] and graph._tail_ledger.count == 0


@pytest.mark.parametrize("failure", [None, "raw", "missing", "count"])
def test_whole_block_startup_counts_only_complete_actual_intermediates(
    monkeypatch, failure
):
    from sglang_omni.models.moss_speech.components import cfm_block_tail

    graph, _, _ = owner()
    graph._collecting = 1400
    graph._key = lambda name, **kwargs: (1400, 0, name)
    graph._query_slots[1400] = torch.zeros(2, 1400, 512)
    for kind in ("k", "v"):
        graph._kv_slots[(1400, 0, "layer", kind)] = torch.zeros(2, 1400, 512)
    for kind in ("q", "k", "v", "to_out", "ff_first", "ff"):
        graph._tail_plans[(1400, "layer", kind)] = object()
    captured = {name: torch.zeros(2, 1400, 256) for name in cfm_block_tail.STAGES}
    monkeypatch.setattr(
        cfm_block_tail, "capture_native", lambda *a: (captured["output"], captured)
    )

    class Candidate:
        def __init__(self, *args):
            pass

        def run(self, hidden, mask, *, key, observe):
            assert key == (1400, 0, "layer")
            for name, tensor in captured.items():
                if failure == "missing" and name == "gelu":
                    continue
                value = tensor if name == "output" else tensor[:, 1280:].contiguous()
                if failure == "raw" and name == "output":
                    value = value.clone()
                    value[0, 0, 0] = -0.0
                observe(name, value)
            return captured["output"]

        def counts(self):
            return (1, 1, 5 if failure == "count" else 6)

    monkeypatch.setattr(cfm_block_tail, "TailBlock", Candidate)
    kwargs = dict(hidden_states=torch.zeros(2, 1400, 256), attention_mask=object())
    if failure is None:
        assert (
            graph._collect_block("layer", object(), lambda: None, (), kwargs)
            is captured["output"]
        )
        assert graph._whole_block_checks == 1 and graph._block_intermediate_checks == 13
        assert len(graph._tail_blocks) == 1
        assert set(graph._prefixes) == {(1400, 0, "layer", "block")}
    else:
        with pytest.raises(RuntimeError):
            graph._collect_block("layer", object(), lambda: None, (), kwargs)
        assert graph._whole_block_checks == graph._block_intermediate_checks == 0
        assert graph._tail_blocks == {}
