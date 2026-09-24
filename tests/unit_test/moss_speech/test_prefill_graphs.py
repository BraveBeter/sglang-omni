"""Bounded full-decode graphs must preserve ownership and engine lifecycle."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from typer.testing import CliRunner

from sglang_omni.cli import app
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechStreamingPipelineConfig,
)

REPO = Path(__file__).resolve().parents[3]


def api():
    return importlib.import_module("sglang_omni.models.moss_speech.prefill_graphs")


@pytest.mark.parametrize(
    "cls", [MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig]
)
def test_flag_default_inheritance_and_explicit_disable(cls):
    config = cls(model_path="local", decode_projection_graphs=True)
    assert config.prefill_full_graphs in (None, False)
    stage = next(s for s in config.stages if s.name == "ar_engine")
    assert not stage.factory_args.get("prefill_full_graphs", False)
    stage.factory_args["prefill_full_graphs"] = True
    inherited = ConfigManager(config).merge_config({})
    assert (
        next(s for s in inherited.stages if s.name == "ar_engine").factory_args[
            "prefill_full_graphs"
        ]
        is True
    )
    disabled = ConfigManager(inherited).merge_config({"prefill_full_graphs": False})
    restored = cls.model_validate_json(disabled.model_dump_json())
    assert (
        next(s for s in restored.stages if s.name == "ar_engine").factory_args[
            "prefill_full_graphs"
        ]
        is False
    )
    assert (
        next(s for s in inherited.stages if s.name == "ar_engine").factory_args[
            "prefill_full_graphs"
        ]
        is True
    )


@pytest.mark.parametrize("profile", ["moss_speech", "moss_speech_streaming"])
@pytest.mark.parametrize("enabled", [False, True])
def test_public_cli_reaches_only_ar(profile, enabled, monkeypatch):
    launch = Mock()
    monkeypatch.setattr(
        importlib.import_module("sglang_omni.cli.serve"), "launch_server", launch
    )
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--config",
            str(REPO / f"examples/configs/{profile}.yaml"),
            "--prefill-full-graphs",
            str(enabled).lower(),
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    config = launch.call_args.args[0]
    assert config.prefill_full_graphs is enabled
    for stage in config.stages:
        if stage.name == "ar_engine":
            assert stage.factory_args["prefill_full_graphs"] is enabled
        else:
            assert "prefill_full_graphs" not in stage.factory_args


def test_stage_factory_passes_independent_flag(monkeypatch):
    stages = importlib.import_module("sglang_omni.models.moss_speech.stages")
    builder_module = importlib.import_module(
        "sglang_omni.models.moss_speech.engine_builder"
    )
    builder = Mock()
    monkeypatch.setattr(stages, "validate_ar_preconditions", Mock())
    monkeypatch.setattr(builder_module, "MossSpeechEngineBuilder", builder)
    stages.create_ar_engine_executor(
        "local", prefill_full_graphs=True, decode_full_graphs=False
    )
    assert builder.call_args.kwargs["prefill_full_graphs"] is True
    assert builder.call_args.kwargs["decode_full_graphs"] is False


def test_every_exact_length_is_covered_without_padding():
    assert api().LENGTHS == tuple(range(1, 129))
    for n in range(1, 129):
        assert (
            api().prefill_length(is_extend=True, batch_size=1, length=n, prefix=0) == n
        )
    for n in (0, 129, 512):
        assert (
            api().prefill_length(is_extend=True, batch_size=1, length=n, prefix=0)
            is None
        )
    for kw in (
        dict(is_extend=False, batch_size=1, length=8, prefix=0),
        dict(is_extend=True, batch_size=2, length=8, prefix=0),
        dict(is_extend=True, batch_size=1, length=8, prefix=1),
    ):
        assert api().prefill_length(**kw) is None


def test_private_scatter_preserves_all_other_slots_and_dtype():
    source = SimpleNamespace(
        k_buffer=[
            torch.arange(6 * 8 * 2, dtype=torch.bfloat16).reshape(6, 8, 2) + i
            for i in range(40)
        ],
        v_buffer=[torch.full((6, 8, 2), i, dtype=torch.bfloat16) for i in range(40)],
    )
    target = SimpleNamespace(
        k_buffer=[torch.full((19, 8, 2), -1, dtype=torch.bfloat16) for _ in range(40)],
        v_buffer=[torch.full((19, 8, 2), -2, dtype=torch.bfloat16) for _ in range(40)],
    )
    loc = torch.tensor([12, 1, 9, 3, 17])
    api().scatter_private_kv(source, target, loc, 5)
    other = torch.tensor([i for i in range(19) if i not in loc.tolist()])
    for i in range(40):
        assert torch.equal(target.k_buffer[i][loc], source.k_buffer[i][1:])
        assert torch.equal(target.v_buffer[i][loc], source.v_buffer[i][1:])
        assert torch.all(target.k_buffer[i][other] == -1)
        assert torch.all(target.v_buffer[i][other] == -2)
    with pytest.raises(ValueError):
        api().scatter_private_kv(source, target, loc[:4], 5)


def test_owned_outputs_leave_inference_scope_and_retain_full_hidden():
    with torch.inference_mode():
        raw = SimpleNamespace(
            text_logits=torch.randn(1, 7),
            audio_logits=torch.randn(1, 5),
            hidden_states=torch.randn(2, 1, 4),
        )
        text, audio, idx = torch.randn(8, 4), torch.randn(8, 4), torch.tensor([7])
    model = SimpleNamespace(_dual_logits_by_rid={})
    batch = SimpleNamespace(rids=["actual"])
    result = api().owned_prefill_output(model, raw, text, audio, idx, batch, 8)
    assert not result.text_logits.is_inference()
    assert model._last_text_hidden.shape == (8, 4)
    assert model._last_request_lengths == [8]
    assert model._last_prefix_lengths == [0]
    assert any(key[0] == "actual" for key in model._dual_logits_by_rid)
    result.text_logits.add_(1)
    assert not torch.equal(result.text_logits, raw.text_logits)
    assert model._last_text_hidden.data_ptr() != text.data_ptr()


def test_partial_capture_closes_before_storage_release():
    calls = []
    cache = api().PrefillGraphCache(synchronize=lambda: calls.append("sync"))

    def factory(n):
        if n == 4:
            raise RuntimeError("capture failed")
        return SimpleNamespace(close=lambda: calls.append(n))

    with pytest.raises(RuntimeError, match="capture failed"):
        cache.initialize(factory)
    assert calls == ["sync", 1, 2, 3]
    assert cache.size == 0
    cache.close()
    assert calls == ["sync", 1, 2, 3]


def test_cache_all_lengths_only_initialize_and_close_lock():
    calls = []
    cache = api().PrefillGraphCache(synchronize=lambda: calls.append("sync"))
    entries = {}

    def factory(n):
        e = Mock()
        e.replay.return_value = n
        entries[n] = e
        return e

    cache.initialize(factory)
    assert cache.size == 128
    cache.initialize(lambda n: pytest.fail("recapture"))
    assert cache.replay(50) == 50
    cache.close()
    assert cache.size == 0 and calls == ["sync"]
    assert all(e.close.call_count == 1 for e in entries.values())
    assert cache.replay(50) is None


def test_builder_post_setup_before_start_and_combined_cleanup():
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder

    b = MossSpeechEngineBuilder(prefill_full_graphs=True, decode_full_graphs=False)
    manager = Mock()
    b._prefill_graph_manager = manager
    runner = Mock()
    scheduler = SimpleNamespace(_running=False, _scheduler_thread_id=None)
    b.post_scheduler_setup(scheduler, runner)
    manager.initialize.assert_called_once_with(runner)
    callback = b.extra_scheduler_callbacks()["shutdown_callback"]
    callback()
    manager.close.assert_called_once()
    b.cleanup_build_failure()
    assert manager.close.call_count == 2


def test_tensor_signature_rejects_inplace_and_replacement():
    tensor = torch.zeros(2, 3)
    expected = api().tensor_signature(tensor)
    assert api().tensor_signature(tensor) == expected
    tensor.add_(1)
    assert api().tensor_signature(tensor) != expected
    assert api().tensor_signature(tensor.clone()) != api().tensor_signature(tensor)


def test_captured_lengths_keep_native_gpu_operation_contract():
    x = torch.tensor([8])
    proxy = api().CapturedLengths(x, 8)
    assert proxy.tolist() == [8]
    assert torch.equal(torch.cumsum(proxy.flatten(), 0), torch.tensor([8]))
    with pytest.raises(ValueError):
        api().CapturedLengths(x, 129)


def fake_admission(monkeypatch):
    m = api()
    manager = m.MossSpeechPrefillGraphManager.__new__(m.MossSpeechPrefillGraphManager)
    manager.cache = m.PrefillGraphCache(synchronize=lambda: None)
    manager.cache._ready = True
    manager.cache._entries = {
        8: SimpleNamespace(accepts=lambda *a: True, replay=Mock(return_value="output"))
    }
    manager.device = torch.device("cpu")
    manager._stream = 0
    manager.backend = object()
    manager.pool = object()
    manager.hits = manager.fallbacks = manager.scatter_calls = 0
    manager._unchanged = lambda: True
    monkeypatch.setattr(m.torch.cuda, "current_device", lambda: None)
    monkeypatch.setattr(
        m.torch.cuda, "current_stream", lambda *a: SimpleNamespace(cuda_stream=0)
    )
    monkeypatch.setattr(m.context_api, "has_forward_context", lambda: True)
    monkeypatch.setattr(m.context_api, "get_attn_backend", lambda: manager.backend)
    batch = SimpleNamespace(
        forward_mode=__import__(
            "sglang.srt.model_executor.forward_batch_info", fromlist=["ForwardMode"]
        ).ForwardMode.EXTEND,
        batch_size=1,
        encoder_lens=None,
        spec_info=None,
        extend_seq_lens_cpu=[8],
        extend_prefix_lens_cpu=[0],
        seq_lens_cpu=[8],
        extend_num_tokens=8,
        rids=["r"],
        out_cache_loc=torch.arange(8),
    )
    return manager, batch


def test_runtime_admission_and_honest_counters(monkeypatch):
    manager, batch = fake_admission(monkeypatch)
    assert (
        manager.try_replay(None, torch.arange(8), batch, torch.zeros(8, 4)) == "output"
    )
    assert (manager.hits, manager.scatter_calls, manager.fallbacks) == (1, 80, 0)
    batch.extend_prefix_lens_cpu = [1]
    assert manager.try_replay(None, None, batch, torch.zeros(8, 4)) is None
    assert (manager.hits, manager.scatter_calls, manager.fallbacks) == (1, 80, 1)


@pytest.mark.parametrize(
    "change",
    [
        "unknown_kw",
        "float_length",
        "missing_cpu",
        "wrong_location_dtype",
        "noncontiguous_location",
        "wrong_backend",
        "changed_epoch",
        "wrong_stream",
    ],
)
def test_runtime_admission_rejects_without_replay(change, monkeypatch):
    manager, batch = fake_admission(monkeypatch)
    kwargs = {}
    if change == "unknown_kw":
        kwargs["custom"] = 1
    elif change == "float_length":
        batch.extend_seq_lens_cpu = batch.seq_lens_cpu = [8.5]
    elif change == "missing_cpu":
        batch.extend_seq_lens_cpu = None
    elif change == "wrong_location_dtype":
        batch.out_cache_loc = torch.arange(8, dtype=torch.int32)
    elif change == "noncontiguous_location":
        batch.out_cache_loc = torch.arange(16)[::2]
    elif change == "wrong_backend":
        monkeypatch.setattr(api().context_api, "get_attn_backend", lambda: object())
    elif change == "changed_epoch":
        manager._unchanged = lambda: False
    elif change == "wrong_stream":
        manager._stream = 1
    assert (
        manager.try_replay(None, torch.arange(8), batch, torch.zeros(8, 4), **kwargs)
        is None
    )
    manager.cache._entries[8].replay.assert_not_called()
    assert (manager.hits, manager.scatter_calls, manager.fallbacks) == (0, 0, 1)


def test_close_failure_retains_storage_then_retry():
    fail = [True]
    events = []

    def sync():
        if fail[0]:
            raise RuntimeError("device sync")
        events.append("sync")

    cache = api().PrefillGraphCache(synchronize=sync)
    entry = SimpleNamespace(close=lambda: events.append("close"))
    cache._entries = {1: entry}
    cache._ready = True
    with pytest.raises(RuntimeError, match="device sync"):
        cache.close()
    assert cache.size == 1 and not cache._ready and not cache._closed
    fail[0] = False
    cache.close()
    assert cache.size == 0 and events == ["sync", "close"]


def test_close_waits_for_active_replay():
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    cache = api().PrefillGraphCache(synchronize=lambda: None)

    def replay():
        entered.set()
        assert release.wait(3)
        return 1

    cache._entries = {1: SimpleNamespace(replay=replay, close=lambda: closed.set())}
    cache._ready = True
    worker = threading.Thread(target=lambda: cache.replay(1))
    worker.start()
    assert entered.wait(3)
    stopper = threading.Thread(target=cache.close)
    stopper.start()
    assert not closed.wait(0.05)
    release.set()
    worker.join(3)
    stopper.join(3)
    assert closed.is_set()


def test_parameter_epoch_includes_registered_buffers_and_parameter_identity():
    manager = api().MossSpeechPrefillGraphManager.__new__(
        api().MossSpeechPrefillGraphManager
    )
    manager.model = torch.nn.Module()
    manager.model._decode_input_embedding = torch.nn.Embedding(2, 3)
    layer = torch.nn.Linear(3, 3)
    layer.register_buffer("rotary", torch.ones(4))
    manager.model.layer = layer
    manager._modules = tuple(manager.model.modules())
    before = manager._parameter_epoch()
    layer.rotary.add_(1)
    assert manager._parameter_epoch() != before
    before = manager._parameter_epoch()
    layer.weight = torch.nn.Parameter(layer.weight.detach().clone())
    assert manager._parameter_epoch() != before


def test_publication_restore_on_failed_startup():
    tensor = torch.tensor([1])
    model = SimpleNamespace(_last_x=tensor, _dual_logits_by_rid={"old": tensor})
    runner = SimpleNamespace(_last_transport="old")
    stash = model._dual_logits_by_rid
    with pytest.raises(RuntimeError):
        with api().preserve_publication(model, runner):
            model._last_x = torch.tensor([2])
            model._last_new = 5
            model._dual_logits_by_rid = {}
            runner._last_transport = "new"
            raise RuntimeError("capture")
    assert model._last_x is tensor and not hasattr(model, "_last_new")
    assert model._dual_logits_by_rid is stash and stash["old"] is tensor
    assert runner._last_transport == "old"


def test_speculative_extend_mode_is_not_native_prefill(monkeypatch):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    manager, batch = fake_admission(monkeypatch)
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    assert manager.try_replay(None, torch.arange(8), batch, torch.zeros(8, 4)) is None
    manager.cache._entries[8].replay.assert_not_called()


def test_execution_policy_includes_matmul_precision(monkeypatch):
    original = api().execution_policy()
    cuda = api().torch.backends.cuda.matmul
    monkeypatch.setattr(
        cuda,
        "allow_bf16_reduced_precision_reduction",
        not cuda.allow_bf16_reduced_precision_reduction,
    )
    assert api().execution_policy() != original


def test_grad_input_preserves_native_autograd_by_fallback():
    entry = api()._PrefillGraph.__new__(api()._PrefillGraph)
    entry.closed = False
    entry.embeds = torch.zeros(8, 4)
    entry.batch = SimpleNamespace(positions=torch.arange(8))
    entry._owners = (entry.embeds, entry.batch.positions)
    entry._signature = tuple(api().tensor_signature(t) for t in entry._owners)
    assert entry.accepts(torch.zeros(8, 4), torch.arange(8))
    assert not entry.accepts(torch.zeros(8, 4, requires_grad=True), torch.arange(8))


def test_changed_native_head_method_rejects_cached_outputs():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._decode_input_embedding = torch.nn.Embedding(2, 3)

        def _forward_eager(self):
            pass

        def compute_dual_logits(self):
            pass

    manager = api().MossSpeechPrefillGraphManager.__new__(
        api().MossSpeechPrefillGraphManager
    )
    manager.model = Model()
    manager.pool = SimpleNamespace(
        k_buffer=[torch.zeros(2, 8, 2)] * 40, v_buffer=[torch.zeros(2, 8, 2)] * 40
    )
    manager.request_pool = SimpleNamespace(
        req_to_token=torch.zeros(2, 8, dtype=torch.int64)
    )
    manager.backend = SimpleNamespace(
        token_to_kv_pool=manager.pool, req_to_token_pool=manager.request_pool
    )
    manager.runtime = SimpleNamespace(
        attn_backend=manager.backend,
        token_to_kv_pool=manager.pool,
        req_to_token_pool=manager.request_pool,
    )
    manager._modules = tuple(manager.model.modules())
    manager._registrations = tuple(
        (m, tuple(m._modules.items())) for m in manager._modules
    )
    manager._epoch = manager._parameter_epoch()
    manager._pool_storage = manager._pool_signature()
    manager._policy = api().execution_policy()
    manager._eager = Model._forward_eager
    manager._heads = Model.compute_dual_logits
    manager._rope_layers = ()
    manager._rope_storage = ()
    assert manager._unchanged()
    manager.model.compute_dual_logits = lambda: None
    assert not manager._unchanged()


def test_actual_unregistered_rotary_dependency_detects_replace_and_mutation():
    manager = api().MossSpeechPrefillGraphManager.__new__(
        api().MossSpeechPrefillGraphManager
    )
    layer = SimpleNamespace(_projection_inv_freq=torch.ones(8))
    manager._rope_layers = (layer,)
    before = manager._rope_epoch()
    layer._projection_inv_freq.add_(1)
    assert manager._rope_epoch() != before
    before = manager._rope_epoch()
    layer._projection_inv_freq = layer._projection_inv_freq.clone()
    assert manager._rope_epoch() != before


def native_norm_model():
    from sglang_omni.models.moss_speech.sglang_model import MossSpeechRMSNorm

    def layer():
        result = torch.nn.Module()
        result.input_layernorm = MossSpeechRMSNorm(4)
        result.post_attention_layernorm = MossSpeechRMSNorm(4)
        result.self_attn = torch.nn.Module()
        result.self_attn.q_norm = MossSpeechRMSNorm(2)
        result.self_attn.k_norm = MossSpeechRMSNorm(2)
        result.self_attn.projection = torch.nn.Linear(4, 4, bias=False)
        result.self_attn.projection.weight.requires_grad_(False)
        return result

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(layer() for _ in range(32))
    model.text_block = torch.nn.ModuleList(layer() for _ in range(4))
    model.audio_block = torch.nn.ModuleList(layer() for _ in range(4))
    model.text_norm = MossSpeechRMSNorm(4)
    model.audio_norm = MossSpeechRMSNorm(4)
    model.head = torch.nn.Linear(4, 2, bias=False)
    model.head.weight.requires_grad_(False)
    model.eval()
    return model


def test_original_native_norm_grad_profile_is_admitted_without_mutation():
    model = native_norm_model()
    before = [
        (id(p), p.requires_grad, p._version, p.detach().clone())
        for p in model.parameters()
    ]
    assert sum(p.requires_grad for p in model.parameters()) == 162
    assert api().native_norm_parameter_profile(model)
    after = list(model.parameters())
    assert len(before) == len(after)
    for (identity, requires_grad, version, value), p in zip(before, after):
        assert (id(p), p.requires_grad, p._version) == (
            identity,
            requires_grad,
            version,
        )
        assert torch.equal(value, p)


@pytest.mark.parametrize(
    "change",
    [
        "head",
        "projection",
        "aliased_head",
        "subclass",
        "overridden_forward",
        "duplicate_norm",
        "extra_norm",
        "missing_layer",
        "training_norm",
    ],
)
def test_native_norm_profile_rejects_unknown_grad_ownership(change):
    from sglang_omni.models.moss_speech.sglang_model import MossSpeechRMSNorm

    model = native_norm_model()
    if change == "head":
        model.head.weight.requires_grad_(True)
    elif change == "projection":
        model.layers[0].self_attn.projection.weight.requires_grad_(True)
    elif change == "aliased_head":
        model.head.weight = model.text_norm.weight
    elif change == "subclass":

        class CustomNorm(MossSpeechRMSNorm):
            pass

        model.text_norm = CustomNorm(4).eval()
    elif change == "overridden_forward":
        model.text_norm.forward = lambda x: x
    elif change == "duplicate_norm":
        model.audio_norm = model.text_norm
    elif change == "extra_norm":
        model.extra_norm = MossSpeechRMSNorm(4).eval()
    elif change == "missing_layer":
        model.audio_block = model.audio_block[:3]
    elif change == "training_norm":
        model.text_norm.train()
    assert not api().native_norm_parameter_profile(model)


def test_native_norm_grad_or_training_change_still_invalidates_epoch():
    model = native_norm_model()
    model._decode_input_embedding = torch.nn.Identity()
    manager = api().MossSpeechPrefillGraphManager.__new__(
        api().MossSpeechPrefillGraphManager
    )
    manager.model = model
    manager._modules = tuple(model.modules())
    initial = manager._parameter_epoch()
    model.text_norm.weight.requires_grad_(False)
    assert manager._parameter_epoch() != initial
    model.text_norm.weight.requires_grad_(True)
    initial = manager._parameter_epoch()
    model.text_norm.train()
    assert manager._parameter_epoch() != initial


def test_qualified_runtime_accepts_only_original_norm_grad_flags(monkeypatch):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    from sglang_omni.models.moss_speech.attention import MossSpeechTorchNativeBackend
    from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel

    # Real model/norm/backend/pool types; only hardware queries and storage are
    # bounded CPU fixtures. This exercises the actual owner admission method.
    native = native_norm_model()
    model = MossSpeechSGLangModel.__new__(MossSpeechSGLangModel)
    torch.nn.Module.__init__(model)
    for name, module in native.named_children():
        setattr(model, name, module)
    model.embed_tokens = torch.nn.Embedding(2, 4, dtype=torch.bfloat16)
    model.embed_tokens.weight.requires_grad_(False)
    model.eval()
    for layer in [*model.layers, *model.text_block, *model.audio_block]:
        layer.self_attn.attn = SimpleNamespace(
            tp_q_head_num=32,
            tp_k_head_num=8,
            qk_head_dim=128,
            v_head_dim=128,
            is_cross_attention=False,
            sliding_window_size=-1,
        )
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.page_size, pool.dtype, pool.head_num, pool.head_dim = 1, torch.bfloat16, 8, 128
    pool.layer_num, pool.kv_cache_layout = 40, "nhd"
    pool.k_buffer = [torch.empty(0)] * 40
    pool.v_buffer = [torch.empty(0)] * 40
    backend = MossSpeechTorchNativeBackend.__new__(MossSpeechTorchNativeBackend)
    backend.use_sliding_window_kv_pool = False
    manager = api().MossSpeechPrefillGraphManager.__new__(
        api().MossSpeechPrefillGraphManager
    )
    manager.model, manager.pool, manager.backend = model, pool, backend
    manager.device = torch.device("cuda", 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *_: "NVIDIA A800-SXM4-80GB"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 0))
    monkeypatch.delenv("MOSS_DUMP_DIR", raising=False)
    assert manager._qualified()
    model.head.weight.requires_grad_(True)
    assert not manager._qualified()
    model.head.weight.requires_grad_(False)
    model.train()
    assert not manager._qualified()


def closing_owner(monkeypatch, synchronize=lambda: None):
    from contextlib import nullcontext

    m = api()
    manager = m.MossSpeechPrefillGraphManager.__new__(m.MossSpeechPrefillGraphManager)
    manager.cache = m.PrefillGraphCache(synchronize=synchronize)
    manager.cache._ready = True
    manager.cache._entries = {
        n: SimpleNamespace(captures=1, body_calls=3, close=Mock())
        for n in range(1, 129)
    }
    manager.device = torch.device("cpu")
    manager.hits, manager.fallbacks, manager.scatter_calls = 7, 2, 560
    manager.startup_seconds, manager.memory, manager.reason = (
        2.5,
        {"allocated_after": 1024},
        "",
    )
    manager._parameter_owners = manager._rope_owners = (torch.zeros(1),)
    monkeypatch.setattr(m.torch.cuda, "device", lambda *_: nullcontext())
    log = Mock()
    monkeypatch.setattr(m.logger, "info", log)
    return manager, log


def test_close_receipt_keeps_startup_counts_and_logs_once(monkeypatch):
    manager, log = closing_owner(monkeypatch)
    manager.close()
    manager.close()
    log.assert_called_once()
    message, receipt = log.call_args.args
    assert message == "MOSS prefill graphs closed: %s"
    assert receipt["graphs"] == receipt["captures"] == 128
    assert receipt["body_calls"] == 384
    assert receipt["lengths"] == list(range(1, 129))
    assert receipt["closed"] is True and receipt["enabled"] is False
    assert receipt["remaining_graphs"] == 0
    assert (receipt["replays"], receipt["scatter_calls"], receipt["fallbacks"]) == (
        7,
        560,
        2,
    )
    assert receipt["memory"] == {"allocated_after": 1024}
    assert manager.cache.size == 0
    assert not manager._parameter_owners and not manager._rope_owners


def test_failed_close_does_not_log_or_release_before_retry(monkeypatch):
    failing = [True]

    def synchronize():
        if failing[0]:
            raise RuntimeError("synchronize failed")

    manager, log = closing_owner(monkeypatch, synchronize)
    with pytest.raises(RuntimeError, match="synchronize failed"):
        manager.close()
    log.assert_not_called()
    assert manager.cache.size == 128 and manager._parameter_owners
    failing[0] = False
    manager.close()
    log.assert_called_once()
    assert log.call_args.args[1]["graphs"] == 128
    assert log.call_args.args[1]["closed"] is True
