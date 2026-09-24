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
    return importlib.import_module("sglang_omni.models.moss_speech.decode_graphs")


@pytest.mark.parametrize(
    "cls", [MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig]
)
def test_flag_default_inheritance_and_explicit_disable(cls):
    config = cls(model_path="local", decode_projection_graphs=True)
    assert config.decode_full_graphs in (None, False)
    stage = next(s for s in config.stages if s.name == "ar_engine")
    assert not stage.factory_args.get("decode_full_graphs", False)
    stage.factory_args["decode_full_graphs"] = True
    inherited = ConfigManager(config).merge_config({})
    assert (
        next(s for s in inherited.stages if s.name == "ar_engine").factory_args[
            "decode_full_graphs"
        ]
        is True
    )
    disabled = ConfigManager(inherited).merge_config({"decode_full_graphs": False})
    restored = cls.model_validate_json(disabled.model_dump_json())
    assert (
        next(s for s in restored.stages if s.name == "ar_engine").factory_args[
            "decode_full_graphs"
        ]
        is False
    )
    assert (
        next(s for s in inherited.stages if s.name == "ar_engine").factory_args[
            "decode_full_graphs"
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
            "--decode-full-graphs",
            str(enabled).lower(),
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    config = launch.call_args.args[0]
    assert config.decode_full_graphs is enabled
    for stage in config.stages:
        if stage.name == "ar_engine":
            assert stage.factory_args["decode_full_graphs"] is enabled
        else:
            assert "decode_full_graphs" not in stage.factory_args


def test_stage_factory_passes_independent_flag(monkeypatch):
    stages = importlib.import_module("sglang_omni.models.moss_speech.stages")
    builder_module = importlib.import_module(
        "sglang_omni.models.moss_speech.engine_builder"
    )
    builder = Mock()
    monkeypatch.setattr(stages, "validate_ar_preconditions", Mock())
    monkeypatch.setattr(builder_module, "MossSpeechEngineBuilder", builder)
    stages.create_ar_engine_executor(
        "local", decode_full_graphs=True, decode_projection_graphs=False
    )
    assert builder.call_args.kwargs["decode_full_graphs"] is True
    assert builder.call_args.kwargs["decode_projection_graphs"] is False


def test_capacity_boundaries_do_not_create_per_length_graphs():
    capacity = api().decode_graph_capacity
    for n, expected in [
        (1, 128),
        (42, 128),
        (128, 128),
        (129, 512),
        (512, 512),
        (513, 1024),
        (1024, 1024),
        (0, None),
        (1025, None),
    ]:
        assert capacity(is_decode=True, batch_size=1, seq_len=n) == expected
    assert capacity(is_decode=False, batch_size=1, seq_len=42) is None
    for size in (0, 2, 4):
        assert capacity(is_decode=True, batch_size=size, seq_len=42) is None


def test_cache_captures_only_at_initialize_and_closes_once():
    events = []
    cache = api().DecodeGraphCache(synchronize=lambda: events.append("sync"))
    entries = {}

    def factory(capacity):
        entry = Mock()
        entry.replay.return_value = capacity
        entry.close.side_effect = lambda: events.append(capacity)
        entries[capacity] = entry
        return entry

    assert cache.replay(128) is None
    cache.initialize(factory)
    cache.initialize(Mock(side_effect=AssertionError("recapture")))
    assert cache.size == 3
    for capacity in (128, 512, 1024):
        assert cache.replay(capacity) == capacity
    assert cache.replay(4096) is None
    cache.close()
    cache.close()
    assert events == ["sync", 128, 512, 1024]
    assert cache.size == 0
    assert cache.replay(128) is None
    cache.initialize(Mock(side_effect=AssertionError("reopen")))


def test_partial_initialization_failure_releases_all_created_entries():
    events = []
    cache = api().DecodeGraphCache(synchronize=lambda: events.append("sync"))
    first = Mock()
    first.close.side_effect = lambda: events.append("free")

    def factory(capacity):
        if capacity == 512:
            raise RuntimeError("capture failed")
        return first

    with pytest.raises(RuntimeError, match="capture failed"):
        cache.initialize(factory)
    assert events == ["sync", "free"]
    assert cache.size == 0
    assert cache.replay(128) is None


def test_close_waits_for_inflight_replay_before_freeing():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    events = []
    cache = api().DecodeGraphCache(synchronize=lambda: events.append("sync"))

    def replay():
        entered.set()
        assert release.wait(2)
        events.append("owned-output")
        return 42

    entry = Mock()
    entry.replay.side_effect = replay
    entry.close.side_effect = lambda: events.append("free")
    cache.initialize(lambda _: entry)
    worker = threading.Thread(target=lambda: cache.replay(128))
    closer = threading.Thread(target=lambda: (cache.close(), closed.set()))
    worker.start()
    assert entered.wait(2)
    closer.start()
    assert not closed.wait(0.05)
    assert not events
    release.set()
    worker.join(2)
    closer.join(2)
    assert not worker.is_alive() and not closer.is_alive()
    assert events[0:2] == ["owned-output", "sync"]


def test_owned_output_and_request_heads_survive_replay_and_sampler_mutation():
    module = api()
    from sglang_omni.models.moss_speech.sglang_model import MossSpeechModelOutput

    text, audio, hidden = (
        torch.arange(5.0)[None],
        torch.arange(3.0)[None],
        torch.arange(8.0).reshape(2, 1, 4),
    )
    raw = MossSpeechModelOutput(
        next_token_logits=text,
        text_logits=text,
        audio_logits=audio,
        hidden_states=hidden,
    )
    model = SimpleNamespace(_dual_logits_by_rid={})
    batch = SimpleNamespace(
        rids=["first"], forward_mode=SimpleNamespace(is_decode=lambda: True)
    )
    first = module.owned_decode_output(model, raw, batch)
    text.fill_(99)
    audio.fill_(99)
    hidden.fill_(99)
    assert torch.equal(first.text_logits, torch.arange(5.0)[None])
    assert torch.equal(first.hidden_states, torch.arange(8.0).reshape(2, 1, 4))
    assert first.next_token_logits is first.text_logits
    first.next_token_logits.fill_(-100)
    assert torch.equal(
        model._dual_logits_by_rid[("first", "decode")][0], torch.arange(5.0)
    )
    batch.rids = ["second"]
    module.owned_decode_output(model, raw, batch)
    assert torch.equal(
        model._dual_logits_by_rid[("first", "decode")][1], torch.arange(3.0)
    )
    assert model._last_text_hidden.data_ptr() != hidden[0].data_ptr()


def test_warmup_restores_only_reserved_storage_and_python_state_on_error():
    module = api()
    model = SimpleNamespace(
        _decode_input_embedding=torch.nn.Embedding(3, 4),
        _dual_logits_by_rid={"existing": object()},
        _last_text_hidden=object(),
    )
    table = torch.arange(16).reshape(2, 8)
    keys, values = [torch.randn(3, 2, 4) for _ in range(2)], [
        torch.randn(3, 2, 4) for _ in range(2)
    ]
    pool = SimpleNamespace(k_buffer=keys, v_buffer=values)
    runtime = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=table), token_to_kv_pool=pool
    )
    before = [
        t.clone() for t in [model._decode_input_embedding.weight, table, *keys, *values]
    ]
    old_hidden = model._last_text_hidden
    old_stash = model._dual_logits_by_rid.copy()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="dummy failed"):
        with module.preserve_warmup_state(model, runtime):
            assert torch.count_nonzero(table[0]) == 0
            assert torch.count_nonzero(model._decode_input_embedding.weight[0]) == 0
            for tensor in keys + values:
                assert torch.count_nonzero(tensor[0]) == 0
                tensor[0].fill_(123)
            model._last_text_hidden = object()
            model._last_audio_hidden = object()
            model._dual_logits_by_rid["dummy"] = object()
            raise ValueError("dummy failed")
    assert model._last_text_hidden is old_hidden
    assert not hasattr(model, "_last_audio_hidden")
    assert model._dual_logits_by_rid == old_stash
    assert torch.equal(torch.get_rng_state(), rng)
    for actual, expected in zip(
        [model._decode_input_embedding.weight, table, *keys, *values], before
    ):
        assert torch.equal(actual, expected)


def test_builder_initializes_before_ready_and_registers_idempotent_shutdown():
    from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder

    builder = MossSpeechEngineBuilder(decode_full_graphs=True)
    manager = Mock()
    model = SimpleNamespace(_decode_graph_manager=manager)
    builder.setup_model_resources(
        model, SimpleNamespace(), generation_cuda_graph_enabled=False
    )
    manager.initialize.assert_called_once_with()
    builder.extra_scheduler_callbacks()["shutdown_callback"]()
    builder.cleanup_build_failure()
    assert manager.close.call_count == 2


def _admission_fixture(monkeypatch):
    module = api()
    manager = object.__new__(module.MossSpeechDecodeGraphManager)
    manager.cache = module.DecodeGraphCache(synchronize=lambda: None)
    entry = Mock()
    entry.replay.return_value = object()
    manager.cache.initialize(lambda _: entry)
    manager.pool, manager.request_pool = object(), object()
    manager.backend = SimpleNamespace(
        token_to_kv_pool=manager.pool, req_to_token_pool=manager.request_pool
    )
    manager.runtime = SimpleNamespace(
        attn_backend=manager.backend,
        token_to_kv_pool=manager.pool,
        req_to_token_pool=manager.request_pool,
    )
    manager._storage = (42,)
    manager._storage_key = lambda: (42,)
    manager._policy = module.sdpa_policy()
    manager.hits = 0
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        batch_size=1,
        encoder_lens=None,
        spec_info=None,
        seq_lens_cpu=torch.tensor([42]),
        rids=["current"],
    )
    monkeypatch.setattr(module.context_api, "has_forward_context", lambda: True)
    monkeypatch.setattr(module.context_api, "get_attn_backend", lambda: manager.backend)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.delenv("MOSS_DUMP_DIR", raising=False)
    return module, manager, batch, entry


@pytest.mark.parametrize(
    "change",
    [
        "policy",
        "backend_pool",
        "runtime_backend",
        "storage",
        "length",
        "prefill",
        "batch",
        "encoder",
        "spec",
        "debug",
    ],
)
def test_changed_runtime_uses_eager_without_replay(change, monkeypatch):
    module, manager, batch, entry = _admission_fixture(monkeypatch)
    assert manager.try_replay(None, None, batch) is entry.replay.return_value
    entry.replay.reset_mock()
    if change == "policy":
        monkeypatch.setattr(module, "sdpa_policy", lambda: ("different",))
    elif change == "backend_pool":
        manager.backend.token_to_kv_pool = object()
    elif change == "runtime_backend":
        manager.runtime.attn_backend = object()
    elif change == "storage":
        manager._storage_key = lambda: (43,)
    elif change == "length":
        batch.seq_lens_cpu = torch.tensor([1025])
    elif change == "prefill":
        batch.forward_mode = SimpleNamespace(is_decode=lambda: False)
    elif change == "batch":
        batch.batch_size = 2
    elif change == "encoder":
        batch.encoder_lens = torch.tensor([10])
    elif change == "spec":
        batch.spec_info = object()
    elif change == "debug":
        monkeypatch.setenv("MOSS_DUMP_DIR", "/tmp/moss-test-unused")
    assert manager.try_replay(None, None, batch) is None
    entry.replay.assert_not_called()


def test_projection_flags_and_forward_context_restore_after_capture_error():
    module = api()
    model = SimpleNamespace(
        layers=[SimpleNamespace(enable_projection_graphs=True)],
        text_block=[SimpleNamespace(enable_projection_graphs=False)],
        audio_block=[],
    )
    original, candidate = object(), object()
    with module.scoped_backend(original):
        with pytest.raises(ValueError, match="capture"):
            with module.projection_mode(model, False), module.scoped_backend(candidate):
                assert module.context_api.get_attn_backend() is candidate
                assert model.layers[0].enable_projection_graphs is False
                raise ValueError("capture")
        assert module.context_api.get_attn_backend() is original
    assert model.layers[0].enable_projection_graphs is True
    assert model.text_block[0].enable_projection_graphs is False


def test_unknown_torch_without_private_priority_api_can_fall_back(monkeypatch):
    module = api()
    monkeypatch.delattr(torch._C, "_get_sdp_priority_order")
    # Manager construction must reach the qualified-runtime check instead of
    # failing during policy bookkeeping on an unsupported Torch installation.
    model = SimpleNamespace(_decode_input_embedding=torch.nn.Embedding(2, 4))
    runtime = SimpleNamespace(
        attn_backend=object(), token_to_kv_pool=object(), req_to_token_pool=object()
    )
    manager = module.MossSpeechDecodeGraphManager(model, runtime)
    manager.initialize()
    assert manager.cache.size == 0
    assert manager.reason is not None


def test_startup_inference_buffers_replay_into_writable_owned_outputs():
    module = api()
    from sglang_omni.models.moss_speech.sglang_model import MossSpeechModelOutput

    model = SimpleNamespace(_dual_logits_by_rid={})
    entry = object.__new__(module._DecodeGraph)
    entry.model = model
    entry.replays = 0
    with torch.inference_mode():
        entry.input_ids = torch.zeros(1, dtype=torch.int64)
        entry.positions = torch.zeros_like(entry.input_ids)
        entry.lengths = torch.ones_like(entry.input_ids)
        entry.batch = SimpleNamespace(
            req_pool_indices=torch.zeros_like(entry.input_ids),
            out_cache_loc=torch.zeros_like(entry.input_ids),
        )
        text, audio, hidden = torch.ones(1, 5), torch.ones(1, 3), torch.ones(2, 1, 4)
        entry.outputs = MossSpeechModelOutput(
            next_token_logits=text,
            text_logits=text,
            audio_logits=audio,
            hidden_states=hidden,
        )
    entry.graph = Mock()
    actual_batch = SimpleNamespace(
        seq_lens=torch.tensor([42]),
        req_pool_indices=torch.tensor([1]),
        out_cache_loc=torch.tensor([17]),
        rids=["first"],
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )
    with torch.no_grad():
        assert not torch.is_inference_mode_enabled()
        output = entry.replay(actual_batch, torch.tensor([3]), torch.tensor([41]))
        assert not torch.is_inference_mode_enabled()
        # The engine sampler remains outside InferenceMode and mutates logits.
        output.next_token_logits.add_(5)
        output.hidden_states.add_(5)
        assert torch.equal(
            model._dual_logits_by_rid[("first", "decode")][0], torch.ones(5)
        )
        assert torch.equal(entry.outputs.text_logits, torch.ones(1, 5))
        actual_batch.rids = ["second"]
        second = entry.replay(actual_batch, torch.tensor([2]), torch.tensor([42]))
    assert entry.graph.replay.call_count == 2
    assert entry.replays == 2
    assert torch.equal(entry.input_ids, torch.tensor([2]))
    assert torch.equal(second.text_logits, torch.ones(1, 5))
    assert torch.equal(output.next_token_logits, torch.full((1, 5), 6.0))
