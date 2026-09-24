"""Explicit factory activation and bank ownership preserve the default path."""

import builtins
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.models.moss_speech import stages, streaming
from sglang_omni.models.moss_speech.components import cfm_prefix_cache, streaming_codec
from sglang_omni.models.moss_speech.config import MossSpeechStreamingPipelineConfig


def fake_codec():
    return streaming_codec.MossSpeechStreamingCodec(
        None,
        device="cpu",
        chunk_size=25,
        codec_optimizations=True,
        flow=SimpleNamespace(),
        hift=object(),
    )


def test_factory_default_does_not_import_or_read_package(monkeypatch):
    calls = []
    codec = SimpleNamespace(
        hift=object(),
        close=Mock(),
        prepare_prefix_cache=lambda p: calls.append("prefix"),
        prepare_full_graphs=lambda: calls.append("graphs"),
        prepare_native_epoch=Mock(side_effect=AssertionError("disabled")),
    )
    monkeypatch.setattr(
        streaming_codec, "MossSpeechStreamingCodec", lambda *a, **k: codec
    )
    monkeypatch.setattr(stages, "_resolve_codec_dir", lambda *a: ".")
    monkeypatch.setattr(
        stages,
        "create_audio_vocoder_executor",
        lambda *a, **k: SimpleNamespace(_fn=None),
    )
    monkeypatch.setattr(streaming, "MossSpeechStreamingVocoder", lambda c, *a, **k: c)
    original = builtins.__import__

    def guarded(name, *a, **k):
        if "cfm_native_epoch" in name or "cfm_epoch_package" in name:
            raise AssertionError("disabled import")
        return original(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert (
        streaming.create_streaming_audio_executor(
            "local",
            chunk_size=25,
            codec_optimizations=True,
            codec_prefix_cache_manifest="prefix.json",
            codec_full_graphs=True,
        )
        is codec
    )
    assert calls == ["prefix", "graphs"]


@pytest.mark.parametrize("failure", [False, True])
def test_factory_prepares_after_graphs_and_closes_on_failure(monkeypatch, failure):
    calls = []

    def prepare(path):
        calls.append(("native", path))
        if failure:
            raise ValueError("native prepare failed")

    codec = SimpleNamespace(
        hift=object(),
        close=lambda: calls.append("close"),
        prepare_prefix_cache=lambda p: calls.append("prefix"),
        prepare_full_graphs=lambda: calls.append("graphs"),
        prepare_native_epoch=prepare,
    )
    monkeypatch.setattr(
        streaming_codec, "MossSpeechStreamingCodec", lambda *a, **k: codec
    )
    monkeypatch.setattr(stages, "_resolve_codec_dir", lambda *a: ".")
    monkeypatch.setattr(
        stages,
        "create_audio_vocoder_executor",
        lambda *a, **k: SimpleNamespace(_fn=None),
    )
    monkeypatch.setattr(streaming, "MossSpeechStreamingVocoder", lambda c, *a, **k: c)
    context = (
        pytest.raises(ValueError, match="native prepare") if failure else nullcontext()
    )
    with context:
        streaming.create_streaming_audio_executor(
            "local",
            chunk_size=25,
            codec_optimizations=True,
            codec_prefix_cache_manifest="prefix.json",
            codec_full_graphs=True,
            codec_native_epoch=True,
            codec_native_epoch_manifest="native.json",
        )
    assert calls == ["prefix", "graphs", ("native", "native.json")] + (
        ["close"] if failure else []
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(codec_native_epoch=True),
        dict(codec_native_epoch=1, codec_native_epoch_manifest="x"),
        dict(codec_native_epoch=False, codec_native_epoch_manifest="x"),
        dict(codec_native_epoch=True, codec_native_epoch_manifest="x", chunk_size=5),
        dict(
            codec_native_epoch=True,
            codec_native_epoch_manifest="x",
            codec_optimizations=False,
        ),
    ],
)
def test_invalid_option_before_loading(monkeypatch, kwargs):
    monkeypatch.setattr(
        stages, "_resolve_codec_dir", Mock(side_effect=AssertionError("loaded"))
    )
    values = dict(
        chunk_size=25,
        codec_optimizations=True,
        codec_prefix_cache_manifest="prefix.json",
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        streaming.create_streaming_audio_executor("local", **values)


def test_config_passes_only_to_vocoder():
    config = MossSpeechStreamingPipelineConfig(
        model_path="local",
        codec_native_epoch=True,
        codec_native_epoch_manifest="native.json",
    )
    for stage in config.stages:
        assert ("codec_native_epoch" in stage.factory_args) == (
            stage.name == "audio_vocoder"
        )


def test_bank_live_owner_and_original_identity_and_detach(monkeypatch):
    bank = object.__new__(cfm_prefix_cache.PrefixCacheBank)
    bank.flow = object()
    bank.active_length = None
    bank.modules = ()
    bank.entries = {}
    bank.binding = SimpleNamespace(invalidate=Mock())
    owner = SimpleNamespace(
        flow=bank.flow, active=True, closed=False, snapshot=Mock(return_value=("live",))
    )
    bank.install_epoch_owner(owner)
    assert bank.current_epoch() == ("live",)
    owner.snapshot.assert_called_once_with(bank.flow, cfm_prefix_cache.tensor_epoch)

    def changed(f):
        return ("changed",)

    monkeypatch.setattr(cfm_prefix_cache, "tensor_epoch", changed)
    bank.current_epoch()
    assert owner.snapshot.call_args.args == (bank.flow, changed)
    bank.close()
    assert bank._epoch_owner is None


def test_codec_close_all_graph_bank_before_owner_and_idempotent(monkeypatch):
    codec = fake_codec()
    order = []
    codec._log_full_graphs = codec._log_prefix_cache = lambda event: None
    codec._full_graphs = SimpleNamespace(close=lambda: order.append("graph"))
    codec._prefix_cache = SimpleNamespace(close=lambda: order.append("bank"))
    codec._native_epoch = SimpleNamespace(
        close=lambda: order.append("native"), status=lambda: {"closed": True}
    )
    codec.close()
    codec.close()
    assert order == ["graph", "bank", "native"]


def test_native_prepare_requires_no_sessions_and_prepared_bank():
    codec = fake_codec()
    with pytest.raises(ValueError, match="prepared"):
        codec.prepare_native_epoch("x")
    codec._prefix_cache = object()
    codec.sessions["active"] = object()
    with pytest.raises(ValueError, match="session"):
        codec.prepare_native_epoch("x")


def test_native_prepare_failure_closes_uninstalled_owner(monkeypatch):
    from sglang_omni.models.moss_speech.components import cfm_native_epoch

    codec = fake_codec()
    codec._prefix_cache = SimpleNamespace(install_epoch_owner=Mock())
    codec._prepared_voice = object()
    owner = SimpleNamespace(prepare=Mock(side_effect=ValueError("tuple")), close=Mock())
    monkeypatch.setattr(cfm_native_epoch, "NativeEpochOwner", lambda *a, **k: owner)
    with pytest.raises(ValueError, match="tuple"):
        codec.prepare_native_epoch("x")
    owner.close.assert_called_once()
    codec._prefix_cache.install_epoch_owner.assert_not_called()
    assert codec._native_epoch is None


def test_every_scope_invokes_current_live_epoch_and_retains_invalidation():
    bank = object.__new__(cfm_prefix_cache.PrefixCacheBank)
    bank.flow = object()
    bank.active_length = None
    bank.mode = "cached"
    bank.fallbacks = 0
    bank.binding = SimpleNamespace(
        epoch=("before",), invalidate=Mock(), matches=lambda *args: False
    )
    owner = SimpleNamespace(
        flow=bank.flow,
        active=True,
        closed=False,
        snapshot=Mock(return_value=("before",)),
    )
    bank.install_epoch_owner(owner)
    for _ in range(3):
        with bank.scope("voice", 1392, False, {}):
            pass
    assert owner.snapshot.call_count == 3
    owner.snapshot.return_value = ("changed",)
    with bank.scope("voice", 1392, False, {}):
        pass
    assert owner.snapshot.call_count == 4 and bank.fallbacks == 4
    bank.binding.invalidate.assert_called_once()


def test_native_prepare_success_then_double_prepare_and_graph_failure_keep_owner(
    monkeypatch,
):
    from sglang_omni.models.moss_speech.components import cfm_native_epoch

    codec = fake_codec()
    codec._prepared_voice = object()
    bank = SimpleNamespace(install_epoch_owner=Mock())
    codec._prefix_cache = bank
    owner = SimpleNamespace(
        prepare=Mock(), close=Mock(), status=lambda: {"enabled": True}
    )
    monkeypatch.setattr(cfm_native_epoch, "NativeEpochOwner", lambda *a, **k: owner)
    codec.prepare_native_epoch("package.json")
    assert codec._native_epoch is owner
    bank.install_epoch_owner.assert_called_once_with(owner)
    with pytest.raises(ValueError, match="already"):
        codec.prepare_native_epoch("package.json")
    codec._log_full_graphs = lambda *a: None
    codec._full_graphs = SimpleNamespace(
        close=Mock(side_effect=RuntimeError("busy graph"))
    )
    with pytest.raises(RuntimeError, match="busy graph"):
        codec.close()
    owner.close.assert_not_called()
    assert codec._native_epoch is owner
