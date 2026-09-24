"""Finite new-policy admission before independent CUDA numerical qualification."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.moss_speech.components import cfm_full_graph as graphs
from sglang_omni.models.moss_speech.components import cfm_prefix_cache as prefix
from sglang_omni.models.moss_speech.components.cosyvoice_flow import cfm_attention

from . import test_cfm_direct_lt as lt
from .test_cfm_graph_codec_bridge import outcome
from .test_cfm_prefix_bank import fake_flow
from .test_cfm_prefix_cache import make_codec, voice

SHAPES = (1392, 1424, 1500)


@pytest.mark.parametrize("length", [1392, 1424])
def test_new_bank_is_finite_and_only_active_prepared_shape_is_admitted(length):
    bank = prefix.PrefixCacheBank(fake_flow(), "voice", SHAPES, {})
    handler = bank.modules[0][1].processor._prefix_cache_handler
    assert not cfm_attention.shape_supported(length, 100)
    assert not handler.shape_supported(length, 100)
    with bank.scope("voice", length, False, {}):
        assert handler.shape_supported(length, 100)
        assert not handler.shape_supported(length, 20)
        assert not handler.shape_supported(1424 if length == 1392 else 1392, 100)
    for supplied_voice, final in (("other", False), ("voice", True)):
        with bank.scope(supplied_voice, length, final, {}):
            assert not handler.shape_supported(length, 100)
    bank.close()


@pytest.mark.parametrize(
    "shapes",
    [
        (1392,),
        (1424,),
        (1392, 1424),
        (1392, 1400, 1424, 1500),
        (1392, 1424, 1500, 1500),
        (1392, 1425, 1500),
    ],
)
def test_partial_mixed_duplicate_or_unmeasured_shape_sets_are_rejected(shapes):
    with pytest.raises(ValueError):
        prefix.CacheBinding("voice", shapes, {}, ())


@pytest.mark.parametrize("length", [1392, 1400, 1424])
def test_nonstandard_bank_build_and_backend_fallback_use_original_full_sdpa(
    monkeypatch, length
):
    shapes = (1400, 1500) if length == 1400 else SHAPES
    bank = prefix.PrefixCacheBank(fake_flow(), "voice", shapes, {})
    name = bank.names[0]
    query = torch.zeros(2, 8, length, 64)
    mask = torch.zeros(1).expand(2, 8, length, length)
    segments = (
        (query[..., :640, :], query, query, mask),
        (query[..., 640:, :], query, query, mask),
    )
    calls = []

    def original(q, k, v, **kwargs):
        assert q is query and k is query and v is query
        assert kwargs == dict(attn_mask=mask, dropout_p=0.0, is_causal=False)
        calls.append("full")
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", original)
    monkeypatch.setattr(
        cfm_attention,
        "_segmented_attention",
        Mock(side_effect=AssertionError("unqualified segmented route")),
    )
    plan = SimpleNamespace(length=length, chunk_size=100)
    bank.mode = "build"
    bank.ledger = prefix.PrefixCallLedger(bank.names)
    assert bank.attention(name, query, segments, plan, mask) is query
    assert len(bank.entries) == 1
    bank.mode = "cached"
    bank.ledger = prefix.PrefixCallLedger(bank.names)
    monkeypatch.setattr(
        torch,
        "_fused_sdp_choice",
        lambda *args: int(torch.nn.attention.SDPBackend.MATH),
    )
    assert bank.attention(name, query, segments, plan, mask) is query
    assert calls == ["full", "full"] and bank.hits == 0 and bank.fallbacks == 1
    bank.close()


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, (1500,)),
        ({"small_first_chunk": True}, (1400, 1500)),
        ({"stream_startup_schedule": "6_8_align25"}, SHAPES),
    ],
)
def test_actual_prefix_prepare_selects_policy_shapes_after_manifest_validation(
    monkeypatch, options, expected
):
    codec = make_codec(chunk_size=25, codec_optimizations=True, **options)
    codec.device = torch.device("cuda:0")
    codec.codec_path = Path("/codec")
    codec.flow.modules = lambda: ()
    prepared = SimpleNamespace(
        manifest_sha256="m", voice_digest="v", verified_files={}, source_sha256={}
    )
    environment = dict(
        device_name="NVIDIA A800-SXM4-80GB",
        capability=[8, 0],
        torch="2.9.1+cu128",
        cuda="12.8",
    )
    monkeypatch.setattr(prefix, "runtime_environment", lambda device: environment)
    monkeypatch.setattr(prefix, "load_prepared_voice", Mock(return_value=prepared))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    calibration = Mock(return_value=outcome())
    monkeypatch.setattr(codec, "_calibrate_prepared_voice", calibration)
    seen = []

    def construct(flow, digest, shapes, env):
        assert flow is codec.flow and digest == "v" and env == environment
        seen.append(shapes)
        raise RuntimeError("stop after shape selection")

    monkeypatch.setattr(prefix, "PrefixCacheBank", construct)
    with pytest.raises(RuntimeError, match="shape selection"):
        codec.prepare_prefix_cache("manifest.json")
    assert seen == [expected] and calibration.call_count == 1
    assert codec._prefix_cache is None
    prefix.load_prepared_voice.side_effect = ValueError("source fingerprint changed")
    with pytest.raises(ValueError, match="source fingerprint"):
        codec.prepare_prefix_cache("manifest.json")
    assert seen == [expected] and calibration.call_count == 1


def test_real_36_code_calibration_covers_three_hops_and_restores_rng():
    codec = make_codec(
        chunk_size=25, codec_optimizations=True, stream_startup_schedule="6_8_align25"
    )
    prepared = SimpleNamespace(voice=voice(), warmup_audio_codes=tuple(range(36)))
    before = torch.get_rng_state().clone()
    first = codec._calibrate_prepared_voice(prepared)
    second = codec._calibrate_prepared_voice(prepared)
    assert codec._same_calibration(first, second)
    assert [wave.numel() for wave in first[0]] == [7680, 15360, 36480]
    assert [
        (row["available"], row["cursor_before"], row["final"]) for row in first[1]
    ] == [(9, 0, False), (17, 6, False), (36, 14, False)]
    assert not codec.sessions and torch.equal(before, torch.get_rng_state())
    codec.hift.inference = Mock(side_effect=RuntimeError("third preparation failed"))
    with pytest.raises(RuntimeError, match="preparation failed"):
        codec._calibrate_prepared_voice(prepared)
    assert not codec.sessions and torch.equal(before, torch.get_rng_state())


@pytest.fixture
def provider():
    fixture = lt.ProviderTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize(
    "length,rows", [(1392, 224), (1392, 2784), (1424, 288), (1424, 2848)]
)
def test_new_qkv_explicit_profile_abi_capability_and_tail_view(provider, length, rows):
    plan = lt.module.DirectLtProjection(
        lt.Tensor((512, 256)), None, rows, profile=f"qkv_{length}"
    )
    assert plan._algorithm_config == {
        0: 0,
        1: 16,
        2: 3,
        3: 1,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
    }
    assert 0 in provider.lib.capability_reads and 1 in provider.lib.capability_reads
    value = lt.Tensor((2, rows // 2, 256))
    if rows < 1000:
        value.custom_stride = (length * 256, 256, 1)
    plan.run(value)
    assert plan.input.copied_from is value and len(provider.lib.launches) == 1
    value.custom_stride = ((length + 1) * 256, 256, 1)
    with pytest.raises(ValueError, match="layout"):
        plan.run(value)
    plan.close()


@pytest.mark.parametrize("length,tail", [(1392, 224), (1424, 288)])
def test_new_biased_rows_retain_separate_full_tail_algorithms(provider, length, tail):
    for profile, k, out in (
        ("output", 512, 256),
        ("output", 1024, 256),
        ("ff_first", 256, 1024),
    ):
        for rows in (tail, 2 * length):
            plan = lt.module.DirectLtProjection(
                lt.Tensor((out, k)), lt.Tensor(out), rows, profile=profile
            )
            expected_tile = 14 if profile == "output" else 15 if rows == tail else 20
            assert plan._algorithm_config[1] == expected_tile
            assert plan._algorithm_config[2] == 1 and plan._algorithm_config[3] == 0
            plan.run(lt.Tensor((2, rows // 2, k)))
            plan.close()


@pytest.mark.parametrize(
    "profile,rows",
    [
        ("qkv_1392", 288),
        ("qkv_1424", 224),
        ("qkv_1400", 224),
        ("qkv_1500", 288),
        ("qkv_1392", 2800),
        ("qkv_1393", 224),
    ],
)
def test_new_qkv_profiles_cannot_borrow_other_length_admission(provider, profile, rows):
    with pytest.raises(ValueError):
        lt.module.DirectLtProjection(lt.Tensor((512, 256)), None, rows, profile=profile)
    assert provider.lib.calls == []


@pytest.mark.parametrize("field", ["split_k_support", "reduction_scheme_mask"])
def test_new_qkv_missing_algorithm_capability_fails_before_launch(provider, field):
    setattr(provider.lib, field, 0)
    with pytest.raises(RuntimeError):
        lt.module.DirectLtProjection(
            lt.Tensor((512, 256)), None, 224, profile="qkv_1392"
        )
    assert not provider.lib.launches


def test_partial_third_graph_failure_closes_created_owners_without_touching_prefix(
    monkeypatch,
):
    class Cfm:
        def solve_euler(self, *args, **kwargs):
            return "native"

    cfm = Cfm()
    bank = SimpleNamespace(
        flow=SimpleNamespace(decoder=cfm),
        binding=SimpleNamespace(shapes=SHAPES, valid=True),
        active_length=None,
        mode="cached",
        hits=0,
    )
    owner = graphs.FullCfmGraphBank(cfm, bank)
    events = []
    codec = make_codec(
        chunk_size=25, codec_optimizations=True, stream_startup_schedule="6_8_align25"
    )
    codec.flow = bank.flow
    codec.device = torch.device("cuda:0")
    codec._prefix_cache, codec._prepared_voice = bank, object()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(codec, "_calibrate_prepared_voice", lambda value: outcome())

    def prepare(calibration):
        for n in SHAPES[:2]:
            owner._graphs[n] = SimpleNamespace(close=lambda n=n: events.append(n))
        owner._lt_context = SimpleNamespace(
            device=codec.device, close=lambda: events.append("lt.close")
        )
        owner._samples[1500] = object()
        owner._tail_plans[(1500, "layer", "q")] = object()
        with graphs.temporary_attribute(cfm, "solve_euler", lambda: "temporary"):
            raise RuntimeError("third shape capture failed")

    monkeypatch.setattr(owner, "prepare", prepare)
    monkeypatch.setattr(graphs, "FullCfmGraphBank", lambda cfm, prefix: owner)
    with pytest.raises(RuntimeError, match="third shape"):
        codec.prepare_full_graphs()
    assert events == [1392, 1424, "lt.close"]
    assert (
        owner._closed
        and not owner._graphs
        and not owner._samples
        and not owner._tail_plans
    )
    assert cfm.solve_euler() == "native" and "solve_euler" not in vars(cfm)
    assert bank.binding.valid and codec._prefix_cache is bank
    assert codec.full_graph_status() == {"enabled": False}


@pytest.mark.parametrize("actual_replays", [2, 3, 4])
def test_three_shape_startup_requires_exact_replay_count_before_ready(
    monkeypatch, actual_replays
):
    codec = make_codec(
        chunk_size=25, codec_optimizations=True, stream_startup_schedule="6_8_align25"
    )
    prepared = SimpleNamespace(voice=voice(), warmup_audio_codes=tuple(range(36)))
    native = codec._calibrate_prepared_voice(prepared)
    bank = SimpleNamespace(
        binding=SimpleNamespace(shapes=SHAPES, valid=True), active_length=None, hits=0
    )
    codec._prefix_cache, codec._prepared_voice = bank, prepared
    codec.device = torch.device("cuda:0")
    codec.flow.decoder = object()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    owner = Mock()
    counter, calls = [0], []
    owner.status.side_effect = lambda: dict(
        enabled=True, shapes=list(SHAPES), graph_replays=counter[0], bytes=0
    )
    owner.prepare.return_value = {"shapes": list(SHAPES)}
    monkeypatch.setattr(graphs, "FullCfmGraphBank", Mock(return_value=owner))

    def calibrate(value):
        assert value is prepared
        calls.append(True)
        if len(calls) == 2:
            counter[0] += actual_replays
        return native

    monkeypatch.setattr(codec, "_calibrate_prepared_voice", calibrate)
    if actual_replays == 3:
        status = codec.prepare_full_graphs()
        assert status["startup"]["startup_graph_replays"] == 3
        assert len(status["startup"]["startup_ledger"]) == 3
        owner.close.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="replay"):
            codec.prepare_full_graphs()
        assert codec.full_graph_status() == {"enabled": False}
        owner.close.assert_called_once()
    assert len(calls) == 2
