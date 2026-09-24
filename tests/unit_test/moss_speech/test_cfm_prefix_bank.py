"""Per-instance installation, CPU identity, and startup-only cache lifetime."""

from __future__ import annotations

import pytest
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from sglang_omni.models.moss_speech.components.cfm_prefix_cache import (
    PrefixCacheBank,
    cpu_voice_digest,
)
from sglang_omni.models.moss_speech.components.cosyvoice_flow.cfm_attention import (
    MossCfmAttnProcessor,
)

from .test_cfm_prefix_cache import make_codec, voice


def fake_flow():
    flow = torch.nn.Module()
    flow.decoder = torch.nn.Module()
    flow.decoder.estimator = torch.nn.ModuleList(
        [Attention(query_dim=8, heads=1, dim_head=8) for _ in range(56)]
    )
    for layer in flow.decoder.estimator:
        layer.set_processor(MossCfmAttnProcessor(AttnProcessor2_0()))
    flow.decoder.rand_noise = torch.zeros(1, 80, 1500)
    return flow


def test_voice_identity_uses_exact_cpu_bits_without_gpu_transfer():
    first = voice()
    assert cpu_voice_digest(first) == cpu_voice_digest(
        {k: v.clone() for k, v in first.items()}
    )
    changed = {k: v.clone() for k, v in first.items()}
    changed["embedding"][0, 0] = -0.0
    assert cpu_voice_digest(first) != cpu_voice_digest(changed)
    assert cpu_voice_digest({k: v.to("meta") for k, v in first.items()}) is None


def test_bank_install_is_per_instance_and_close_restores_exact_processors():
    flow, other = fake_flow(), fake_flow()
    originals = [layer.processor for layer in flow.decoder.estimator]
    bank = PrefixCacheBank(flow, "voice", (1400, 1500), {"runtime": "test"})
    assert all(a.processor is not b for a, b in zip(flow.decoder.estimator, originals))
    assert all(
        type(layer.processor) is MossCfmAttnProcessor
        for layer in other.decoder.estimator
    )
    bank.close()
    bank.close()
    assert all(a.processor is b for a, b in zip(flow.decoder.estimator, originals))
    assert not bank.binding.valid and not bank.entries


def test_scope_never_constructs_on_unknown_voice_and_resets_after_exception():
    flow = fake_flow()
    bank = PrefixCacheBank(flow, "voice", (1400,), {})
    with bank.scope("other", 1400, False, {}):
        assert bank.active_length is None
    assert not bank.entries
    with pytest.raises(RuntimeError, match="failure"):
        with bank.scope("voice", 1400, False, {}):
            assert bank.active_length == 1400
            raise RuntimeError("failure")
    assert bank.active_length is None
    bank.close()


def test_mutated_weights_disable_cache_before_any_attention():
    flow = fake_flow()
    bank = PrefixCacheBank(flow, "voice", (1500,), {})
    with torch.no_grad():
        next(flow.parameters()).add_(1)
    with bank.scope("voice", 1500, False, {}):
        assert bank.active_length is None
    assert not bank.binding.valid
    bank.close()


def test_default_codec_has_no_bank_and_prepare_rejects_live_sessions():
    codec = make_codec(chunk_size=25, codec_optimizations=True)
    assert codec.prefix_cache_status()["enabled"] is False
    codec.begin("live", voice(), seed=7)
    with pytest.raises(ValueError, match="active"):
        codec.prepare_prefix_cache("missing.json")
    assert "live" in codec.sessions
    codec.close()
    assert not codec.sessions


def test_codec_attaches_voice_digest_only_when_explicit_bank_exists():
    codec = make_codec(chunk_size=25)
    codec.begin("plain", voice(), seed=7)
    assert codec.sessions["plain"].prefix_voice_digest is None

    class Bank:
        def status(self):
            return dict(enabled=True, attention_hits=0)

        def close(self):
            pass

    codec._prefix_cache = Bank()
    codec.begin("cached", voice(), seed=19)
    assert codec.sessions["cached"].prefix_voice_digest == cpu_voice_digest(voice())
    codec.close()


def test_noncanonical_float_conditioning_cannot_match_by_lossy_cast():
    inputs = voice()
    inputs["embedding"] = inputs["embedding"].double()
    assert cpu_voice_digest(inputs) is None


def test_cpu_identity_is_computed_before_any_device_transfer(monkeypatch):
    from sglang_omni.models.moss_speech.components import cfm_prefix_cache as api

    codec = make_codec(chunk_size=25)
    codec._prefix_cache = object()
    seen = []
    original = torch.Tensor.to
    monkeypatch.setattr(
        api, "cpu_voice_digest", lambda value: seen.append("digest") or "bits"
    )

    def transfer(tensor, *args, **kwargs):
        assert seen == ["digest"]
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    codec.begin("r", voice(), seed=7)
    assert codec.sessions["r"].prefix_voice_digest == "bits"


def test_cache_lifecycle_logs_only_scalars_and_digests_before_release(caplog):
    import json
    import logging

    codec = make_codec(chunk_size=25)

    class Bank:
        def status(self):
            return dict(
                enabled=True,
                shapes=[1400, 1500],
                entries=1120,
                bytes=123,
                attention_hits=2240,
                eligible_solves=6,
                fallback_calls=2,
            )

        def close(self):
            assert any(
                "MOSS prefix cache closed:" in row.message for row in caplog.records
            )

    codec._prefix_cache = Bank()
    codec._prefix_cache_receipt = dict(
        voice_digest="voice-sha",
        manifest_sha256="manifest-sha",
        startup_attention_hits=1120,
        total_prepare_seconds=4.0,
        native_codec_and_hift_warmup_seconds=1.0,
        cache_build_codec_seconds=1.2,
        cached_codec_validation_seconds=0.9,
        peak_allocated_bytes=1000,
        ready_allocated_bytes=900,
        accidental_tensor=torch.tensor([999.0]),
    )
    with caplog.at_level(logging.INFO):
        codec._log_prefix_cache("ready")
        codec.close()
        codec.close()
    rows = [
        row.message for row in caplog.records if "MOSS prefix cache " in row.message
    ]
    assert len(rows) == 2
    for row in rows:
        payload = json.loads(row.split(": ", 1)[1])
        assert payload["request_attention_hits"] == 1120
        assert payload["voice_digest"] == "voice-sha"
        assert "accidental_tensor" not in payload and "tensor(" not in row


def test_actual_decoder_keeps_mask_provenance_after_bank_install(monkeypatch):
    from sglang_omni.models.moss_speech.components.cosyvoice_flow import (
        cfm_attention as api,
    )
    from sglang_omni.models.moss_speech.components.cosyvoice_flow.decoder import (
        CausalConditionalDecoder,
    )

    decoder = CausalConditionalDecoder(
        in_channels=320,
        out_channels=80,
        channels=(32,),
        dropout=0,
        attention_head_dim=64,
        n_blocks=28,
        num_mid_blocks=0,
        num_heads=8,
        act_fn="gelu",
        static_chunk_size=100,
    ).eval()
    api.install_cfm_processors(decoder)
    flow = torch.nn.Module()
    flow.decoder = torch.nn.Module()
    flow.decoder.estimator = decoder
    bank = PrefixCacheBank(flow, "voice", (1400, 1500), {})
    seen = []

    def record(self, length, chunk_size):
        seen.append((length, chunk_size))
        return False

    monkeypatch.setattr(api.MossCfmAttnProcessor, "_shape_supported", record)
    mask, proof = api.new_full_valid_mask(torch.empty(1, 80, 24), 24)
    cfg = mask.expand(2, -1, -1).clone()
    proof = api.expand_cfg_provenance(proof, mask, cfg)
    with torch.inference_mode():
        decoder(
            torch.randn(2, 80, 24),
            cfg,
            torch.randn(2, 80, 24),
            torch.zeros(2),
            spks=torch.randn(2, 80),
            cond=torch.randn(2, 80, 24),
            streaming=True,
            full_valid_mask=proof,
        )
    assert seen == [(24, 100)] * 56
    bank.close()


def test_changed_cudnn_enable_context_falls_back_before_attention():
    flow = fake_flow()
    bank = PrefixCacheBank(flow, "voice", (1500,), {})
    with torch.backends.cudnn.flags(enabled=not torch.backends.cudnn.enabled):
        with bank.scope("voice", 1500, False, {}):
            assert bank.active_length is None
    bank.close()
