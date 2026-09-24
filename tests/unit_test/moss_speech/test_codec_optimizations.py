"""Exact codec preparation, overlap and configuration contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)
from tests.unit_test.moss_speech.test_streaming_codec import FakeFlow, FakeHift, voice


def make_codec(enabled: bool, chunk: int = 5) -> MossSpeechStreamingCodec:
    return MossSpeechStreamingCodec(
        None,
        device="cpu",
        chunk_size=chunk,
        flow=FakeFlow(),
        hift=FakeHift(),
        codec_optimizations=enabled,
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("channels", [[16], [16, 16]])
def test_cfm_reuses_masks_with_exact_outputs(
    streaming: bool, channels: list[int]
) -> None:
    from sglang_omni.models.moss_speech.components.cosyvoice_flow import (
        decoder as decoder_module,
    )
    from sglang_omni.models.moss_speech.components.cosyvoice_flow.flow_matching import (
        CausalConditionalCFM,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        estimator = decoder_module.CausalConditionalDecoder(
            in_channels=320,
            out_channels=80,
            channels=channels,
            dropout=0,
            attention_head_dim=8,
            n_blocks=1,
            num_mid_blocks=1,
            num_heads=2,
            act_fn="gelu",
            static_chunk_size=8,
            num_decoding_left_chunks=-1,
        ).eval()
        params = SimpleNamespace(
            sigma_min=1e-6,
            solver="euler",
            t_scheduler="cosine",
            training_cfg_rate=0.2,
            inference_cfg_rate=0.7,
        )
        cfm = CausalConditionalCFM(
            240, params, n_spks=1, spk_emb_dim=80, estimator=estimator
        ).eval()
        inputs = dict(
            mu=torch.randn(1, 80, 24),
            mask=torch.ones(1, 1, 24),
            spks=torch.randn(1, 80),
            cond=torch.randn(1, 80, 24),
            n_timesteps=3,
            streaming=streaming,
        )
        inputs["mask"][:, :, -2:] = 0
        original = decoder_module.add_optional_chunk_mask
        cfm.inference_optimizations = False
        with patch.object(
            decoder_module, "add_optional_chunk_mask", wraps=original
        ) as calls:
            expected = cfm(**inputs)[0]
            baseline_calls = calls.call_count
        cfm.inference_optimizations = True
        with patch.object(
            decoder_module, "add_optional_chunk_mask", wraps=original
        ) as calls:
            actual = cfm(**inputs)[0]
            assert calls.call_count == len(channels)
            assert baseline_calls == 3 * (2 * len(channels) + 1)
        assert torch.equal(actual, expected)
        # A different mask at the same shape must not reuse a prior request's bias.
        inputs["mask"][:, :, -4:] = 0
        changed = cfm(**inputs)[0]
        cfm.inference_optimizations = False
        assert torch.equal(changed, cfm(**inputs)[0])


def test_attention_bias_prepares_heads_once_per_solve() -> None:
    from sglang_omni.models.moss_speech.components.cosyvoice_flow.decoder import (
        CausalConditionalDecoder,
    )

    model = CausalConditionalDecoder(
        320,
        80,
        channels=[16],
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=2,
        attention_head_dim=8,
        act_fn="gelu",
        static_chunk_size=8,
    ).eval()
    x, mask = torch.zeros(2, 24, 16), torch.ones(2, 1, 24)
    mask[1, :, -2:] = 0
    cache = {}
    expected = model._attention_bias(x, mask, True, None).repeat_interleave(2, dim=0)
    actual = model._attention_bias(x, mask, True, cache)
    assert torch.equal(actual, expected)
    assert model._attention_bias(x, mask, True, cache) is actual
    assert len(cache) == 1


def test_decoder_reuses_hift_without_loading_speaker_model() -> None:
    from sglang_omni.models.moss_speech.components.hf_codec import modeling

    flow = torch.nn.Identity()
    flow.input_frame_rate = 12.5
    shared = torch.nn.Linear(2, 2)
    with (
        patch.object(modeling, "_build_flow_decoder", return_value=flow),
        patch.object(
            modeling, "_build_hift", side_effect=AssertionError("duplicate HiFT")
        ),
        patch.object(modeling.torch, "load", return_value={}),
        patch.object(
            modeling.onnxruntime,
            "InferenceSession",
            side_effect=AssertionError("unused speaker model"),
        ),
    ):
        decoder = modeling.AudioDecoder(
            "flow",
            "hift",
            "speaker",
            device="cpu",
            shared_hift=shared,
            load_speaker_encoder=False,
        )
    assert decoder.hift is shared
    assert decoder.campplus_session is None
    assert not decoder.hift.training


def test_optimized_decoder_adapter_skips_unused_voice_enrollment() -> None:
    from sglang_omni.models.moss_speech.components import codec_adapter

    cls = codec_adapter.MossSpeechCodecAdapter
    with (
        patch.object(cls, "_validate_codec_dir"),
        patch.object(cls, "_load_decoder"),
        patch.object(codec_adapter, "CampplusSpeakerEncoder") as speaker,
    ):
        adapter = cls("local", load_encoder=False, codec_optimizations=True)
        assert adapter._speaker_encoder is None
        speaker.assert_not_called()
    with (
        patch.object(cls, "_validate_codec_dir"),
        patch.object(cls, "_load_decoder"),
        patch.object(codec_adapter, "CampplusSpeakerEncoder") as speaker,
    ):
        cls("local", load_encoder=False, codec_optimizations=False)
        speaker.assert_called_once()
