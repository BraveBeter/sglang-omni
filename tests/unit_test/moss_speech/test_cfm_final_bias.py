"""Final native-mask layout admission, solve ownership and public opt-in."""

import importlib
import logging
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech.components.cosyvoice_flow import (
    cfm_attention as provenance,
)
from sglang_omni.models.moss_speech.components.cosyvoice_flow import decoder as native
from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechStreamingPipelineConfig,
)


def api():
    return importlib.import_module(
        "sglang_omni.models.moss_speech.components.cosyvoice_flow.final_bias"
    )


@pytest.fixture
def data(monkeypatch):
    module = api()
    decoder = object.__new__(native.CausalConditionalDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.attention_heads = 8
    decoder.static_chunk_size = 100
    decoder.eval()
    mask, proof = provenance.new_full_valid_mask(torch.empty(1), 1492)
    mask2 = mask.expand(2, -1, -1).clone()
    proof = provenance.expand_cfg_provenance(proof, mask, mask2)
    bias = torch.full((16, 1492, 1492), -0.0)
    x = torch.empty(2, 1492, 256)
    owner = module.FinalBiasAlignment()
    monkeypatch.setattr(module, "_runtime_supported", lambda tensor: True)
    return SimpleNamespace(
        module=module,
        decoder=decoder,
        mask=mask2,
        proof=proof,
        bias=bias,
        x=x,
        owner=owner,
        cache={},
    )


def apply(d, **changes):
    args = dict(
        decoder=d.decoder,
        x=d.x,
        mask=d.mask,
        bias=d.bias,
        proof=d.proof,
        streaming=False,
        cache=d.cache,
    )
    args.update(changes)
    return d.owner.apply(**args)


def test_owned_full_query_rows_solve_cache_and_no_tensor_reads(data, monkeypatch):
    d = data
    with torch.inference_mode():
        for name in ("equal",):
            monkeypatch.setattr(
                torch, name, Mock(side_effect=AssertionError("device read"))
            )
        monkeypatch.setattr(
            torch.Tensor, "item", Mock(side_effect=AssertionError("item"))
        )
        monkeypatch.setattr(
            torch.cuda, "synchronize", Mock(side_effect=AssertionError("sync"))
        )
        result = apply(d)
        assert result.shape == (16, 1492, 1492) and result.stride() == (0, 1496, 1)
        assert result.data_ptr() != d.bias.data_ptr()
        assert result.untyped_storage().nbytes() == 1492 * 1496 * 4
        assert apply(d) is result
        other = apply(d, cache={})
        assert other.data_ptr() != result.data_ptr()
    monkeypatch.undo()
    assert torch.equal(result.contiguous().view(torch.uint8), d.bias.view(torch.uint8))
    assert torch.signbit(result).all()
    assert d.owner.status()["prepared"] == 2 and d.owner.status()["reused"] == 1
    ref = weakref.ref(result)
    del result
    d.cache.clear()
    assert ref() is None
    assert not any(isinstance(v, torch.Tensor) for v in vars(d.owner).values())


@pytest.mark.parametrize(
    "case",
    [
        "streaming",
        "no_proof",
        "fake_proof",
        "mask_alias",
        "no_cache",
        "layout",
        "dtype",
        "length",
        "subclass",
        "method",
        "mask_function",
        "training",
        "grad",
        "policy",
        "runtime",
        "closed",
    ],
)
def test_unknown_context_returns_original_bias(data, monkeypatch, case):
    d = data
    kw = {}
    if case == "streaming":
        kw["streaming"] = True
    elif case == "no_proof":
        kw["proof"] = None
    elif case == "fake_proof":
        kw["proof"] = SimpleNamespace(matches=lambda _: True)
    elif case == "mask_alias":
        kw["mask"] = d.mask.view_as(d.mask)
    elif case == "no_cache":
        kw["cache"] = None
    elif case == "layout":
        kw["bias"] = d.bias.transpose(1, 2)
    elif case == "dtype":
        kw["x"] = d.x.to(torch.bfloat16)
    elif case == "length":
        kw["x"] = d.x[:, :1400]
    elif case == "subclass":

        class Custom(native.CausalConditionalDecoder):
            pass

        custom = object.__new__(Custom)
        torch.nn.Module.__init__(custom)
        custom.eval()
        custom.attention_heads = 8
        kw["decoder"] = custom
    elif case == "method":
        monkeypatch.setattr(d.decoder, "_attention_bias", lambda *args: d.bias)
    elif case == "mask_function":
        monkeypatch.setattr(native, "mask_to_bias", lambda *args: d.bias)
    elif case == "training":
        d.decoder.train()
    elif case == "policy":
        monkeypatch.setattr(d.module, "_execution_context", lambda: ("changed",))
    elif case == "runtime":
        monkeypatch.setattr(d.module, "_runtime_supported", lambda tensor: False)
    elif case == "closed":
        d.owner.close()
    with torch.set_grad_enabled(case == "grad"):
        assert apply(d, **kw) is kw.get("bias", d.bias)
    assert d.owner.status()["prepared"] == 0 and not d.cache
    assert d.owner.status()["fallback_calls"] == 1


def test_runtime_qualification_does_not_admit_cpu():
    assert not api()._runtime_supported(torch.empty(1))


def test_public_flag_defaults_and_streaming_merge():
    for cls in (MossSpeechPipelineConfig, MossSpeechStreamingPipelineConfig):
        config = cls(model_path="local")
        assert config.codec_final_bias_alignment is None
    with pytest.raises(ValueError):
        MossSpeechPipelineConfig(model_path="local", codec_final_bias_alignment=True)
    config = MossSpeechStreamingPipelineConfig(
        model_path="local",
        codec_final_bias_alignment=True,
        codec_optimizations=True,
        stream_startup_schedule="6_8_align25",
        prefill_full_graphs=True,
    )
    changed = ConfigManager(config).merge_config({"codec_final_bias_alignment": False})
    stages = {s.name: s for s in changed.stages}
    assert stages["audio_vocoder"].factory_args["codec_final_bias_alignment"] is False
    assert (
        stages["audio_vocoder"].factory_args["stream_startup_schedule"] == "6_8_align25"
    )
    assert stages["ar_engine"].factory_args["prefill_full_graphs"] is True
    assert config.codec_final_bias_alignment is True


def test_constructor_close_is_cumulative_and_idempotent(data, caplog):
    d = data
    args = dict(
        codec_path=None,
        device="cpu",
        chunk_size=25,
        flow=SimpleNamespace(decoder=SimpleNamespace(estimator=d.decoder)),
        hift=object(),
    )
    with pytest.raises(ValueError):
        MossSpeechStreamingCodec(**args, codec_final_bias_alignment=True)
    codec = MossSpeechStreamingCodec(
        **args, codec_final_bias_alignment=True, codec_optimizations=True
    )
    owner = codec._final_bias_alignment
    assert d.decoder._final_bias_alignment is owner
    with torch.inference_mode():
        owner.apply(
            decoder=d.decoder,
            x=d.x,
            mask=d.mask,
            bias=d.bias,
            proof=d.proof,
            streaming=False,
            cache=d.cache,
        )
    with caplog.at_level(logging.INFO):
        codec.close()
        codec.close()
    rows = [
        r
        for r in caplog.records
        if "MOSS final bias alignment closed:" in r.getMessage()
    ]
    assert len(rows) == 1 and '"prepared": 1' in rows[0].getMessage()
    assert owner.status()["closed"] is True
    assert codec.final_bias_alignment_status()["persistent_tensor_bytes"] == 0


def test_actual_decoder_three_sites_use_original_constructor_and_shared_view(
    data, monkeypatch
):
    d = data
    with torch.random.fork_rng(devices=[]):
        decoder = native.CausalConditionalDecoder(
            in_channels=320,
            out_channels=80,
            channels=(256,),
            n_blocks=1,
            num_mid_blocks=1,
            num_heads=8,
            attention_head_dim=64,
            dropout=0,
            act_fn="gelu",
            static_chunk_size=100,
        ).eval()
    masks = []
    for group in (decoder.down_blocks, decoder.mid_blocks, decoder.up_blocks):
        for item in group:
            for block in item[1]:

                def forward(*, hidden_states, attention_mask, **kwargs):
                    masks.append(attention_mask)
                    return hidden_states

                monkeypatch.setattr(block, "forward", forward)
    decoder._final_bias_alignment = d.owner
    original = decoder._attention_bias.__func__
    x = torch.zeros(2, 80, 1492)
    with torch.inference_mode():
        decoder(
            x,
            d.mask,
            x,
            torch.zeros(2),
            torch.zeros(2, 80),
            x,
            streaming=False,
            attention_bias_cache=d.cache,
            full_valid_mask=d.proof,
        )
    assert decoder._attention_bias.__func__ is original
    assert len(masks) == 3 and all(m is masks[0] for m in masks)
    assert masks[0].stride() == (0, 1496, 1)
    assert d.owner.status()["prepared"] == 1 and d.owner.status()["reused"] == 2


def test_dynamic_cli_and_streaming_factory_preserve_independence(monkeypatch):
    from pathlib import Path

    from typer.testing import CliRunner

    from sglang_omni.cli import app

    launch = Mock()
    monkeypatch.setattr(
        importlib.import_module("sglang_omni.cli.serve"), "launch_server", launch
    )
    repo = Path(__file__).resolve().parents[3]
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--config",
            str(repo / "examples/configs/moss_speech_streaming.yaml"),
            "--codec-final-bias-alignment",
            "true",
            "--codec-optimizations",
            "true",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    config = launch.call_args.args[0]
    assert (
        next(s for s in config.stages if s.name == "audio_vocoder").factory_args[
            "codec_final_bias_alignment"
        ]
        is True
    )
    streaming = importlib.import_module("sglang_omni.models.moss_speech.streaming")
    stages = importlib.import_module("sglang_omni.models.moss_speech.stages")
    codec_module = importlib.import_module(
        "sglang_omni.models.moss_speech.components.streaming_codec"
    )
    build = Mock(return_value=SimpleNamespace(hift=object()))
    monkeypatch.setattr(codec_module, "MossSpeechStreamingCodec", build)
    monkeypatch.setattr(stages, "_resolve_codec_dir", Mock(return_value="codec"))
    monkeypatch.setattr(
        stages,
        "create_audio_vocoder_executor",
        Mock(return_value=SimpleNamespace(_fn=lambda *args: None)),
    )
    monkeypatch.setattr(streaming, "MossSpeechStreamingVocoder", Mock())
    streaming.create_streaming_audio_executor(
        "local",
        chunk_size=25,
        codec_optimizations=True,
        codec_final_bias_alignment=True,
    )
    assert build.call_args.kwargs["codec_final_bias_alignment"] is True
    assert build.call_args.kwargs.get("codec_full_graphs", False) is False
    assert build.call_args.kwargs.get("small_first_chunk", False) is False
    build.reset_mock()
    streaming.create_streaming_audio_executor(
        "local", chunk_size=25, codec_optimizations=True
    )
    assert "codec_final_bias_alignment" not in build.call_args.kwargs
