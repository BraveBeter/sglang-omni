# SPDX-License-Identifier: Apache-2.0
"""Codec loading, tensor ownership, audio boundaries and eager session accounting."""

import copy
import hashlib
import json
import os
import random
import threading
import warnings
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgpack
import numpy as np
import pytest
import soundfile
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer
from safetensors.torch import load_file, save_file

import sglang_omni.models.moss_speech.components.voice as voice_module
from sglang_omni.models.moss_speech.components.cfm_graph import (
    MossSpeechCFMGraph,
    SolverInputs,
)
from sglang_omni.models.moss_speech.components.codec_adapter import (
    CodecAdapterError,
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.flow_precision import (
    BF16Forward,
    FlowBF16Scope,
)
from sglang_omni.models.moss_speech.components.streaming_codec import (
    CodecStreamSession,
    MossSpeechStreamingCodec,
    load_flow_weights,
)
from sglang_omni.models.moss_speech.components.voice import (
    CampplusSpeakerEncoder,
    VoiceConditioning,
    compute_voice_mel,
)
from sglang_omni.models.moss_speech.config import MossSpeechVocoderArgs
from sglang_omni.models.moss_speech.hf_config import MAX_CONTEXT_TOKENS
from sglang_omni.models.moss_speech.voice_transport import (
    pack_voice_payload,
    unpack_voice_payload,
    voice_payload_digest,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.decoder import (
    CausalConditionalDecoder,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow import (
    CausalMaskedDiffWithXvec,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow_matching import (
    CausalConditionalCFM,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.utils_mask import (
    add_optional_chunk_mask,
)
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import scoped_global_rng
from sglang_omni.vendor.moss_speech_codec.hf_codec.whisper import WhisperVQEncoder
from sglang_omni.vendor.moss_speech_codec.matcha_components.bias_broadcast import (
    LiveBiasBroadcastProcessor,
    install_broadcast_bias_attention,
)
from tests.unit_test.fixtures.moss_speech_bias_broadcast import (
    qualify_bias_broadcast_streaming,
)
from tests.unit_test.fixtures.moss_speech_codec import (
    TinyHiFT,
    TinyOfflineDecoder,
    audio_payload,
    encoder_adapter,
    fullweight_streaming_inputs,
    offline_adapter,
    streaming_codec,
    tiny_whisper_config,
    vocoder,
    voice_tensors,
)


def test_bias_broadcast_cpu_fallback_preserves_public_preparer() -> None:
    with scoped_global_rng():
        attention = Attention(query_dim=16, heads=2, dim_head=8).eval()
        attention.set_processor(AttnProcessor2_0())
        hidden_states = torch.randn(2, 7, 16)
        attention_bias = torch.zeros(2, 7, 7)
        expected = attention(hidden_states, attention_mask=attention_bias)
        parameter_ids = {
            name: id(parameter) for name, parameter in attention.named_parameters()
        }
        processors = install_broadcast_bias_attention(attention)
        assert len(processors) == 1
        assert isinstance(attention.processor, LiveBiasBroadcastProcessor)
        actual = attention(hidden_states, attention_mask=attention_bias)
        assert torch.equal(actual, expected)
        assert processors[0].broadcast_calls == 0 and processors[0].fallback_calls == 1
        assert attention.prepare_attention_mask(attention_bias, 7, 2).shape == (4, 7, 7)
        assert parameter_ids == {
            name: id(parameter) for name, parameter in attention.named_parameters()
        }
        custom_processor = Mock(wraps=AttnProcessor2_0())
        attention.set_processor(custom_processor)
        assert install_broadcast_bias_attention(attention) == ()
        assert attention.processor is custom_processor


@pytest.mark.accelerator
def test_bias_broadcast_cuda_exact_live_inputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for actual bias broadcasting")
    else:
        pass
    with scoped_global_rng(), torch.inference_mode():
        generator = torch.Generator(device="cpu").manual_seed(20261102)
        for head_count, frame_count in [
            (2, 17),
            (4, 32),
            (8, 65),
            (8, 1500),
            (8, 1512),
        ]:
            attention = (
                Attention(query_dim=256, heads=head_count, dim_head=64).cuda().eval()
            )
            attention.set_processor(AttnProcessor2_0())
            original_processor = attention.processor
            hidden_states = torch.randn(2, frame_count, 256, generator=generator).cuda()
            attention_bias = torch.where(
                torch.rand(2, frame_count, frame_count, generator=generator) > 0.2,
                torch.tensor(-0.0),
                torch.tensor(-1.0e10),
            ).cuda()
            attention_bias[0, 0] = -1.0e10
            processors = install_broadcast_bias_attention(attention)
            parameter_ids = {
                name: id(parameter) for name, parameter in attention.named_parameters()
            }
            rng_before = torch.cuda.get_rng_state().clone()
            for _ in range(2):
                expected = original_processor(
                    attention, hidden_states, attention_mask=attention_bias
                )
                actual = attention(hidden_states, attention_mask=attention_bias)
                assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                held_output = actual.clone()
                attention_bias[:, :, frame_count // 2] = -1.0e10
                assert torch.equal(actual, held_output)
                attention.to_q.weight.mul_(0.75)
            assert processors[0].broadcast_calls == 2
            assert processors[0].last_head_stride == 0
            assert torch.equal(rng_before, torch.cuda.get_rng_state())
            assert parameter_ids == {
                name: id(parameter) for name, parameter in attention.named_parameters()
            }
            assert (
                attention.prepare_attention_mask(attention_bias, frame_count, 2).shape[
                    0
                ]
                == 2 * head_count
            )
            attention.train()
            expected = original_processor(
                attention, hidden_states, attention_mask=attention_bias
            )
            actual = attention(hidden_states, attention_mask=attention_bias)
            assert torch.equal(actual, expected) and processors[0].fallback_calls == 1


@pytest.mark.accelerator
def test_bias_broadcast_trained_streaming_ownership(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA and explicit real codec assets are required")
    else:
        pass
    codec_path = Path(os.environ["MOSS_CODEC_PATH"])
    code_profile_path = Path(os.environ["MOSS_CODE_PROFILE"])
    golden_tensors_path = Path(os.environ["MOSS_CODEC_GOLDEN_TENSORS"])
    profile = json.loads(code_profile_path.read_text())
    assert (
        hashlib.sha256(Path(os.environ["MOSS_AUDIO_VOICE"]).read_bytes()).hexdigest()
        == profile["voice_input_sha256"]
    )
    report_path = Path(
        os.environ.get(
            "MOSS_BIAS_BROADCAST_REPORT",
            str(tmp_path / "bias-broadcast-streaming.json"),
        )
    )
    report = qualify_bias_broadcast_streaming(
        codec_path, code_profile_path, golden_tensors_path, report_path.parent
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed" and report["broadcast_calls"] > 0


@pytest.mark.parametrize("streaming", [False, True])
def test_real_flow_estimator_constructs_and_runs(streaming: bool) -> None:
    with scoped_global_rng(), torch.inference_mode():
        estimator = CausalConditionalDecoder(
            in_channels=32,
            out_channels=8,
            channels=(16,),
            dropout=0.0,
            attention_head_dim=8,
            n_blocks=1,
            num_mid_blocks=1,
            num_heads=2,
            act_fn="gelu",
            static_chunk_size=4,
        ).eval()
        waveform_features = torch.randn(1, 8, 8)
        output = estimator(
            waveform_features,
            torch.ones(1, 1, 8),
            torch.zeros_like(waveform_features),
            torch.tensor([0.5]),
            spks=torch.zeros(1, 8),
            cond=torch.zeros_like(waveform_features),
            streaming=streaming,
        )
    assert output.shape == waveform_features.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "dtype,load_encoder,load_decoder,error",
    [
        (torch.bfloat16, True, False, NotImplementedError),
        (torch.float32, False, False, ValueError),
        (torch.float32, True, False, FileNotFoundError),
    ],
)
def test_loading_rejects_unsupported_components_before_model_construction(
    tmp_path: Path,
    dtype: torch.dtype,
    load_encoder: bool,
    load_decoder: bool,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        MossSpeechCodecAdapter(
            tmp_path,
            device="cpu",
            dtype=dtype,
            load_encoder=load_encoder,
            load_decoder=load_decoder,
        )


def test_real_encoder_accepts_known_ema_metadata_and_preserves_batched_audio_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with scoped_global_rng(), encoder_adapter(tmp_path, monkeypatch) as adapter:
        waveform = torch.sin(torch.arange(3200, dtype=torch.float32) / 20)
        short_waveform = waveform[:1920]
        soundfile.write(
            tmp_path / "input.wav", waveform.numpy(), 16000, subtype="FLOAT"
        )
        encoded = adapter.encode(
            [(waveform, 16000), (short_waveform, 16000)], batch_size=2
        )
        assert [len(codes) for codes in encoded] == [5, 3]
        assert encoded[0] == adapter.encode(waveform)[0]
        assert encoded[1] == adapter.encode([(short_waveform, 16000)])[0]
        assert encoded[0] == adapter.encode([tmp_path / "input.wav"])[0]
        assert all(0 <= code < 16 for codes in encoded for code in codes)
        assert adapter.encoder_loaded and not adapter.decoder_loaded
        checkpoint = load_file(tmp_path / "model.safetensors")
        checkpoint["encoder.unexpected_inference_weight"] = torch.ones(1)
        save_file(checkpoint, tmp_path / "model.safetensors")
        with pytest.raises(RuntimeError, match="unexpected"):
            adapter.load_encoder()


@pytest.mark.parametrize("sample_rate", [16000, 24000])
def test_voice_enrollment_keeps_codec_input_rate_and_aligns_mel_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    with scoped_global_rng(), encoder_adapter(tmp_path, monkeypatch) as adapter:
        waveform = torch.sin(
            torch.arange(sample_rate // 2, dtype=torch.float32) / 20
        ).unsqueeze(0)
        original_codes = adapter.encode([(waveform, sample_rate)])[0]
        conditioning = adapter.encode_voice_ref((waveform, sample_rate))
        assert (
            conditioning.prompt_token.tolist()[0]
            == original_codes[: conditioning.prompt_token.shape[1]]
        )
        assert conditioning.prompt_feat.shape == (
            1,
            conditioning.prompt_token.shape[1] * 4,
            80,
        )
        assert conditioning.embedding.shape == (1, 192)
        assert torch.isfinite(conditioning.prompt_feat).all()
        assert conditioning.meta["sr"] == 24000
        assert adapter.speaker_encoder is not None
        assert adapter.speaker_encoder.waveform_16khz.shape[-1] == 8000


def test_voice_mel_preserves_checkpoint_24khz_frame_rate() -> None:
    waveform = torch.sin(torch.arange(24000, dtype=torch.float32) / 20).unsqueeze(0)
    mel_features = compute_voice_mel(waveform)
    assert mel_features.shape == (1, 50, 80)
    assert torch.isfinite(mel_features).all()


def test_offline_decode_releases_actual_owner_on_failure_and_close_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = offline_adapter(tmp_path, monkeypatch)
    decoder = adapter.decoder
    assert isinstance(decoder, TinyOfflineDecoder)
    voice = VoiceConditioning(**voice_tensors(), meta={"sr": 24000})
    decoder.should_fail = True
    with pytest.raises(RuntimeError, match="Injected"):
        adapter.decode([3, 4], voice, request_id="failed")
    assert not adapter.sessions
    decoder.should_fail = False
    sample_rate, waveform = adapter.decode([3, 4, 5], voice, request_id="recovered")
    assert sample_rate == 24000 and waveform.shape == (5760,)
    assert decoder.received_codes == [[3, 4], [3, 4, 5]]
    assert not adapter.encoder_loaded and adapter.decoder_loaded
    adapter.cleanup("missing")
    adapter.close()
    adapter.close()
    assert not adapter.sessions and adapter.decoder is None
    with pytest.raises(CodecAdapterError, match="closed"):
        adapter.decode([3], voice, request_id="closed")


@pytest.mark.parametrize("codes", [[], [-1], [16384], [True], [0.5]])
def test_offline_decoder_rejects_invalid_codes_without_entering_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codes: list[int | float]
) -> None:
    with offline_adapter(tmp_path, monkeypatch) as adapter:
        with pytest.raises(ValueError, match="codes"):
            adapter.decode(
                codes,
                VoiceConditioning(**voice_tensors(), meta={}),
                request_id="invalid",
            )
        assert not adapter.sessions
        assert isinstance(adapter.decoder, TinyOfflineDecoder)
        assert not adapter.decoder.received_codes


def test_campplus_uses_cpu_session_options_and_centered_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name="features")]
    session.run.return_value = [np.arange(192, dtype=np.float32)]
    constructor = Mock(return_value=session)
    monkeypatch.setattr(voice_module.onnxruntime, "InferenceSession", constructor)
    speaker = CampplusSpeakerEncoder("campplus.onnx")
    embedding = speaker.extract(
        torch.sin(torch.arange(3200, dtype=torch.float32) / 20).unsqueeze(0)
    )
    assert constructor.call_args.kwargs["sess_options"].intra_op_num_threads == 1
    assert constructor.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    np.testing.assert_array_equal(
        embedding.numpy(), np.arange(192, dtype=np.float32)[None]
    )
    features = session.run.call_args.args[1]["features"]
    assert features.shape[0] == 1 and features.shape[-1] == 80
    np.testing.assert_allclose(features.mean(axis=1), 0, atol=1e-5)


@pytest.mark.parametrize("return_dict", [None, False, True])
def test_whisper_return_format_uses_supported_transformers_configuration(
    return_dict: bool | None,
) -> None:
    """Transformers removed the deprecated use_return_dict accessor."""
    with scoped_global_rng():
        encoder = WhisperVQEncoder(tiny_whisper_config()).eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = encoder(torch.zeros(1, 80, 16), return_dict=return_dict)
    if return_dict is False:
        assert isinstance(output, tuple) and output[0].shape == (1, 4, 8)
    else:
        assert output.quantized_token_ids.shape == (1, 4)
    assert not any("use_return_dict" in str(warning.message) for warning in caught)


def test_voice_wire_preserves_fp32_bits_and_detaches_tensor_ownership() -> None:
    voice = voice_tensors()
    voice["embedding"].view(torch.int32)[0, :4] = torch.tensor(
        [-2147483648, 1, 2139095040, 2143289345], dtype=torch.int32
    )
    packed = pack_voice_payload(voice)
    digest = voice_payload_digest(packed)
    transported = msgpack.unpackb(msgpack.packb(packed, use_bin_type=True), raw=False)
    restored = unpack_voice_payload(transported)
    for name, original in voice.items():
        assert torch.equal(original.view(torch.int32), restored[name].view(torch.int32))
    restored["embedding"][0, 0] = 99
    voice["prompt_feat"][0, 0, 0] = 99
    assert voice_payload_digest(packed) == digest
    assert (
        unpack_voice_payload(packed)["embedding"].view(torch.int32)[0, 0] == -2147483648
    )


@pytest.mark.parametrize(
    "mutation", ["truncated", "dtype", "shape", "extra", "missing"]
)
def test_malformed_voice_wire_is_rejected(mutation: str) -> None:
    packed = copy.deepcopy(pack_voice_payload(voice_tensors()))
    if mutation == "truncated":
        packed["embedding_bytes"] = packed["embedding_bytes"][:-1]
    elif mutation == "dtype":
        packed["embedding_dtype"] = "float64"
    elif mutation == "shape":
        packed["prompt_feat_shape"] = [1, 7, 80]
    elif mutation == "extra":
        packed["voice"] = "legacy"
    else:
        packed.pop("prompt_token_shape")
    with pytest.raises(ValueError):
        unpack_voice_payload(packed)


@pytest.mark.parametrize("code_count", [1, 7, 8, 11, 12, 24, 511, 512])
def test_eager_streaming_sample_ledger_keeps_short_boundaries_and_final_code(
    code_count: int,
) -> None:
    codec = streaming_codec()
    codec.begin("audio", voice_tensors(), seed=19)
    session = codec.sessions["audio"]
    first_part = list(range(min(code_count, 11)))
    waveforms = codec.push("audio", first_part)
    waveforms.extend(
        codec.push("audio", list(range(len(first_part), code_count)), final=True)
    )
    assert sum(waveform.numel() for waveform in waveforms) == code_count * 1920
    assert session.codes[-1] == code_count - 1
    assert session.ledger[-1]["final"] is True
    assert sum(entry["final"] for entry in session.ledger) == 1
    assert session.cursor == code_count and not codec.sessions


@pytest.mark.parametrize(
    "chunk_size,prompt_length", [(5, 2), (5, 5), (25, 2), (25, 25)]
)
def test_unready_stream_accumulates_codes_without_entering_rng_or_cudnn_scopes(
    chunk_size: int, prompt_length: int
) -> None:
    codec = streaming_codec(chunk_size=chunk_size)
    voice = voice_tensors()
    voice["prompt_token"] = torch.zeros((1, prompt_length), dtype=torch.int32)
    voice["prompt_feat"] = torch.zeros((1, prompt_length * 4, 80))
    codec.begin("audio", voice, seed=19)
    session = codec.sessions["audio"]
    hop = chunk_size + (-prompt_length) % chunk_size
    threshold = hop + codec.flow.pre_lookahead_len
    caller_rng, request_rng = torch.get_rng_state().clone(), session.cpu_rng.clone()
    with (
        patch.object(torch.random, "fork_rng", side_effect=AssertionError("RNG scope")),
        patch.object(
            torch.backends.cudnn, "flags", side_effect=AssertionError("cuDNN scope")
        ),
    ):
        for code in range(threshold - 1):
            assert codec.push("audio", [code]) == []
        assert codec.push("audio", []) == []
    assert session.codes == list(range(threshold - 1))
    assert session.cursor == 0 and session.emitted_samples == 0
    assert session.ledger == [] and session.cache is None
    assert not session.final_requested and session.device_rng is None
    assert torch.equal(session.cpu_rng, request_rng)
    assert torch.equal(torch.get_rng_state(), caller_rng)
    waveforms = codec.push("audio", [threshold - 1])
    samples = hop * codec.samples_per_code - codec.source_cache_len
    assert len(waveforms) == 1 and waveforms[0].numel() == samples
    assert session.cursor == hop
    assert session.ledger == [
        {"available": threshold, "cursor_before": 0, "samples": samples, "final": False}
    ]
    codec.close()


@pytest.mark.parametrize("latched", [False, True])
def test_unready_short_stream_still_enters_rng_scope_and_flushes_when_final(
    latched: bool,
) -> None:
    codec = streaming_codec()
    codec.begin("audio", voice_tensors(), seed=19)
    session = codec.sessions["audio"]
    assert codec.push("audio", [7]) == []
    session.final_requested = latched
    caller_rng = torch.get_rng_state().clone()
    with (
        patch.object(
            torch.random, "fork_rng", wraps=torch.random.fork_rng
        ) as rng_scope,
        patch.object(
            torch.backends.cudnn, "flags", wraps=torch.backends.cudnn.flags
        ) as cudnn_scope,
    ):
        waveforms = codec.push("audio", [], final=not latched)
    assert rng_scope.call_count == cudnn_scope.call_count == 1
    assert sum(waveform.numel() for waveform in waveforms) == codec.samples_per_code
    assert session.cursor == 1 and session.final_requested
    assert session.ledger == [
        {"available": 1, "cursor_before": 0, "samples": 1920, "final": True}
    ]
    assert torch.equal(torch.get_rng_state(), caller_rng)
    assert not codec.sessions


@pytest.mark.parametrize(
    "codes,error",
    [([True], "Invalid"), ([0] * MAX_CONTEXT_TOKENS, str(MAX_CONTEXT_TOKENS))],
)
def test_unready_stream_fast_return_keeps_validation_and_session_cleanup(
    codes: list[int], error: str
) -> None:
    codec = streaming_codec()
    codec.begin("failed", voice_tensors(), seed=19)
    codec.begin("kept", voice_tensors(), seed=23)
    caller_rng = torch.get_rng_state().clone()
    kept_rng = codec.sessions["kept"].cpu_rng.clone()
    assert codec.push("failed", [7]) == []
    with (
        patch.object(torch.random, "fork_rng", side_effect=AssertionError("RNG scope")),
        patch.object(
            torch.backends.cudnn, "flags", side_effect=AssertionError("cuDNN scope")
        ),
        pytest.raises(ValueError, match=error),
    ):
        codec.push("failed", codes)
    assert set(codec.sessions) == {"kept"}
    assert torch.equal(codec.sessions["kept"].cpu_rng, kept_rng)
    assert torch.equal(torch.get_rng_state(), caller_rng)
    codec.close()


def test_unready_stream_accepts_codes_beyond_the_old_length_limit() -> None:
    codec = streaming_codec(chunk_size=25)
    codec.begin("audio", voice_tensors(), seed=19)
    codec.push("audio", [7] * 512)
    session = codec.sessions["audio"]
    cursor = session.cursor
    caller_rng, request_rng = torch.get_rng_state().clone(), session.cpu_rng.clone()
    with patch.object(
        torch.random, "fork_rng", side_effect=AssertionError("RNG scope")
    ):
        assert codec.push("audio", [8]) == []
    assert len(session.codes) == 513 and session.codes[-1] == 8
    assert session.cursor == cursor
    assert torch.equal(torch.get_rng_state(), caller_rng)
    assert torch.equal(session.cpu_rng, request_rng)
    codec.close()


def test_stream_rng_isolation_survives_interleaving_and_abort() -> None:
    isolated = streaming_codec()
    isolated.begin("alone", voice_tensors(), seed=123)
    expected = isolated.push("alone", list(range(20)), final=True)
    interleaved = streaming_codec()
    interleaved.begin("kept", voice_tensors(), seed=123)
    interleaved.begin("aborted", voice_tensors(), seed=456)
    caller_rng = torch.get_rng_state().clone()
    actual = interleaved.push("kept", list(range(11)))
    interleaved.push("aborted", list(range(12)))
    interleaved.cleanup("aborted")
    actual.extend(interleaved.push("kept", list(range(11, 20)), final=True))
    assert torch.equal(torch.cat(actual), torch.cat(expected))
    assert torch.equal(torch.get_rng_state(), caller_rng)
    assert not interleaved.sessions


def test_stream_empty_final_cleans_up_and_decoder_failure_recovers() -> None:
    codec = streaming_codec()
    codec.begin("duplicate", voice_tensors(), seed=1)
    with pytest.raises(ValueError, match="already active"):
        codec.begin("duplicate", voice_tensors(), seed=2)
    caller_rng = torch.get_rng_state().clone()
    session = codec.sessions["duplicate"]
    assert codec.push("duplicate", [], final=True) == []
    assert session.emitted_samples == 0 and session.ledger == []
    assert torch.equal(torch.get_rng_state(), caller_rng)
    assert not codec.sessions
    codec.begin("failure", voice_tensors(), seed=1)
    assert isinstance(codec.hift, TinyHiFT)
    codec.hift.should_fail = True
    with pytest.raises(RuntimeError, match="Injected"):
        codec.push("failure", [1, 2], final=True)
    assert not codec.sessions
    codec.hift.should_fail = False
    codec.begin("recovered", voice_tensors(), seed=1)
    assert (
        sum(
            waveform.numel() for waveform in codec.push("recovered", [1, 2], final=True)
        )
        == 3840
    )


@pytest.mark.parametrize("invalid_code", [-1, 16384, True])
def test_invalid_stream_code_releases_only_its_actual_session(
    invalid_code: int,
) -> None:
    codec = streaming_codec()
    codec.begin("failed", voice_tensors(), seed=1)
    codec.begin("kept", voice_tensors(), seed=2)
    with pytest.raises(ValueError, match="Invalid"):
        codec.push("failed", [invalid_code])
    assert set(codec.sessions) == {"kept"}
    codec.close()
    assert not codec.sessions


def test_flow_loader_ignores_only_training_counters() -> None:
    module = torch.nn.Linear(2, 1)
    weights = {
        "weight": torch.full((1, 2), 3.0),
        "bias": torch.ones(1),
        "epoch": 5,
        "step": 10,
    }
    load_flow_weights(module, weights)
    torch.testing.assert_close(module(torch.ones(1, 2)), torch.tensor([[7.0]]))
    with pytest.raises(RuntimeError, match="Unexpected"):
        load_flow_weights(module, {**weights, "wrong": torch.ones(1)})


def test_scoped_rng_restores_python_numpy_and_torch_after_exception() -> None:
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    with pytest.raises(RuntimeError, match="scope"):
        with scoped_global_rng():
            random.seed(2)
            np.random.seed(3)
            torch.manual_seed(4)
            random.random(), np.random.rand(), torch.rand(3)
            raise RuntimeError("scope")
    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0] and restored_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(restored_numpy[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("seed", [-1, 2**40])
def test_nonstream_audio_accepts_full_torch_seed_range_and_restores_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """Valid Torch seeds used to fail at NumPy's narrower seed boundary."""
    scheduler = vocoder(tmp_path, monkeypatch)
    try:
        caller_rng = torch.get_rng_state().clone()
        result = scheduler.compute(
            audio_payload("offline", [3, 4], seed=seed, stream=False)
        )
        assert (
            result.data["audio_data"]
            == torch.rand(3840, generator=torch.Generator().manual_seed(seed)).tolist()
        )
        assert isinstance(scheduler.offline_codec.decoder, TinyOfflineDecoder)
        assert scheduler.offline_codec.decoder.received_codes == [[3, 4]]
        assert not scheduler.offline_codec.sessions
        assert torch.equal(torch.get_rng_state(), caller_rng)
    finally:
        scheduler.stop()


def tiny_cfm_solver(device: torch.device) -> CausalConditionalCFM:
    with scoped_global_rng(), torch.device(device):
        return (
            CausalConditionalCFM(
                in_channels=240,
                n_spks=1,
                spk_emb_dim=80,
                cfm_params=SimpleNamespace(
                    sigma_min=1e-6,
                    solver="euler",
                    t_scheduler="cosine",
                    training_cfg_rate=0.2,
                    inference_cfg_rate=0.7,
                ),
                estimator=CausalConditionalDecoder(
                    in_channels=320,
                    out_channels=80,
                    channels=(16,),
                    dropout=0.0,
                    attention_head_dim=8,
                    n_blocks=1,
                    num_mid_blocks=1,
                    num_heads=2,
                    act_fn="gelu",
                    static_chunk_size=100,
                    num_decoding_left_chunks=-1,
                ),
            )
            .to(dtype=torch.float32)
            .eval()
        )


def cfm_solver_inputs(
    device: torch.device, *, mel_length: int = 100, shift: float = 0.0
) -> SolverInputs:
    x = (
        torch.linspace(-0.2, 0.2, 80 * mel_length, device=device).reshape(
            1, 80, mel_length
        )
        + shift
    )
    time_steps = torch.linspace(0, 1 + shift / 10, 11, device=device)
    time_steps = 1 - torch.cos(time_steps * 0.5 * torch.pi)
    mel_conditioning = torch.zeros(1, mel_length, 80, device=device).transpose(1, 2)
    mel_conditioning.copy_(x * 0.3 + shift / 3)
    return (
        x,
        time_steps,
        x * 0.2 + shift / 7,
        torch.full((1, 1, mel_length), 1 - shift / 20, device=device),
        torch.linspace(-0.1, 0.1, 80, device=device).reshape(1, 80) + shift / 5,
        mel_conditioning,
    )


class CFMTestStream:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def synchronize(self) -> None:
        self.events.append(self.name)


class CFMTestReplay:
    def __init__(self, runner: MossSpeechCFMGraph, events: list[str]) -> None:
        self.runner = runner
        self.events = events
        self.calls = 0
        self.should_fail = False

    def replay(self) -> None:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("Injected graph replay failure")
        else:
            assert self.runner.inputs is not None and self.runner.output is not None
            self.runner.output.copy_(
                self.runner.original_solver(*self.runner.inputs, streaming=True)
            )

    def reset(self) -> None:
        self.events.append("reset")


def prepared_cpu_cfm_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MossSpeechCFMGraph, CFMTestReplay, list[str]]:
    runner = MossSpeechCFMGraph.__new__(MossSpeechCFMGraph)
    runner.decoder = tiny_cfm_solver(torch.device("cpu"))
    runner.device = torch.device("cpu")
    runner.mel_length = 100
    runner.lock = threading.RLock()
    runner.original_solver = runner.decoder.solve_euler
    runner.installed_solver = runner.solve_euler
    runner.decoder.solve_euler = runner.installed_solver
    runner.inputs = cfm_solver_inputs(runner.device)
    runner.output = torch.empty_like(runner.inputs[0])
    runner.closed = False
    runner.replays = runner.fallbacks = 0
    runner.prepared_model_state = runner.model_state()
    runner.prepared_numerical_context = runner.numerical_context()
    events: list[str] = []
    runner.owner_stream = CFMTestStream(events, "owner")
    runner.capture_stream = CFMTestStream(events, "capture")
    replay = CFMTestReplay(runner, events)
    runner.graph = replay

    def allow_test_replay(inputs: SolverInputs, *, streaming: bool) -> bool:
        return streaming and not runner.closed

    def cpu_device_scope(device: torch.device) -> nullcontext[None]:
        assert device == torch.device("cpu")
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "device", cpu_device_scope)
    monkeypatch.setattr(runner, "can_replay", allow_test_replay)
    return runner, replay, events


def test_cfm_replay_copies_six_inputs_and_keeps_session_mel_tail_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later replay must not overwrite an earlier session's retained mel view."""
    runner, replay, events = prepared_cpu_cfm_graph(monkeypatch)
    with torch.inference_mode():
        first_inputs = cfm_solver_inputs(runner.device)
        assert not MossSpeechCFMGraph.can_replay(runner, first_inputs, streaming=True)
        expected = runner.original_solver(*first_inputs, streaming=True)
        first = runner.decoder.solve_euler(*first_inputs, streaming=True)
        retained_tail = first[:, :, -8:]
        second_inputs = cfm_solver_inputs(runner.device, shift=0.5)
        for first_input, second_input in zip(first_inputs, second_inputs, strict=True):
            assert not torch.equal(first_input, second_input)
        second_expected = runner.original_solver(*second_inputs, streaming=True)
        second = runner.decoder.solve_euler(*second_inputs, streaming=True)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(retained_tail, expected[:, :, -8:], rtol=0, atol=0)
        torch.testing.assert_close(second, second_expected, rtol=0, atol=0)
        assert runner.replays == replay.calls == 2
        final_expected = runner.original_solver(*second_inputs, streaming=False)
        final = runner.decoder.solve_euler(*second_inputs, streaming=False)
        torch.testing.assert_close(final, final_expected, rtol=0, atol=0)
        assert runner.replays == 2 and runner.fallbacks == 1
    runner.close()
    runner.close()
    assert "solve_euler" not in vars(runner.decoder)
    assert events == ["owner", "capture", "reset"]


def test_cfm_failed_replay_does_not_duplicate_eager_work_and_preserves_new_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, replay, events = prepared_cpu_cfm_graph(monkeypatch)
    eager = Mock(wraps=runner.original_solver)
    runner.original_solver = eager
    replay.should_fail = True
    with torch.inference_mode(), pytest.raises(RuntimeError, match="Injected graph"):
        runner.decoder.solve_euler(*cfm_solver_inputs(runner.device), streaming=True)
    eager.assert_not_called()
    assert replay.calls == 1 and runner.closed and runner.replays == 0
    newer_hook = Mock(wraps=eager)
    runner.decoder.solve_euler = newer_hook
    runner.close()
    assert runner.decoder.solve_euler is newer_hook
    assert events == ["owner", "capture", "reset"]


@pytest.mark.parametrize("chunk_size", [5, 25])
def test_cfm_graph_cpu_startup_rejects_explicit_opt_in_without_installing_hook(
    chunk_size: int,
) -> None:
    decoder = tiny_cfm_solver(torch.device("cpu"))
    original = decoder.solve_euler
    with pytest.raises(ValueError, match="CUDA FP32"):
        MossSpeechCFMGraph(decoder, mel_length=100)
    assert "solve_euler" not in vars(decoder)
    assert decoder.solve_euler == original
    with pytest.raises(ValueError, match="CFM|graph|CUDA|chunk"):
        MossSpeechStreamingCodec(
            None,
            chunk_size=chunk_size,
            device="cpu",
            flow=SimpleNamespace(decoder=decoder),
            hift=TinyHiFT(),
            cfm_graph_mel_length=100,
        )
    assert decoder.solve_euler == original


@pytest.mark.parametrize("static_chunk_size", [0, 2])
@pytest.mark.parametrize(
    "valid_frames,full_expected,chunk_expected",
    [
        pytest.param(
            [True, True, True, True],
            [True, True, True, True],
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
            ],
            id="all-valid",
        ),
        pytest.param(
            [False, False, False, False],
            [True, True, True, True],
            [[True, True, True, True]] * 4,
            id="all-empty",
        ),
        pytest.param(
            [False, False, True, False],
            [False, False, True, False],
            [
                [True, True, True, True],
                [True, True, True, True],
                [False, False, True, False],
                [False, False, True, False],
            ],
            id="empty-first-chunk",
        ),
    ],
)
def test_chunk_mask_repairs_only_empty_rows_and_preserves_full_context_alias(
    monkeypatch: pytest.MonkeyPatch,
    static_chunk_size: int,
    valid_frames: list[bool],
    full_expected: list[bool],
    chunk_expected: list[list[bool]],
) -> None:
    cuda_init = Mock(side_effect=AssertionError("CPU mask initialized CUDA"))
    capture_query = Mock(side_effect=AssertionError("CPU mask queried CUDA capture"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", cuda_init)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capture_query)
    masks = torch.zeros(1, 1, 8, device="cpu", dtype=torch.bool)[:, :, ::2]
    masks.copy_(torch.tensor([[valid_frames]], device="cpu"))
    actual = add_optional_chunk_mask(
        torch.zeros(1, 4, 1, device="cpu"),
        masks,
        use_dynamic_chunk=False,
        use_dynamic_left_chunk=False,
        decoding_chunk_size=0,
        static_chunk_size=static_chunk_size,
        num_decoding_left_chunks=-1,
    )
    expected = chunk_expected if static_chunk_size else [full_expected]
    torch.testing.assert_close(
        actual, torch.tensor([expected], device="cpu"), rtol=0, atol=0
    )
    if static_chunk_size:
        assert actual is not masks
        assert masks.tolist() == [[valid_frames]]
    else:
        assert actual is masks
        assert masks.tolist() == [[full_expected]]
    assert not masks.is_contiguous()
    cuda_init.assert_not_called()
    capture_query.assert_not_called()


@pytest.mark.accelerator
def test_chunk_mask_cuda_graph_replays_empty_row_repair_from_current_input() -> None:
    """Empty attention rows must be repaired without a host read during capture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    device = torch.device("cuda", 0)
    for static_chunk_size in (0, 2):
        features = torch.zeros(1, 4, 1, device=device)
        masks = torch.ones(1, 1, 8, device=device, dtype=torch.bool)[:, :, ::2]
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.inference_mode(), torch.cuda.stream(capture_stream):
                for _ in range(3):
                    add_optional_chunk_mask(
                        features, masks, False, False, 0, static_chunk_size, -1
                    )
            torch.cuda.current_stream(device).wait_stream(capture_stream)
            with torch.inference_mode(), torch.cuda.graph(graph, stream=capture_stream):
                actual = add_optional_chunk_mask(
                    features, masks, False, False, 0, static_chunk_size, -1
                )
            torch.cuda.current_stream(device).wait_stream(capture_stream)
            if static_chunk_size:
                assert actual is not masks
            else:
                assert actual is masks
            for valid_frames in (
                [False, False, False, False],
                [False, False, True, False],
                [True, True, True, True],
            ):
                masks.copy_(torch.tensor([[valid_frames]], device=device))
                graph.replay()
                expected = []
                for query_index in range(4 if static_chunk_size else 1):
                    row = [
                        is_valid
                        and (
                            not static_chunk_size
                            or frame_index // static_chunk_size
                            <= query_index // static_chunk_size
                        )
                        for frame_index, is_valid in enumerate(valid_frames)
                    ]
                    expected.append(row if any(row) else [True] * 4)
                torch.testing.assert_close(
                    actual.cpu(), torch.tensor([expected], device="cpu"), rtol=0, atol=0
                )
                expected_input = valid_frames if static_chunk_size else expected[0]
                assert masks.cpu().tolist() == [[expected_input]]
                assert not masks.is_contiguous()
        finally:
            capture_stream.synchronize()
            graph.reset()


@pytest.mark.accelerator
def test_cfm_cuda_graph_matches_native_solver_and_preserves_owned_outputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    device = torch.device("cuda", 0)
    decoder = tiny_cfm_solver(device)
    assert all(
        parameter.device == device and parameter.dtype == torch.float32
        for parameter in decoder.parameters()
    )
    original = decoder.solve_euler
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    caller_flags = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    runner = MossSpeechCFMGraph(decoder, mel_length=100)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    ) == caller_flags
    try:
        with (
            torch.inference_mode(),
            torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ),
        ):
            first_inputs = cfm_solver_inputs(device)
            first_expected = original(*first_inputs, streaming=True)
            assert runner.can_replay(first_inputs, streaming=True)
            first = decoder.solve_euler(*first_inputs, streaming=True)
            first_tail = first[:, :, -8:]
            second_inputs = cfm_solver_inputs(device, shift=0.5)
            second_expected = original(*second_inputs, streaming=True)
            second = decoder.solve_euler(*second_inputs, streaming=True)
            torch.testing.assert_close(first, first_expected, rtol=0, atol=0)
            torch.testing.assert_close(
                first_tail, first_expected[:, :, -8:], rtol=0, atol=0
            )
            torch.testing.assert_close(second, second_expected, rtol=0, atol=0)
            assert runner.replays == 2
            for inputs, streaming in (
                (cfm_solver_inputs(device, mel_length=200), True),
                (second_inputs, False),
            ):
                expected = original(*inputs, streaming=streaming)
                actual = decoder.solve_euler(*inputs, streaming=streaming)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert runner.replays == 2 and runner.fallbacks == 2
            decoder.train()
            assert not runner.can_replay(second_inputs, streaming=True)
            decoder.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                assert not runner.can_replay(second_inputs, streaming=True)
            with torch.backends.cudnn.flags(
                enabled=not torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ):
                assert not runner.can_replay(second_inputs, streaming=True)
            assert runner.can_replay(second_inputs, streaming=True)
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=not torch.backends.cudnn.allow_tf32,
            ):
                assert not runner.can_replay(second_inputs, streaming=True)
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=True,
                deterministic=True,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ):
                assert not runner.can_replay(second_inputs, streaming=True)
            with torch.cuda.stream(torch.cuda.Stream(device=device)):
                assert not runner.can_replay(second_inputs, streaming=True)
            decoder.inference_cfg_rate = 0.5
            expected = original(*second_inputs, streaming=True)
            actual = decoder.solve_euler(*second_inputs, streaming=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            decoder.inference_cfg_rate = 0.7
            next(decoder.parameters()).add_(0.01)
            expected = original(*second_inputs, streaming=True)
            actual = decoder.solve_euler(*second_inputs, streaming=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert runner.replays == 2 and runner.fallbacks == 4
            assert torch.equal(torch.get_rng_state(), cpu_rng)
            assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    finally:
        runner.close()
    assert "solve_euler" not in vars(decoder)


@pytest.mark.accelerator
def test_cfm_graph_fullweights_streaming_pcm_and_cleanup() -> None:
    """Real graph replay preserves full native streaming PCM and session isolation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    codec_path, voice, codes = fullweight_streaming_inputs()
    eager = MossSpeechStreamingCodec(codec_path, chunk_size=25, device="cuda:0")
    eager.begin("eager", voice, seed=42)
    expected = torch.cat(eager.push("eager", codes, final=True)).cpu()
    assert expected.numel() == len(codes) * eager.samples_per_code
    first_hop = 25 + (-voice["prompt_token"].shape[1]) % 25
    mel_length = (voice["prompt_token"].shape[1] + first_hop) * 4
    codec = MossSpeechStreamingCodec(
        None,
        chunk_size=25,
        device="cuda:0",
        flow=eager.flow,
        hift=eager.hift,
        cfm_graph_mel_length=mel_length,
    )
    graph = codec.cfm_graph
    assert graph is not None and graph.graph is not None and graph.output is not None
    retained: dict[str, list[torch.Tensor]] = {}
    sessions: dict[str, CodecStreamSession] = {}
    try:
        for request_id in ("graph-a", "graph-b", "cancelled"):
            codec.begin(request_id, voice, seed=42)
            sessions[request_id] = codec.sessions[request_id]
            retained[request_id] = codec.push(
                request_id, codes[: first_hop + codec.flow.pre_lookahead_len]
            )
            assert len(retained[request_id]) == 1
        assert graph.replays == 3, "Ineligible eager fallback cannot qualify the graph"
        codec.cleanup("cancelled")
        codec.begin("invalid", voice, seed=42)
        with pytest.raises(ValueError, match="Invalid streaming audio code"):
            codec.push("invalid", [-1])
        assert set(codec.sessions) == {"graph-a", "graph-b"}
        for request_id in ("graph-a", "graph-b"):
            retained[request_id].extend(
                codec.push(
                    request_id,
                    codes[first_hop + codec.flow.pre_lookahead_len :],
                    final=True,
                )
            )
            actual = torch.cat(retained[request_id]).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert sessions[request_id].ledger[-1]["final"]
            assert (
                sessions[request_id].emitted_samples
                == len(codes) * codec.samples_per_code
            )
        assert not codec.sessions
        assert graph.replays == 3 and graph.fallbacks > 0
    finally:
        codec.close()
        eager.close()
    assert codec.cfm_graph is None and "solve_euler" not in vars(codec.flow.decoder)


@pytest.mark.parametrize(
    "stage",
    [
        "ffn_first",
        "ffn",
        "attention_first",
        "attention",
        "convolutions_first",
        "convolutions",
    ],
)
@torch.no_grad()
def test_flow_bf16_scope_preserves_fp32_solver_and_restores_on_error(
    stage: str,
) -> None:
    decoder = tiny_cfm_solver(torch.device("cpu"))
    inputs = cfm_solver_inputs(torch.device("cpu"), mel_length=8)
    baseline = decoder.solve_euler(*inputs, streaming=True)
    original_weights = {
        name: value.clone() for name, value in decoder.state_dict().items()
    }
    caller_rng = torch.get_rng_state().clone()
    scope = FlowBF16Scope(decoder.estimator, stage)
    with pytest.raises(RuntimeError, match="cancelled"):
        with scope:
            actual = decoder.solve_euler(*inputs, streaming=True)
            assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
            assert all(
                operator.forward.bf16_executions > 0 for operator in scope.operators
            )
            assert all(
                operator.forward.last_compute_dtype == torch.bfloat16
                for operator in scope.operators
            )
            assert all(
                parameter.dtype == torch.float32 for parameter in decoder.parameters()
            )
            raise RuntimeError("cancelled")
    assert torch.equal(decoder.solve_euler(*inputs, streaming=True), baseline)
    assert torch.equal(caller_rng, torch.get_rng_state())
    for name, parameter in decoder.state_dict().items():
        assert torch.equal(parameter, original_weights[name])


@pytest.mark.parametrize(
    "stage",
    [
        "ffn_first",
        "ffn",
        "attention_first",
        "attention",
        "convolutions_first",
        "convolutions",
        "m1",
        "ffn_attention",
        "ffn_attention_sdpa",
        "ffn_attention_sdpa_resident",
    ],
)
def test_bf16_flow_rejects_unqualified_fp32_graph_combination(stage: str) -> None:
    with pytest.raises(ValueError, match="excludes the FP32 CFM graph"):
        MossSpeechVocoderArgs(flow_precision=stage, cfm_graph_mel_length=1500)
    with pytest.raises(ValueError, match="without an FP32 CFM graph"):
        MossSpeechStreamingCodec(
            None,
            chunk_size=25,
            device="cuda:0",
            flow_precision=stage,
            cfm_graph_mel_length=1500,
        )


@pytest.mark.parametrize(
    "stage",
    ["ffn", "ffn_attention", "ffn_attention_sdpa", "ffn_attention_sdpa_resident"],
)
@torch.no_grad()
def test_bf16_owner_close_and_reactivate_keeps_same_precision(stage: str) -> None:
    flow = CausalMaskedDiffWithXvec.__new__(CausalMaskedDiffWithXvec)
    torch.nn.Module.__init__(flow)
    flow.decoder = tiny_cfm_solver(torch.device("cpu"))
    install_broadcast_bias_attention(flow.decoder.estimator)
    codec = MossSpeechStreamingCodec.__new__(MossSpeechStreamingCodec)
    codec.flow = flow
    codec.flow_precision = stage
    codec.precision_context = ExitStack()
    codec.precision_scopes = ()
    codec.sessions = {}
    codec.lock = threading.Lock()
    codec.cfm_graph = None
    inputs = cfm_solver_inputs(torch.device("cpu"), mel_length=8)
    original = flow.decoder.solve_euler(*inputs, streaming=True)
    codec.activate_precision()
    first = flow.decoder.solve_euler(*inputs, streaming=True)
    codec.activate_precision()
    assert torch.equal(first, flow.decoder.solve_euler(*inputs, streaming=True))
    assert all(
        operator.forward.bf16_executions > 0
        for scope in codec.precision_scopes
        for operator in scope.operators
    )
    codec.close()
    assert not codec.precision_scopes
    assert torch.equal(original, flow.decoder.solve_euler(*inputs, streaming=True))
    codec.activate_precision()
    assert torch.equal(first, flow.decoder.solve_euler(*inputs, streaming=True))
    codec.close()


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
def test_resident_bf16_linear_matches_autocast_and_releases_copies(device: str) -> None:
    """Resident inference must preserve autocast output, FP32 masters and gradients."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    with scoped_global_rng():
        layer = torch.nn.Linear(256, 512).to(device).eval()
        control = BF16Forward(layer, layer.forward)
        resident = BF16Forward(layer, layer.forward, resident_weights=True)
        weights = {name: value.clone() for name, value in layer.state_dict().items()}
        resident.prepare()
        assert resident.resident_weight is not None
        pointer = resident.resident_weight.data_ptr()
        copies = resident.parameter_copies
        with torch.no_grad():
            for frames in (17, 65, 1512):
                hidden_states = torch.randn(2, frames, 256, device=device)
                assert torch.equal(resident(hidden_states), control(hidden_states))
                assert resident.resident_weight.data_ptr() == pointer
                assert resident.parameter_copies == copies
        assert resident.bf16_executions == 3 and resident.resident_parameter_bytes > 0
        for name, value in layer.state_dict().items():
            assert value.dtype is torch.float32 and torch.equal(value, weights[name])
        resident.release()
        assert resident.resident_parameter_bytes == 0
        hidden_states = torch.randn(1, 256, device=device, requires_grad=True)
        result = resident(hidden_states)
        assert torch.equal(result, layer(hidden_states))
        result.sum().backward()
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()


def test_resident_bf16_rejects_active_lora() -> None:
    """A cached plain linear must never silently drop an active adapter."""
    layer = LoRACompatibleLinear(256, 512).eval()
    layer.set_lora_layer(LoRALinearLayer(256, 512, rank=4))
    with pytest.raises(ValueError, match="unadapted FP32 Linear"):
        BF16Forward(layer, layer.forward, resident_weights=True).prepare()


def test_resident_bf16_failed_entry_releases_partial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed cache allocation must not leave installed forwards or owned tensors."""
    decoder = tiny_cfm_solver(torch.device("cpu"))
    scope = FlowBF16Scope(decoder.estimator, "ffn", resident_weights=True)

    def fail() -> None:
        raise MemoryError("resident preparation failed")

    monkeypatch.setattr(scope.operators[1].forward, "prepare", fail)
    with pytest.raises(MemoryError, match="resident preparation failed"):
        scope.__enter__()
    assert not scope.active
    assert all("forward" not in vars(operator.module) for operator in scope.operators)
    assert all(
        operator.forward.resident_parameter_bytes == 0 for operator in scope.operators
    )
    with pytest.raises(ValueError, match="complete FFN or attention"):
        FlowBF16Scope(decoder.estimator, "convolutions", resident_weights=True)


def test_bf16_sdpa_requires_live_bias_and_full_projection_stage() -> None:
    decoder = tiny_cfm_solver(torch.device("cpu"))
    with pytest.raises(ValueError, match="complete attention"):
        FlowBF16Scope(decoder.estimator, "ffn", bf16_sdpa=True)
    with pytest.raises(ValueError, match="live bias"):
        FlowBF16Scope(decoder.estimator, "attention", bf16_sdpa=True)
    with pytest.raises(ValueError, match="live bias"):
        MossSpeechVocoderArgs(flow_precision="ffn_attention_sdpa")
    with pytest.raises(ValueError, match="live bias"):
        MossSpeechStreamingCodec(
            None, chunk_size=25, device="cuda:0", flow_precision="ffn_attention_sdpa"
        )
    processors = install_broadcast_bias_attention(decoder.estimator)
    scope = FlowBF16Scope(decoder.estimator, "attention", bf16_sdpa=True)
    with pytest.raises(RuntimeError, match="cancelled"):
        with scope:
            assert processors and all(
                processor.use_bf16_sdpa for processor in processors
            )
            raise RuntimeError("cancelled")
    assert all(not processor.use_bf16_sdpa for processor in processors)


@pytest.mark.accelerator
@pytest.mark.parametrize("frame_count", [17, 65, 1500, 1512])
@torch.no_grad()
def test_bf16_sdpa_preserves_live_mask_and_fp32_boundary(frame_count: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for BF16 SDPA")
    else:
        pass
    with scoped_global_rng():
        attention = Attention(query_dim=256, heads=8, dim_head=64).cuda().eval()
        original = attention.processor
        processor = install_broadcast_bias_attention(attention)[0]
        processor.use_bf16_sdpa = True
        hidden_states = torch.randn(2, frame_count, 256, device="cuda")
        attention_bias = torch.zeros(2, frame_count, frame_count, device="cuda")
        visible_frames = frame_count // 2
        attention_bias[:, :, visible_frames:] = -1.0e10
        original_bias = attention_bias.clone()
        rng_before = torch.cuda.get_rng_state().clone()
        output = attention(hidden_states, attention_mask=attention_bias)
        assert output.dtype == torch.float32 and torch.isfinite(output).all()
        assert processor.last_sdpa_dtype == torch.bfloat16
        assert processor.bf16_sdpa_calls == processor.broadcast_calls == 1
        assert processor.last_head_stride == 0 and torch.equal(
            attention_bias, original_bias
        )
        held_output = output.clone()
        hidden_states[:, visible_frames:] += 100
        masked = attention(hidden_states, attention_mask=attention_bias)
        assert torch.equal(masked[:, :visible_frames], output[:, :visible_frames])
        attention_bias[:, :, -1] = 0
        changed = attention(hidden_states, attention_mask=attention_bias)
        assert not torch.equal(changed[:, :visible_frames], output[:, :visible_frames])
        assert torch.equal(output, held_output)
        assert processor.bf16_sdpa_calls == processor.broadcast_calls == 3
        assert torch.equal(rng_before, torch.cuda.get_rng_state())
        processor.use_bf16_sdpa = False
        expected = original(attention, hidden_states, attention_mask=attention_bias)
        restored = attention(hidden_states, attention_mask=attention_bias)
        assert torch.equal(restored, expected)
        assert (
            processor.last_sdpa_dtype == torch.float32
            and processor.bf16_sdpa_calls == 3
        )
