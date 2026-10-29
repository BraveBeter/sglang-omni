# SPDX-License-Identifier: Apache-2.0
"""Codec loading, tensor ownership, audio boundaries and eager session accounting."""

import copy
import random
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import numpy as np
import pytest
import soundfile
import torch
from safetensors.torch import load_file, save_file

import sglang_omni.models.moss_speech.components.voice as voice_module
from sglang_omni.models.moss_speech.components.codec_adapter import (
    CodecAdapterError,
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.streaming_codec import load_flow_weights
from sglang_omni.models.moss_speech.components.voice import (
    CampplusSpeakerEncoder,
    VoiceConditioning,
    compute_voice_mel,
)
from sglang_omni.models.moss_speech.voice_transport import (
    pack_voice_payload,
    unpack_voice_payload,
    voice_payload_digest,
)
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.decoder import (
    CausalConditionalDecoder,
)
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import scoped_global_rng
from sglang_omni.vendor.moss_speech_codec.hf_codec.whisper import WhisperVQEncoder
from tests.unit_test.fixtures.moss_speech_codec import (
    TinyHiFT,
    TinyOfflineDecoder,
    audio_payload,
    encoder_adapter,
    offline_adapter,
    streaming_codec,
    tiny_whisper_config,
    vocoder,
    voice_tensors,
)


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


def test_stream_rejects_duplicates_empty_final_and_decoder_failure_then_recovers() -> (
    None
):
    codec = streaming_codec()
    codec.begin("duplicate", voice_tensors(), seed=1)
    with pytest.raises(ValueError, match="already active"):
        codec.begin("duplicate", voice_tensors(), seed=2)
    with pytest.raises(ValueError, match="empty"):
        codec.push("duplicate", [], final=True)
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
