"""CPU sample-rate regression for the retained HF compatibility API."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_speech.components.hf_codec import modeling
from sglang_omni.models.moss_speech.components.hf_codec.modeling import MossSpeechCodec
from sglang_omni.models.moss_speech.components.hf_codec.utils import (
    extract_speech_token,
)


def test_voice_conditioning_preserves_24khz():
    seen = {}

    class Features(dict):
        def __init__(self):
            super().__init__()
            self.attention_mask = torch.ones(1, 80)

        def to(self, **kwargs):
            return self

    def features(audios, **kwargs):
        seen["encoder_samples"] = len(audios[0])
        return Features()

    features.hop_length = 200
    encoder = SimpleNamespace(
        codebook=SimpleNamespace(weight=torch.zeros(1)),
        config=SimpleNamespace(pooling_kernel_size=1),
        conv1=SimpleNamespace(stride=[1]),
        conv2=SimpleNamespace(stride=[1]),
        forward=lambda **k: SimpleNamespace(
            quantized_token_ids=torch.ones(1, 80, dtype=torch.long)
        ),
    )

    def encode(inputs):
        assert isinstance(inputs[0], tuple) and inputs[0][1] == 24000
        return extract_speech_token(encoder, features, inputs)

    def mel(wav):
        seen["mel_samples"] = wav.shape[-1]
        return torch.zeros(1, 100, 80), torch.tensor([100])

    def speaker(wav):
        seen["speaker_samples"] = wav.shape[-1]
        return torch.zeros(1, 192)

    codec = SimpleNamespace(
        encode=encode, _extract_speech_feat=mel, _extract_spk_embedding=speaker
    )
    result = MossSpeechCodec.compute_voice_conditioning(codec, torch.zeros(24000))
    assert seen == {
        "encoder_samples": 16000,
        "mel_samples": 24000,
        "speaker_samples": 16000,
    }
    assert result["prompt_token"].shape == (1, 25)
    assert result["prompt_feat"].shape == (1, 100, 80)


def test_audio_decoder_applies_onnx_thread_options(monkeypatch):
    # Keep the actual constructor and ORT options, without loading GPU weights.
    flow = torch.nn.Identity()
    flow.input_frame_rate = 12.5
    monkeypatch.setattr(modeling, "_build_flow_decoder", lambda: flow)
    monkeypatch.setattr(modeling, "_build_hift", torch.nn.Identity)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})

    def session(path, sess_options=None, providers=None, **kwargs):
        # ORT silently accepts unknown kwargs; a misspelling restores defaults.
        options = sess_options or modeling.onnxruntime.SessionOptions()
        assert options.intra_op_num_threads == 1
        assert providers == ["CPUExecutionProvider"]
        return SimpleNamespace(get_session_options=lambda: options)

    monkeypatch.setattr(modeling.onnxruntime, "InferenceSession", session)
    decoder = modeling.AudioDecoder("flow.pt", "hift.pt", "campplus.onnx", "cpu")
    assert decoder.campplus_session.get_session_options().intra_op_num_threads == 1


@pytest.mark.parametrize(
    "configured,explicit,torchscript,expected_dict",
    [
        (True, None, False, True),
        (False, None, False, False),
        (False, True, False, True),
        (True, False, False, False),
        (True, None, True, False),
    ],
)
def test_whisper_return_format_without_deprecated_config_access(
    monkeypatch, configured, explicit, torchscript, expected_dict
):
    from sglang_omni.models.moss_speech.components.hf_codec.utils import WhisperVQConfig
    from sglang_omni.models.moss_speech.components.hf_codec.whisper import (
        WhisperVQEncoder,
    )

    def deprecated_access(config):
        raise AssertionError("use_return_dict is deprecated")

    config = WhisperVQConfig(
        d_model=8,
        num_mel_bins=4,
        encoder_layers=0,
        encoder_attention_heads=2,
        max_source_positions=4,
        pad_token_id=0,
        return_dict=configured,
        torchscript=torchscript,
    )
    encoder = WhisperVQEncoder(config).eval()
    monkeypatch.setattr(WhisperVQConfig, "use_return_dict", property(deprecated_access))
    with torch.no_grad():
        result = encoder(torch.zeros(1, 4, 8), return_dict=explicit)
    assert isinstance(result, dict) is expected_dict
    hidden = result.last_hidden_state if expected_dict else result[0]
    assert hidden.shape == (1, 4, 8)
