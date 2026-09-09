"""CPU sample-rate regression for the retained HF compatibility API."""

from types import SimpleNamespace

import torch

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
