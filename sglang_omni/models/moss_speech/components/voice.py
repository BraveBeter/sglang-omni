"""Tensor-native prompt codes, mel features and speaker conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from sglang_omni.vendor.moss_speech_codec.matcha_components.audio import mel_spectrogram

CODEC_REVISION = "eeec733e4e1dea7da444d332d8e1621ef257414c"
MEL_PARAMS = {
    "n_fft": 1920,
    "num_mels": 80,
    "sampling_rate": 24000,
    "hop_size": 480,
    "win_size": 1920,
    "fmin": 0,
    "fmax": 8000,
    "center": False,
}


@dataclass(kw_only=True)
class VoiceConditioning:
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor
    meta: dict[str, str | int]

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "prompt_token": self.prompt_token,
            "prompt_feat": self.prompt_feat,
            "embedding": self.embedding,
        }


def compute_voice_mel(waveform_24khz: torch.Tensor) -> torch.Tensor:
    mel_features = (
        mel_spectrogram(waveform_24khz, **MEL_PARAMS).squeeze(dim=0).transpose(0, 1)
    )
    return mel_features.unsqueeze(dim=0)


class CampplusSpeakerEncoder:
    """Extract the 192-dimensional speaker embedding with CPU ONNX execution."""

    def __init__(self, model_path: str | Path) -> None:
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    @torch.no_grad()
    def extract(self, waveform_16khz: torch.Tensor) -> torch.Tensor:
        mel_features = kaldi.fbank(
            waveform_16khz, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        mel_features = mel_features - mel_features.mean(dim=0, keepdim=True)
        embedding = self.session.run(
            None,
            {
                self.session.get_inputs()[0]
                .name: mel_features.unsqueeze(0)
                .cpu()
                .numpy()
            },
        )[0].flatten()
        return torch.from_numpy(np.asarray(embedding)).float().unsqueeze(0)


def build_voice_conditioning(
    waveform_24khz: torch.Tensor,
    prompt_codes: list[int],
    speaker_encoder: CampplusSpeakerEncoder,
) -> VoiceConditioning:
    prompt_mel = compute_voice_mel(waveform_24khz)
    prompt_code_count = min(int(prompt_mel.shape[1] / 4), len(prompt_codes))
    prompt_mel = prompt_mel[:, : 4 * prompt_code_count, :]
    prompt_token = torch.tensor([prompt_codes[:prompt_code_count]], dtype=torch.int32)
    waveform_16khz = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(
        waveform_24khz
    )
    speaker_embedding = speaker_encoder.extract(waveform_16khz)
    return VoiceConditioning(
        prompt_token=prompt_token,
        prompt_feat=prompt_mel,
        embedding=speaker_embedding,
        meta={"sr": 24000, "token_len": prompt_code_count, "revision": CODEC_REVISION},
    )
