"""Trimmed port of fnlp/MOSS-Speech-Codec `utils.py` (snapshot eeec733e).

Kept for the inference closure: `WhisperVQConfig`, `extract_speech_token`
and the per-sample-rate resample buffer. The remaining ~1800 lines of the
source file (whisper training utilities, dataset helpers, tests) are dropped.

Documented deviations from the source (see components/VENDORED_SOURCES.md):
- `torchaudio.load(path)` replaced by a soundfile-backed loader returning the
  same float32 (channels, samples) tensor + sample rate (torchaudio 2.9 routes
  `load` through TorchCodec, whose wheels are not installable in the target
  environment).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import soundfile as _sf
import torch
import torchaudio
from transformers import WhisperConfig


def _load_audio(path: str) -> Tuple[torch.Tensor, int]:
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.copy()).T, sr


class WhisperVQConfig(WhisperConfig):
    """Whisper config with VQ-quantizer and causal-encoder extensions."""

    def __init__(
        self,
        pooling_kernel_size=None,
        pooling_type="max",
        pooling_position=0,
        quantize_vocab_size=None,
        quantize_position=16,
        quantize_commit_coefficient=0.25,
        quantize_loss_scale=1.0,
        quantize_ema_decay=None,
        quantize_restart_interval=None,
        quantize_encoder_only=False,
        quantize_causal_encoder=False,
        quantize_causal_block_size=None,
        skip_language_detection=False,
        encoder_causal_attention=False,
        encoder_causal_convolution=False,
        **kwargs,
    ):
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_type = pooling_type
        self.pooling_position = pooling_position
        self.quantize_vocab_size = quantize_vocab_size
        self.quantize_position = quantize_position
        self.quantize_commit_coefficient = quantize_commit_coefficient
        self.quantize_loss_scale = quantize_loss_scale
        self.quantize_ema_decay = quantize_ema_decay
        self.quantize_restart_interval = quantize_restart_interval
        self.quantize_encoder_only = quantize_encoder_only
        self.quantize_causal_encoder = quantize_causal_encoder
        self.quantize_causal_block_size = quantize_causal_block_size
        self.skip_language_detection = skip_language_detection
        self.encoder_causal_attention = encoder_causal_attention
        self.encoder_causal_convolution = encoder_causal_convolution
        super().__init__(**kwargs)


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def extract_speech_token(model, feature_extractor, utts, batch_size: int = 128):
    """Encode a list of audio inputs into codec token ids.

    Accepts `(waveform, sr)` tuples, 1D/2D tensors (sr assumed 16 kHz), or
    file paths. Chunks longer than 30 s are split into 30 s segments before
    the internal batching, then re-concatenated per input.
    """
    device = model.codebook.weight.device
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            elif isinstance(utt, torch.Tensor):
                if utt.ndim == 2:
                    audio = utt
                elif utt.ndim == 1:
                    audio = utt.unsqueeze(0)
                sample_rate = 16000
            else:
                audio, sample_rate = _load_audio(utt)
            audio = audio.to(device)
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=16000
                    ).to(device)
                    if hasattr(_resample_buffer[sample_rate], "kernel"):
                        _resample_buffer[sample_rate].kernel = _resample_buffer[sample_rate].kernel.to(device)
                audio = _resample_buffer[sample_rate](audio)
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000 : (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = (
            model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        )
        all_speech_tokens: List[List[int]] = [[] for _ in range(len(utts))]
        for start in range(0, len(audios), batch_size):
            features = feature_extractor(
                audios[start : start + batch_size],
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
                device=device,
                padding="longest",
                pad_to_multiple_of=stride,
            )
            features = features.to(device=device)

            outputs = model.forward(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[:, :: model.conv1.stride[0] * model.conv2.stride[0]]
            attention_mask = attention_mask[:, :: model.config.pooling_kernel_size]
            assert attention_mask.shape == speech_tokens.shape
            for i in range(len(speech_tokens)):
                idx = indices[start + i]
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)
        return all_speech_tokens
