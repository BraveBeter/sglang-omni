"""Voice conditioning for the MOSS-Speech codec adapter.

`VoiceConditioning` bundles the three conditioning artifacts the flow decoder
needs (prompt codec codes, prompt 80-mel @24 kHz, campplus xvector @16 kHz).
Mel and xvector extraction are standalone (no flow/HiFT weights required),
so voice enrollment never loads the decoder stack.

Caching reuses the framework `ReferenceEncodeService` via a
`KeyedReferenceEncodeHook` (same pattern as fun_cosyvoice3).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from .matcha_components.audio import mel_spectrogram
from sglang_omni.scheduling.reference_encoder import KeyedReferenceEncodeHook

# Frozen feature parameters (transcribed from the locked codec flow/config.yaml).
MEL_PARAMS: Dict[str, Any] = {
    "n_fft": 1920,
    "num_mels": 80,
    "sampling_rate": 24000,
    "hop_size": 480,
    "win_size": 1920,
    "fmin": 0,
    "fmax": 8000,
    "center": False,
}
XVECTOR_DIM = 192
CODEC_MODEL_ID = "fnlp/MOSS-Speech-Codec"
CODEC_REVISION = "eeec733e4e1dea7da444d332d8e1621ef257414c"


@dataclass
class VoiceConditioning:
    """Per-voice decode conditioning tensors (caller-owned clones)."""

    prompt_token: torch.Tensor  # (1, T) int32 — prompt codec codes
    prompt_feat: torch.Tensor  # (1, F, 80) float32 — prompt mel @24 kHz
    embedding: torch.Tensor  # (1, 192) float32 — campplus xvector
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "prompt_token": self.prompt_token,
            "prompt_feat": self.prompt_feat,
            "embedding": self.embedding,
        }


def compute_voice_mel(wav_24k: torch.Tensor) -> torch.Tensor:
    """Prompt mel features, mirroring AudioDecoder._extract_speech_feat."""
    feat = mel_spectrogram(wav_24k, **MEL_PARAMS).squeeze(dim=0).transpose(0, 1)
    return feat.unsqueeze(dim=0)


class CampplusSpeakerEncoder:
    """campplus ONNX xvector extractor (CPU execution, reference parity)."""

    def __init__(self, model_path: str) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    @torch.no_grad()
    def extract(self, wav_16k: torch.Tensor) -> torch.Tensor:
        feat = kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        out = self._session.run(
            None, {self._session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()}
        )[0].flatten()
        return torch.from_numpy(np.asarray(out)).float().unsqueeze(0)  # (1, 192)


@dataclass
class _VoiceRefInput:
    wav_24k: torch.Tensor  # (1, T) float32
    input_key: str


class MossSpeechVoiceHook(
    KeyedReferenceEncodeHook[_VoiceRefInput, VoiceConditioning, Dict[str, Any]]
):
    """Cache hook for voice conditioning artifacts (LRU + single-flight)."""

    model_id = CODEC_MODEL_ID
    encoder_id = "whisper_vq_codes+matcha_mel+campplus"
    artifact_kind = "moss_speech_voice"

    def __init__(
        self,
        *,
        encode_codes_fn: Callable[[torch.Tensor], list[int]],
        speaker_encoder: CampplusSpeakerEncoder,
    ) -> None:
        self._encode_codes_fn = encode_codes_fn
        self._speaker_encoder = speaker_encoder
        self.model_revision = CODEC_REVISION
        self.encoder_config_hash = hashlib.sha256(
            json.dumps({"mel": MEL_PARAMS, "xvector": "campplus+fbank80@16k"}, sort_keys=True).encode()
        ).hexdigest()[:16]

    def normalize_input(self, raw_input: Any) -> _VoiceRefInput:
        if isinstance(raw_input, _VoiceRefInput):
            return raw_input
        if isinstance(raw_input, VoiceConditioning):
            raise TypeError("VoiceConditioning is an output artifact, not an input")
        wav = raw_input
        if not isinstance(wav, torch.Tensor):
            raise TypeError(f"expected 24 kHz float tensor, got {type(wav)!r}")
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        digest = hashlib.blake2b(wav.detach().cpu().numpy().tobytes(), digest_size=16).hexdigest()
        return _VoiceRefInput(wav_24k=wav.to(torch.float32), input_key=digest)

    def input_key(self, item: _VoiceRefInput) -> str | None:
        return item.input_key

    def options_key(self, item: _VoiceRefInput) -> str:
        return "24k_mono_codes+mel+xvector"

    def encode_one(self, item: _VoiceRefInput) -> VoiceConditioning:
        wav = item.wav_24k
        codes = self._encode_codes_fn(wav)
        speech_feat = compute_voice_mel(wav)
        token_len = min(int(speech_feat.shape[1] / 4), len(codes))
        speech_feat = speech_feat[:, : 4 * token_len, :]
        prompt_token = torch.tensor([codes[:token_len]], dtype=torch.int32)
        wav_16k = torchaudio_resample_24k_to_16k(wav)
        embedding = self._speaker_encoder.extract(wav_16k)
        return VoiceConditioning(
            prompt_token=prompt_token,
            prompt_feat=speech_feat,
            embedding=embedding,
            meta={"sr": 24000, "token_len": token_len, "revision": CODEC_REVISION},
        )

    def store_artifact(self, artifact: VoiceConditioning) -> Dict[str, Any]:
        return {
            "prompt_token": artifact.prompt_token.detach().cpu().clone(),
            "prompt_feat": artifact.prompt_feat.detach().cpu().clone(),
            "embedding": artifact.embedding.detach().cpu().clone(),
            "meta": dict(artifact.meta),
        }

    def load_artifact(self, stored: Dict[str, Any]) -> VoiceConditioning:
        def _clone(t: torch.Tensor) -> torch.Tensor:
            return t.to(device="cpu", dtype=torch.float32).clone() if t.dtype.is_floating_point else t.clone()

        return VoiceConditioning(
            prompt_token=stored["prompt_token"].to(torch.int32).clone(),
            prompt_feat=_clone(stored["prompt_feat"]),
            embedding=_clone(stored["embedding"]),
            meta=dict(stored["meta"]),
        )


def torchaudio_resample_24k_to_16k(wav: torch.Tensor) -> torch.Tensor:
    import torchaudio

    return torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(wav)
