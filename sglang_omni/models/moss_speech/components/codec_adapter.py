"""MOSS-Speech codec adapter (P1 prototype).

Wraps the vendored inference closure with serving-oriented semantics:

- component-selective loading: encoder-only and decoder-only modes never
  construct (or allocate) the disabled side;
- tensor-native voice conditioning (see `voice.py`) — the reference's
  file-path prompt API is not used on the hot path;
- V1 contract: codec FP32, serial decode (`max_concurrency=1` semantics via
  an internal lock), one finalize-per-request decode, idempotent cleanup.

Decode determinism note: the vendored flow derives its noise from a fixed
buffer built at construction (seed 0, RNG-scoped), so `decode` is a pure
function of (codes, conditioning, weights); no per-request seeding exists.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import soundfile as _sf
import torch
from transformers import WhisperFeatureExtractor

from .hf_codec.modeling import AudioDecoder, _load_audio
from .hf_codec.utils import WhisperVQConfig, extract_speech_token
from .hf_codec.whisper import WhisperVQEncoder
from .voice import CampplusSpeakerEncoder, VoiceConditioning

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000


class CodecAdapterError(RuntimeError):
    """Base error for adapter misuse (component not loaded / closed)."""


class MossSpeechCodecAdapter:
    """Serving adapter over the vendored MOSS-Speech codec closure."""

    def __init__(
        self,
        codec_path: Union[str, os.PathLike],
        *,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        load_encoder: bool = True,
        load_decoder: bool = True,
    ) -> None:
        if dtype is not torch.float32:
            raise NotImplementedError("V1 codec baseline is FP32 only; low-precision is a T1.3 optional experiment")
        if not load_encoder and not load_decoder:
            raise ValueError("at least one of load_encoder/load_decoder must be true")
        self._device = torch.device(device)
        self._dtype = dtype
        self._closed = False
        self._decode_lock = threading.Lock()
        self._codec_dir = Path(codec_path)
        self._validate_codec_dir(load_encoder=load_encoder, load_decoder=load_decoder)
        self._sessions: Dict[str, bool] = {}

        self._encoder: Optional[WhisperVQEncoder] = None
        self._feature_extractor: Optional[WhisperFeatureExtractor] = None
        self._decoder: Optional[AudioDecoder] = None
        self._speaker_encoder: Optional[CampplusSpeakerEncoder] = None

        if load_encoder:
            self._load_encoder()
        if load_decoder:
            self._load_decoder()
        # campplus is cheap and needed for voice enrollment on both configs
        # where decode-side access is unavailable (encoder-only deployments).
        self._speaker_encoder = CampplusSpeakerEncoder(self._codec_dir / "flow" / "campplus.onnx")

    # ------------------------------------------------------------------ load
    def _validate_codec_dir(self, *, load_encoder: bool, load_decoder: bool) -> None:
        required: List[str] = []
        if load_encoder:
            required += ["config.json", "model.safetensors", "preprocessor_config.json"]
        if load_decoder:
            required += ["flow/flow.pt", "flow/hift.pt", "flow/campplus.onnx"]
        if load_encoder or load_decoder:
            required.append("flow/campplus.onnx")
        missing = [r for r in required if not (self._codec_dir / r).exists()]
        if missing:
            raise FileNotFoundError(
                f"codec directory {str(self._codec_dir)!r} is missing required files: {missing}"
            )

    def _load_encoder(self) -> None:
        from safetensors.torch import load_file

        config = WhisperVQConfig.from_pretrained(str(self._codec_dir / "config.json"))
        encoder = WhisperVQEncoder(config).to(self._device, self._dtype)
        state = load_file(str(self._codec_dir / "model.safetensors"))
        stripped = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(stripped, strict=False)
        real_missing = [k for k in missing if "masked_spec_embed" not in k]
        if real_missing:
            raise RuntimeError(f"encoder load: missing keys {real_missing[:5]}")
        encoder.eval()
        self._encoder = encoder
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(str(self._codec_dir))

    def _load_decoder(self) -> None:
        decoder = AudioDecoder(
            flow_ckpt_path=self._codec_dir / "flow" / "flow.pt",
            hift_ckpt_path=self._codec_dir / "flow" / "hift.pt",
            campplus_model=self._codec_dir / "flow" / "campplus.onnx",
            device=self._device,
        ).eval()
        self._decoder = decoder

    # ------------------------------------------------------------------ state
    @property
    def encoder_loaded(self) -> bool:
        return self._encoder is not None

    @property
    def decoder_loaded(self) -> bool:
        return self._decoder is not None

    def _require_open(self) -> None:
        if self._closed:
            raise CodecAdapterError("adapter is closed")

    def _require_encoder(self) -> Tuple[WhisperVQEncoder, WhisperFeatureExtractor]:
        self._require_open()
        if self._encoder is None or self._feature_extractor is None:
            raise CodecAdapterError("codec encoder not loaded (load_encoder=False)")
        return self._encoder, self._feature_extractor

    def _require_decoder(self) -> AudioDecoder:
        self._require_open()
        if self._decoder is None:
            raise CodecAdapterError("codec decoder not loaded (load_decoder=False)")
        return self._decoder

    # ------------------------------------------------------------------ input
    @staticmethod
    def _normalize_audio(
        inputs: Union[str, os.PathLike, Tuple[torch.Tensor, int], torch.Tensor],
        sampling_rate: Optional[int],
    ) -> Tuple[torch.Tensor, int]:
        if isinstance(inputs, (str, os.PathLike)):
            wav, sr = _load_audio(inputs)
        elif isinstance(inputs, torch.Tensor):
            wav = inputs if inputs.dim() == 2 else inputs.unsqueeze(0)
            sr = sampling_rate or SAMPLE_RATE_IN
        elif isinstance(inputs, tuple):
            wav, sr = inputs
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        else:
            raise TypeError(f"unsupported audio input {type(inputs)!r}")
        return wav.to(torch.float32), int(sr)

    # ---------------------------------------------------------------- encode
    @torch.no_grad()
    def encode(
        self,
        inputs: Union[Sequence[Union[str, os.PathLike, Tuple[torch.Tensor, int], torch.Tensor]], torch.Tensor],
        *,
        sampling_rate: Optional[int] = None,
        batch_size: int = 128,
    ) -> List[List[int]]:
        """Audio -> codec codes. `batch_size` = encoder internal chunk batch."""
        encoder, feature_extractor = self._require_encoder()
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 2:
                inputs = [inputs]
            elif inputs.dim() == 3:
                inputs = [inputs[i] for i in range(inputs.size(0))]
            else:
                raise ValueError("tensor inputs must be (T,), (C, T) or (B, C, T)")
        items = [self._normalize_audio(item, sampling_rate) for item in inputs]
        return extract_speech_token(encoder, feature_extractor, items, batch_size=batch_size)

    # ----------------------------------------------------------------- voice
    @torch.no_grad()
    def encode_voice_ref(
        self,
        wav: Union[str, os.PathLike, torch.Tensor, Tuple[torch.Tensor, int]],
        *,
        sampling_rate: Optional[int] = None,
    ) -> VoiceConditioning:
        """Reference audio -> full voice conditioning.

        Resample-chain parity with the reference decode path (matters for
        bit-exactness): prompt **codes** are encoded from the ORIGINAL sample
        rate directly to 16 kHz (reference `codec.encode([path])`), while
        mel/xvector go through the 24 kHz chain (reference `_extract_speech_feat`
        + 24k->16k resample). Uses only the encoder + standalone feature
        paths; the decoder stack is never required.
        """
        from .voice import MossSpeechVoiceHook

        self._require_open()
        if self._encoder is None:
            raise CodecAdapterError("voice enrollment requires the codec encoder")
        audio, sr = self._normalize_audio(wav, sampling_rate)
        orig_audio, orig_sr = audio, sr  # codes path: original rate -> 16 kHz
        import torchaudio

        if sr != SAMPLE_RATE_OUT:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE_OUT)(audio)
        hook = MossSpeechVoiceHook(
            encode_codes_fn=lambda w: self.encode([(orig_audio, orig_sr)])[0],
            speaker_encoder=self._speaker_encoder,
        )
        return hook.encode_one(hook.normalize_input(audio))

    # ---------------------------------------------------------------- decode
    @torch.no_grad()
    def decode(
        self,
        codes: Union[Sequence[int], torch.Tensor],
        voice: VoiceConditioning,
        *,
        request_id: str,
    ) -> Tuple[int, torch.Tensor]:
        """Codes + voice -> (24000, waveform). Serial (V1), finalize-per-call."""
        decoder = self._require_decoder()
        self._require_open()
        if request_id in self._sessions:
            raise CodecAdapterError(f"request {request_id!r} already has an active decode session")
        if isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().reshape(-1).tolist()
        code_list = [int(c) for c in codes]
        for c in code_list:
            if not (0 <= c < 16384):
                raise ValueError(f"code {c} out of valid range [0, 16384)")
        self._sessions[request_id] = True
        try:
            with self._decode_lock:
                result = decoder.token2wav(
                    torch.tensor([code_list], dtype=torch.long, device=self._device),
                    uuid=request_id,
                    prompt_token=voice.prompt_token,
                    prompt_feat=voice.prompt_feat,
                    embedding=voice.embedding,
                    finalize=True,
                )
        finally:
            # finalize=True already drops per-uuid caches; cleanup is defensive
            decoder.release_session(request_id)
            self._sessions.pop(request_id, None)
        return SAMPLE_RATE_OUT, result[0].squeeze().detach().cpu()

    # --------------------------------------------------------------- cleanup
    def cleanup(self, request_id: str) -> None:
        """Idempotent request cleanup; never touches shared voice cache."""
        if self._decoder is not None:
            self._decoder.release_session(request_id)
        self._sessions.pop(request_id, None)

    def close(self) -> None:
        """Idempotent shutdown. Shared/default voice artifacts are owned by the
        ReferenceEncodeService (if used) and are NOT released here."""
        if self._closed:
            return
        self._closed = True
        for request_id in list(self._sessions):
            self.cleanup(request_id)
        self._decoder = None
        self._encoder = None
        self._feature_extractor = None
        self._speaker_encoder = None

    def __enter__(self) -> "MossSpeechCodecAdapter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
