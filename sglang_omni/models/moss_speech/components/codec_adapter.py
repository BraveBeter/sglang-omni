"""Component-selective codec loading, voice enrollment and serial eager decoding."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

import torch
import torchaudio
from safetensors.torch import load_file
from transformers import WhisperFeatureExtractor

from sglang_omni.models.moss_speech.components.voice import (
    CampplusSpeakerEncoder,
    VoiceConditioning,
    build_voice_conditioning,
)
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import (
    AudioDecoder,
    load_audio,
)
from sglang_omni.vendor.moss_speech_codec.hf_codec.utils import (
    WhisperVQConfig,
    extract_speech_token,
)
from sglang_omni.vendor.moss_speech_codec.hf_codec.whisper import WhisperVQEncoder

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
AudioInput = str | Path | tuple[torch.Tensor, int] | torch.Tensor


class CodecAdapterError(RuntimeError):
    """The selected component is unavailable or the adapter is closed."""


class MossSpeechCodecAdapter:
    def __init__(
        self,
        codec_path: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        load_encoder: bool = True,
        load_decoder: bool = True,
    ) -> None:
        if dtype is not torch.float32:
            raise NotImplementedError("The MOSS-Speech codec requires FP32")
        else:
            pass
        if not load_encoder and not load_decoder:
            raise ValueError("At least one codec component must be enabled")
        else:
            pass
        self.device = torch.device(device)
        self.dtype = dtype
        self.is_closed = False
        self.decode_lock = threading.Lock()
        self.codec_directory = Path(codec_path)
        required_files = ["flow/campplus.onnx"] if load_encoder else []
        if load_encoder:
            required_files.extend(
                ["config.json", "model.safetensors", "preprocessor_config.json"]
            )
        else:
            pass
        if load_decoder:
            required_files.extend(["flow/flow.pt", "flow/hift.pt"])
        else:
            pass
        missing_files = [
            name
            for name in required_files
            if not (self.codec_directory / name).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                f"Codec directory {self.codec_directory} is missing {missing_files}"
            )
        else:
            pass
        self.sessions: set[str] = set()
        self.encoder: WhisperVQEncoder | None = None
        self.feature_extractor: WhisperFeatureExtractor | None = None
        self.decoder: AudioDecoder | None = None
        self.speaker_encoder: CampplusSpeakerEncoder | None = None
        if load_encoder:
            self.load_encoder()
            self.speaker_encoder = CampplusSpeakerEncoder(
                self.codec_directory / "flow/campplus.onnx"
            )
        else:
            pass
        if load_decoder:
            self.decoder = AudioDecoder(
                self.codec_directory / "flow/flow.pt",
                self.codec_directory / "flow/hift.pt",
                device=self.device,
            ).eval()
        else:
            pass

    def load_encoder(self) -> None:
        encoder_config = WhisperVQConfig.from_pretrained(
            str(self.codec_directory / "config.json")
        )
        encoder = WhisperVQEncoder(encoder_config).to(self.device, self.dtype)
        checkpoint = load_file(str(self.codec_directory / "model.safetensors"))
        encoder_weights = {
            name[len("encoder.") :]: tensor
            for name, tensor in checkpoint.items()
            if name.startswith("encoder.")
            and name not in {"encoder.ema_count", "encoder.ema_weight"}
        }
        missing_weights, unexpected_weights = encoder.load_state_dict(
            encoder_weights, strict=False
        )
        missing_weights = [
            name for name in missing_weights if "masked_spec_embed" not in name
        ]
        if missing_weights or unexpected_weights:
            raise RuntimeError(
                f"Encoder checkpoint mismatch: missing={missing_weights[:5]}, unexpected={unexpected_weights[:5]}"
            )
        else:
            pass
        self.encoder = encoder.eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            str(self.codec_directory)
        )

    @property
    def encoder_loaded(self) -> bool:
        return self.encoder is not None

    @property
    def decoder_loaded(self) -> bool:
        return self.decoder is not None

    def require_open(self) -> None:
        if self.is_closed:
            raise CodecAdapterError("Adapter is closed")
        else:
            pass

    @staticmethod
    def normalize_audio(
        audio: AudioInput, sampling_rate: int | None
    ) -> tuple[torch.Tensor, int]:
        if isinstance(audio, (str, Path)):
            waveform, sample_rate = load_audio(audio)
        elif isinstance(audio, torch.Tensor):
            waveform = audio if audio.dim() == 2 else audio.unsqueeze(0)
            sample_rate = sampling_rate or SAMPLE_RATE_IN
        elif isinstance(audio, tuple):
            waveform, sample_rate = audio
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                pass
        else:
            raise TypeError(f"Unsupported audio input {type(audio)!r}")
        return waveform.to(torch.float32), int(sample_rate)

    @torch.no_grad()
    def encode(
        self,
        inputs: Sequence[AudioInput] | torch.Tensor,
        *,
        sampling_rate: int | None = None,
        batch_size: int = 128,
    ) -> list[list[int]]:
        self.require_open()
        if self.encoder is None or self.feature_extractor is None:
            raise CodecAdapterError("Codec encoder was not loaded")
        else:
            pass
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() in (1, 2):
                inputs = [inputs]
            elif inputs.dim() == 3:
                inputs = list(inputs.unbind(0))
            else:
                raise ValueError("Tensor inputs must have one, two or three dimensions")
        else:
            pass
        waveforms = [self.normalize_audio(audio, sampling_rate) for audio in inputs]
        return extract_speech_token(
            self.encoder, self.feature_extractor, waveforms, batch_size=batch_size
        )

    @torch.no_grad()
    def encode_voice_ref(
        self, audio: AudioInput, *, sampling_rate: int | None = None
    ) -> VoiceConditioning:
        self.require_open()
        if self.encoder is None or self.speaker_encoder is None:
            raise CodecAdapterError("Voice enrollment requires the codec encoder")
        else:
            pass
        waveform, sample_rate = self.normalize_audio(audio, sampling_rate)
        prompt_codes = self.encode([(waveform, sample_rate)])[0]
        if sample_rate != SAMPLE_RATE_OUT:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=SAMPLE_RATE_OUT
            )(waveform)
        else:
            pass
        return build_voice_conditioning(waveform, prompt_codes, self.speaker_encoder)

    @torch.no_grad()
    def decode(
        self,
        codes: Sequence[int] | torch.Tensor,
        voice: VoiceConditioning,
        *,
        request_id: str,
    ) -> tuple[int, torch.Tensor]:
        self.require_open()
        if self.decoder is None:
            raise CodecAdapterError("Codec decoder was not loaded")
        else:
            pass
        if isinstance(codes, torch.Tensor):
            codes = codes.detach().cpu().reshape(-1).tolist()
        else:
            pass
        if not codes or any(
            isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 16384
            for code in codes
        ):
            raise ValueError("Codec codes must be nonempty integers within [0, 16384)")
        else:
            pass
        if request_id in self.sessions:
            raise CodecAdapterError(
                f"Request {request_id!r} already has an active decode"
            )
        else:
            pass
        self.sessions.add(request_id)
        try:
            with self.decode_lock:
                waveform = self.decoder.decode(
                    torch.tensor([list(codes)], dtype=torch.long, device=self.device),
                    voice.to_dict(),
                )
        finally:
            self.cleanup(request_id)
        return SAMPLE_RATE_OUT, waveform.squeeze().detach().cpu()

    def cleanup(self, request_id: str) -> None:
        self.sessions.discard(request_id)

    def close(self) -> None:
        self.is_closed = True
        self.sessions.clear()
        self.decoder = None
        self.encoder = None
        self.feature_extractor = None
        self.speaker_encoder = None

    def __enter__(self) -> MossSpeechCodecAdapter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
