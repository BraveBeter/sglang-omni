# SPDX-License-Identifier: Apache-2.0
"""Small neural boundaries for the real MOSS codec and scheduler contracts."""

import queue
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import WhisperFeatureExtractor

import sglang_omni.models.moss_speech.components.codec_adapter as codec_adapter_module
from sglang_omni.models.moss_speech.components.codec_adapter import (
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.payload_types import (
    EOSP_TOKEN_ID,
    MODALITY_PAD_TOKEN_ID,
    MossSpeechState,
)
from sglang_omni.models.moss_speech.streaming import (
    MossSpeechStreamingText,
    MossSpeechStreamingVocoder,
)
from sglang_omni.models.moss_speech.voice_transport import (
    pack_voice_payload,
    voice_payload_digest,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import scoped_global_rng
from sglang_omni.vendor.moss_speech_codec.hf_codec.utils import WhisperVQConfig
from sglang_omni.vendor.moss_speech_codec.hf_codec.whisper import WhisperVQEncoder


class TinyFlow(torch.nn.Module):
    pre_lookahead_len = 3
    token_mel_ratio = 4

    def inference(
        self,
        *,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_len: torch.Tensor,
        embedding: torch.Tensor,
        streaming: bool,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generated_codes = token.shape[1] - (0 if finalize else self.pre_lookahead_len)
        predicted_mel = torch.randn(1, 80, generated_codes * self.token_mel_ratio)
        return predicted_mel, predicted_mel


class TinyHiFT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.should_fail = False

    def inference(
        self, *, speech_feat: torch.Tensor, cache_source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.should_fail:
            raise RuntimeError("Injected HiFT failure")
        else:
            waveform = speech_feat.mean(dim=1).repeat_interleave(480, dim=-1)
            waveform = waveform + torch.randn_like(waveform)
            return waveform, waveform.unsqueeze(1)


class TinyOfflineDecoder(torch.nn.Module):
    def __init__(
        self, flow_ckpt_path: Path, hift_ckpt_path: Path, *, device: torch.device
    ) -> None:
        super().__init__()
        self.device = device
        self.checkpoint_paths = (flow_ckpt_path, hift_ckpt_path)
        self.should_fail = False
        self.received_codes: list[list[int]] = []

    def decode(
        self, audio_token: torch.Tensor, voice: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        self.received_codes.append(audio_token[0].tolist())
        if self.should_fail:
            raise RuntimeError("Injected offline decoder failure")
        else:
            return torch.rand(1, audio_token.shape[1] * 1920)


class TinySpeakerEncoder:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.waveform_16khz: torch.Tensor | None = None

    def extract(self, waveform_16khz: torch.Tensor) -> torch.Tensor:
        self.waveform_16khz = waveform_16khz.clone()
        return torch.zeros(1, 192)


class Utf8Tokenizer:
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


def voice_tensors() -> dict[str, torch.Tensor]:
    return {
        "prompt_token": torch.tensor([[0, 16383]], dtype=torch.int32),
        "prompt_feat": torch.linspace(-1, 1, 8 * 80).reshape(1, 8, 80),
        "embedding": torch.linspace(-1, 1, 192).reshape(1, 192),
    }


def streaming_codec(*, chunk_size: int = 5) -> MossSpeechStreamingCodec:
    return MossSpeechStreamingCodec(
        None, chunk_size=chunk_size, device="cpu", flow=TinyFlow(), hift=TinyHiFT()
    )


def tiny_whisper_config() -> WhisperVQConfig:
    return WhisperVQConfig(
        vocab_size=32,
        num_mel_bins=80,
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=16,
        max_source_positions=64,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pooling_kernel_size=2,
        pooling_position=1,
        quantize_vocab_size=16,
        quantize_position=1,
        quantize_encoder_only=True,
        pooling_type="avg",
        encoder_causal_attention=True,
        encoder_causal_convolution=True,
        quantize_causal_encoder=True,
        quantize_causal_block_size=4,
        decoder_start_token_id=1,
        start_token_id=1,
    )


def encoder_adapter(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> MossSpeechCodecAdapter:
    configuration = tiny_whisper_config()
    with scoped_global_rng():
        encoder = WhisperVQEncoder(configuration).eval()
    configuration.save_pretrained(directory)
    checkpoint = {
        f"encoder.{name}": tensor for name, tensor in encoder.state_dict().items()
    }
    checkpoint.update(
        {"encoder.ema_count": torch.zeros(16), "encoder.ema_weight": torch.zeros(16, 8)}
    )
    save_file(checkpoint, directory / "model.safetensors")
    WhisperFeatureExtractor().save_pretrained(directory)
    (directory / "flow").mkdir(exist_ok=True)
    (directory / "flow/campplus.onnx").touch()
    monkeypatch.setattr(
        codec_adapter_module, "CampplusSpeakerEncoder", TinySpeakerEncoder
    )
    return MossSpeechCodecAdapter(
        directory, device="cpu", load_encoder=True, load_decoder=False
    )


def offline_adapter(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> MossSpeechCodecAdapter:
    (directory / "flow").mkdir(parents=True, exist_ok=True)
    for name in ("flow.pt", "hift.pt"):
        (directory / "flow" / name).touch()
    monkeypatch.setattr(codec_adapter_module, "AudioDecoder", TinyOfflineDecoder)
    return MossSpeechCodecAdapter(
        directory, device="cpu", load_encoder=False, load_decoder=True
    )


def audio_payload(
    request_id: str,
    codes: list[int],
    *,
    seed: int = 7,
    finish_reason: str = "stop",
    stream: bool = True,
) -> StagePayload:
    packed_voice = pack_voice_payload(voice_tensors())
    rows = [[MODALITY_PAD_TOKEN_ID, code] for code in codes]
    if finish_reason == "stop":
        rows.append([MODALITY_PAD_TOKEN_ID, EOSP_TOKEN_ID])
    else:
        pass
    state = MossSpeechState(
        output_modality="audio",
        effective_seed=seed,
        voice_payload=packed_voice,
        voice_key=voice_payload_digest(packed_voice),
        output_grid=rows,
        finish_reason=finish_reason,
    )
    return StagePayload(
        request_id, OmniRequest(inputs=None, params={"stream": stream}), state.to_dict()
    )


def audio_chunk(codes: list[int], *, cursor: int = 0, seed: int = 7) -> StreamItem:
    metadata: dict[str, object] = {
        "stream": True,
        "modality": "audio_codes",
        "row_index": cursor,
    }
    if cursor == 0:
        packed_voice = pack_voice_payload(voice_tensors())
        metadata.update(
            seed=seed,
            voice_key=voice_payload_digest(packed_voice),
            voice_payload=packed_voice,
        )
    else:
        pass
    return StreamItem(
        chunk_id=cursor,
        from_stage="ar_engine",
        data=torch.tensor(codes, dtype=torch.int64),
        metadata=metadata,
    )


def vocoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> MossSpeechStreamingVocoder:
    return MossSpeechStreamingVocoder(
        streaming_codec(chunk_size=25), offline_adapter(tmp_path, monkeypatch)
    )


def drain_messages(
    scheduler: MossSpeechStreamingVocoder | MossSpeechStreamingText,
) -> list[OutgoingMessage]:
    messages: list[OutgoingMessage] = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages
