# SPDX-License-Identifier: Apache-2.0
"""Stream generated codes through the chunk-trained Flow and shared HiFT."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Protocol

import torch
from hyperpyyaml import load_hyperpyyaml
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.moss_speech.codec import (
    SAMPLE_RATE_HZ,
    MossSpeechCodec,
    VoiceConditioning,
    codec_rng,
)
from sglang_omni.models.moss_speech.flow_attention import (
    install_flow_attention_broadcast,
)
from sglang_omni.models.moss_speech.flow_precision import FlowEulerDecoder
from sglang_omni.models.moss_speech.fsm import EOSP, MODALITY_PAD
from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechSGLangRequestData,
    MossSpeechState,
    RequestValidationError,
    extract_output_codes,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.types import RequestOutput

STREAM_CHUNK_CODES = 25
HIFT_OVERLAP_MEL_FRAMES = 8
SAMPLES_PER_MEL_FRAME = 480
HIFT_OVERLAP_SAMPLES = HIFT_OVERLAP_MEL_FRAMES * SAMPLES_PER_MEL_FRAME


class StreamingFlow(Protocol):
    pre_lookahead_len: int
    token_mel_ratio: int

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
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def audio_stream_targets(request_id: str, payload: StagePayload) -> list[str]:
    return ["audio_vocoder"] if payload.request.params.get("stream") is True else []


def build_audio_stream(
    request_id: str,
    request_data: MossSpeechSGLangRequestData,
    request_output: RequestOutput,
) -> list[OutgoingMessage]:
    if not request_data.stream_audio or request_data.stream_audio_ended:
        return []
    else:
        pass
    rows = request_data.output_rows[request_data.streamed_rows :]
    codes = extract_output_codes(rows)
    request_data.streamed_rows = len(request_data.output_rows)
    request_data.stream_audio_ended = any(
        text_token == MODALITY_PAD and audio_token == EOSP
        for text_token, audio_token in rows
    )
    if not codes:
        return []
    else:
        pass
    metadata: dict[str, object] = {"stream": True}
    if not request_data.stream_audio_started:
        payload = request_data.stage_payload
        assert payload is not None
        metadata["reference_audio"] = MossSpeechState.from_dict(
            payload.data
        ).reference_audio
        metadata["effective_seed"] = request_data.effective_seed
        request_data.stream_audio_started = True
    else:
        pass
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            target="audio_vocoder",
            data=torch.tensor(codes, dtype=torch.long),
            metadata=metadata,
        )
    ]


@dataclass(kw_only=True)
class MossSpeechStreamState:
    voice: VoiceConditioning | None = None
    codes: list[int] = field(default_factory=list)
    decoded_codes: int = 0
    emitted_samples: int = 0
    cpu_rng: torch.Tensor | None = None
    cuda_rng: torch.Tensor | None = None
    overlap_mel: torch.Tensor | None = None
    overlap_source: torch.Tensor | None = None
    overlap_waveform: torch.Tensor | None = None


class MossSpeechStreamingVocoder(StreamingVocoderBase[MossSpeechStreamState, None]):
    def __init__(
        self,
        codec: MossSpeechCodec,
        codec_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        *,
        flow_dtype: torch.dtype | None = None,
        streaming_flow_steps: int | None = None,
    ) -> None:
        self.codec: MossSpeechCodec = codec
        self.codec_path: Path = codec_path
        self.flow_dtype: torch.dtype | None = flow_dtype
        self.streaming_flow_steps: int | None = streaming_flow_steps
        self.flow: StreamingFlow | None = None
        self.fade_window: torch.Tensor = torch.hamming_window(
            2 * HIFT_OVERLAP_SAMPLES, periodic=False, dtype=torch.float64
        )
        super().__init__(
            partial(codec.decode, tokenizer=tokenizer),
            sample_rate=SAMPLE_RATE_HZ,
            stream_source_hint="MOSS-Speech",
        )

    def create_stream_state(self, request_id: str) -> MossSpeechStreamState:
        return MossSpeechStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: MossSpeechStreamState,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        if isinstance(source, StagePayload) or state.voice is not None:
            return
        else:
            reference_audio = source.get("reference_audio")
            effective_seed = source.get("effective_seed")
        assert isinstance(reference_audio, str) and isinstance(effective_seed, int)
        state.voice = self.codec.encode_reference(reference_audio)
        if self.flow_dtype is torch.bfloat16:
            state.voice["prompt_feat"] = state.voice["prompt_feat"].bfloat16()
            state.voice["embedding"] = state.voice["embedding"].bfloat16()
        else:
            pass
        state.cpu_rng = torch.Generator().manual_seed(effective_seed).get_state()
        if self.codec.device.type == "cuda":
            state.cuda_rng = (
                torch.Generator(device=self.codec.device)
                .manual_seed(effective_seed)
                .get_state()
            )
        else:
            pass
        if self.flow is None:
            flow_directory = self.codec_path / "flow"
            checkpoint_path = flow_directory / f"flow-chunk-{STREAM_CHUNK_CODES}.pt"
            with codec_rng(), (flow_directory / "config.yaml").open() as config:
                flow = load_hyperpyyaml(config, {"chunk_size": STREAM_CHUNK_CODES})[
                    "flow"
                ]
                weights = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                weights.pop("epoch", None)
                weights.pop("step", None)
                flow.load_state_dict(weights)
                if self.flow_dtype is torch.bfloat16:
                    flow.bfloat16()
                else:
                    pass
                if (
                    self.flow_dtype is torch.bfloat16
                    or self.streaming_flow_steps is not None
                ):
                    flow.decoder = FlowEulerDecoder(
                        flow.decoder, streaming_flow_steps=self.streaming_flow_steps
                    ).eval()
                else:
                    pass
                install_flow_attention_broadcast(flow.decoder)
                self.flow = flow.to(self.codec.device).eval()
        else:
            pass

    def validate_chunk(
        self, request_id: str, state: MossSpeechStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        assert codes.ndim == 1 and codes.dtype == torch.long
        return codes

    def ingest(
        self, request_id: str, state: MossSpeechStreamState, codes: torch.Tensor
    ) -> None:
        state.codes.extend(codes.tolist())

    @torch.inference_mode()
    def decode_delta(
        self, request_id: str, state: MossSpeechStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        if not state.codes:
            assert is_final
            raise RequestValidationError(
                "Generation ended before producing audio; increase max_new_tokens or retry"
            )
        else:
            pass
        flow = self.flow
        voice = state.voice
        decoder = self.codec.decoder
        assert flow is not None and voice is not None and decoder is not None
        device = self.codec.device
        prompt_codes = voice["prompt_token"].shape[1]
        committed_codes = state.decoded_codes + STREAM_CHUNK_CODES
        if is_final:
            committed_codes = available_codes = len(state.codes)
        else:
            if state.decoded_codes == 0:
                # note (Codex): Flow's chunk grid begins at the first voice prompt code.
                committed_codes += -prompt_codes % STREAM_CHUNK_CODES
            else:
                pass
            available_codes = committed_codes + flow.pre_lookahead_len
            if available_codes > len(state.codes):
                return None
            else:
                pass
        audio_codes = torch.tensor([state.codes[:available_codes]], device=device)
        devices = [device] if state.cuda_rng is not None else []
        with (
            torch.random.fork_rng(devices=devices),
            torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ),
        ):
            assert state.cpu_rng is not None
            torch.set_rng_state(state.cpu_rng)
            if state.cuda_rng is not None:
                torch.cuda.set_rng_state(state.cuda_rng, device)
            else:
                pass
            mel, _ = flow.inference(
                token=audio_codes,
                token_len=torch.tensor(
                    [available_codes], dtype=torch.int32, device=device
                ),
                prompt_token_len=torch.tensor(
                    [prompt_codes], dtype=torch.int32, device=device
                ),
                prompt_feat_len=torch.tensor(
                    [voice["prompt_feat"].shape[1]], dtype=torch.int32, device=device
                ),
                **voice,
                streaming=True,
                finalize=is_final,
            )
            # note (Codex): Flow attends to the prefix again, but HiFT only needs its overlap.
            mel = mel[..., state.decoded_codes * flow.token_mel_ratio :]
            if state.overlap_mel is None:
                source = torch.zeros(1, 1, 0)
            else:
                mel = torch.cat((state.overlap_mel, mel), dim=-1)
                source = state.overlap_source
                assert source is not None
            waveform, source = decoder.hift.inference(
                speech_feat=mel, cache_source=source
            )
            waveform = waveform.cpu().reshape(-1)
            state.cpu_rng = torch.get_rng_state()
            if state.cuda_rng is not None:
                state.cuda_rng = torch.cuda.get_rng_state(device)
            else:
                pass
        if state.overlap_waveform is not None:
            waveform[:HIFT_OVERLAP_SAMPLES] = (
                waveform[:HIFT_OVERLAP_SAMPLES]
                * self.fade_window[:HIFT_OVERLAP_SAMPLES]
                + state.overlap_waveform * self.fade_window[HIFT_OVERLAP_SAMPLES:]
            )
        else:
            pass
        if not is_final:
            state.overlap_mel = mel[..., -HIFT_OVERLAP_MEL_FRAMES:].clone()
            state.overlap_source = source[..., -HIFT_OVERLAP_SAMPLES:].clone()
            state.overlap_waveform = waveform[-HIFT_OVERLAP_SAMPLES:].clone()
            waveform = waveform[:-HIFT_OVERLAP_SAMPLES]
        else:
            pass
        assert waveform.numel() and torch.isfinite(waveform).all()
        state.decoded_codes = committed_codes
        state.emitted_samples += waveform.numel()
        expected_samples = (
            committed_codes * flow.token_mel_ratio * SAMPLES_PER_MEL_FRAME
        )
        assert state.emitted_samples == expected_samples - (
            0 if is_final else HIFT_OVERLAP_SAMPLES
        )
        return waveform

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: MossSpeechStreamState
    ) -> Mapping[str, object]:
        result = MossSpeechState.from_dict(payload.data)
        assert result.audio_codes == state.codes
        return {"finish_reason": result.finish_reason, "usage": result.usage()}

    def on_serving_stop(self) -> None:
        self.flow = None
        self.codec.close()
