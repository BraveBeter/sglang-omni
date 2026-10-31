# SPDX-License-Identifier: Apache-2.0
"""MOSS-Speech token streams and eager text and audio terminals."""

import random
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.moss_speech.components.codec_adapter import (
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.streaming_codec import (
    CodecStreamSession,
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.components.voice import VoiceConditioning
from sglang_omni.models.moss_speech.hf_config import MAX_CONTEXT_TOKENS
from sglang_omni.models.moss_speech.payload_types import (
    EOSP_TOKEN_ID,
    MODALITY_PAD_TOKEN_ID,
)
from sglang_omni.models.moss_speech.request_builders import (
    extract_output_codes,
    resolve_output_terminal,
    state_from_payload,
    terminal_result,
)
from sglang_omni.models.moss_speech.request_data import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.voice_transport import (
    state_voice_payload,
    unpack_voice_payload,
    validate_voice_payload,
    voice_payload_digest,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.types import RequestOutput
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import scoped_global_rng


def resolve_stream_terminal(request_id: str, payload: StagePayload) -> list[str]:
    if payload.request.params.get("stream", False):
        return [resolve_output_terminal(request_id, payload)]
    else:
        return []


class MossSpeechStreamOutputBuilder:
    def __call__(
        self,
        request_id: str,
        request_data: MossSpeechSGLangRequestData,
        request_output: RequestOutput,
    ) -> list[OutgoingMessage]:
        return self.flush(request_id, request_data)

    def flush(
        self, request_id: str, request_data: MossSpeechSGLangRequestData
    ) -> list[OutgoingMessage]:
        payload = request_data.stage_payload
        assert payload is not None
        if not payload.request.params.get("stream", False):
            return []
        elif request_data.req.inflight_middle_chunks:
            return []
        else:
            pass
        state = state_from_payload(payload)
        audio_output = state.output_modality == "audio"
        rows = request_data.output_rows[request_data.stream_grid_cursor :]
        request_data.stream_grid_cursor = len(request_data.output_rows)
        tokens: list[int] = []
        for text_token, audio_token in rows:
            if not audio_output:
                tokens.append(int(text_token))
            elif request_data.stream_audio_ended or text_token != MODALITY_PAD_TOKEN_ID:
                continue
            elif audio_token == EOSP_TOKEN_ID:
                request_data.stream_audio_ended = True
            elif 0 <= audio_token < EOSP_TOKEN_ID:
                tokens.append(int(audio_token))
            else:
                raise ValueError(
                    "Generated streaming code is outside the audio vocabulary"
                )
        if not tokens:
            return []
        else:
            pass
        cursor = request_data.stream_token_cursor
        metadata: dict[str, object] = {
            "stream": True,
            "modality": "audio_codes" if audio_output else "text_tokens",
            "row_index": cursor,
        }
        if audio_output and cursor == 0:
            metadata.update(
                seed=state.effective_seed,
                voice_key=state.voice_key,
                voice_payload=state_voice_payload(state),
            )
        else:
            pass
        request_data.stream_token_cursor += len(tokens)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target="audio_vocoder" if audio_output else "text_decode",
                data=torch.tensor(tokens, dtype=torch.int64),
                metadata=metadata,
            )
        ]


def make_stream_output_builder() -> MossSpeechStreamOutputBuilder:
    return MossSpeechStreamOutputBuilder()


@dataclass(kw_only=True)
class AudioStreamState:
    codes: list[int] = field(default_factory=list)
    pending: list[int] = field(default_factory=list)
    voice_key: str | None = None
    voice_digest: str | None = None
    seed: int | None = None
    session: CodecStreamSession | None = None
    chunks: int = 0
    samples: int = 0


class MossSpeechStreamingVocoder(StreamingVocoderBase[AudioStreamState, None]):
    def __init__(
        self, codec: MossSpeechStreamingCodec, offline_codec: MossSpeechCodecAdapter
    ) -> None:
        self.codec = codec
        self.offline_codec = offline_codec
        super().__init__(
            self.compute, sample_rate=24000, stream_source_hint="MOSS-Speech"
        )

    def compute(self, payload: StagePayload) -> StagePayload:
        state = state_from_payload(payload)
        voice = VoiceConditioning(
            **unpack_voice_payload(state_voice_payload(state)), meta={"sr": 24000}
        )
        with (
            scoped_global_rng(),
            torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ),
        ):
            random.seed(state.effective_seed)
            np.random.seed(state.effective_seed % (1 << 32))
            torch.manual_seed(state.effective_seed)
            sample_rate, waveform = self.offline_codec.decode(
                extract_output_codes(state), voice, request_id=payload.request_id
            )
        state.audio_samples = waveform.tolist()
        state.audio_sample_rate = sample_rate
        return StagePayload(payload.request_id, payload.request, terminal_result(state))

    def create_stream_state(self, request_id: str) -> AudioStreamState:
        return AudioStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: AudioStreamState,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        if isinstance(source, StagePayload):
            payload_state = state_from_payload(source)
            voice_key = payload_state.voice_key
            seed = payload_state.effective_seed
            packed_voice = state_voice_payload(payload_state)
        else:
            if source.get("stream") is not True or source.get("row_index") != len(
                state.codes
            ):
                raise ValueError(
                    "Noncontiguous audio stream cursor or invalid stream flag"
                )
            else:
                pass
            voice_key = source.get("voice_key")
            seed = source.get("seed")
            raw_voice = source.get("voice_payload")
            if state.session is not None and all(
                value is None for value in (voice_key, seed, raw_voice)
            ):
                return
            else:
                packed_voice = validate_voice_payload(raw_voice)
        if not isinstance(voice_key, str) or type(seed) is not int:
            raise ValueError(
                "The first audio chunk requires a voice key and integer seed"
            )
        else:
            pass
        digest = voice_payload_digest(packed_voice)
        if state.session is not None:
            if (voice_key, seed, digest) != (
                state.voice_key,
                state.seed,
                state.voice_digest,
            ):
                raise ValueError("Streaming voice or seed changed within a request")
            else:
                return
        else:
            self.codec.begin(request_id, unpack_voice_payload(packed_voice), seed=seed)
            state.voice_key, state.seed, state.voice_digest = voice_key, seed, digest
            state.session = self.codec.sessions[request_id]

    def validate_chunk(
        self, request_id: str, state: AudioStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        if (
            codes.ndim != 1
            or codes.dtype not in (torch.int32, torch.int64)
            or not codes.numel()
            or ((codes < 0) | (codes >= EOSP_TOKEN_ID)).any()
            or len(state.codes) + codes.numel() > MAX_CONTEXT_TOKENS
        ):
            raise ValueError(
                "Audio chunk must contain valid codes within the generation budget"
            )
        else:
            return codes

    def ingest(
        self, request_id: str, state: AudioStreamState, codes: torch.Tensor
    ) -> None:
        state.codes.extend(codes.tolist())
        state.pending.extend(codes.tolist())

    def decode_delta(
        self, request_id: str, state: AudioStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        waveforms = self.codec.push(request_id, state.pending, final=is_final)
        state.pending.clear()
        if waveforms:
            state.chunks += len(waveforms)
            state.samples += sum(waveform.numel() for waveform in waveforms)
            return torch.cat(waveforms)
        else:
            return None

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: AudioStreamState
    ) -> Mapping[str, object]:
        final_state = state_from_payload(payload)
        if extract_output_codes(final_state) != state.codes:
            raise RuntimeError(
                "Streamed audio codes disagree with the terminal AR grid"
            )
        else:
            pass
        result = terminal_result(final_state)
        result.pop("audio_data", None)
        assert state.session is not None
        result.update(
            stream_chunks=state.chunks,
            stream_samples=state.samples,
            stream_codec_ledger=list(state.session.ledger),
        )
        return result

    def release_stream_resources(
        self, request_id: str, state: AudioStreamState
    ) -> None:
        self.codec.cleanup(request_id)
        state.session = None

    def on_serving_stop(self) -> None:
        self.codec.close()
        self.offline_codec.close()


@dataclass(kw_only=True)
class TextStreamState:
    tokens: list[int] = field(default_factory=list)
    emitted: str = ""


class MossSpeechStreamingText(StreamingSimpleScheduler):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self.states: dict[str, TextStreamState] = {}
        self.completed: dict[str, None] = {}
        super().__init__(self.compute)

    def compute(self, payload: StagePayload) -> StagePayload:
        state = state_from_payload(payload)
        if not state.output_grid:
            raise ValueError("Text decoding received an empty output grid")
        else:
            state.generated_text = (
                self.tokenizer.decode(
                    [row[0] for row in state.output_grid], skip_special_tokens=True
                )
                .replace("<|empty|>", ".")
                .replace("<|end_empty|>", ":")
                .rstrip("\ufffd")
            )
        return StagePayload(payload.request_id, payload.request, terminal_result(state))

    def start(self) -> None:
        try:
            super().start()
        finally:
            self.stop()

    def stop(self) -> None:
        super().stop()
        with self.state_lock:
            self.states.clear()
            self.completed.clear()
            self.stream_payloads.clear()
            self.pending_done.clear()

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return bool(payload.request.params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        self.completed.pop(request_id, None)

    def handle_stream_done(self, request_id: str) -> None:
        with self.state_lock:
            if request_id in self.completed:
                return
            else:
                super().handle_stream_done(request_id)

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if self.is_aborted(request_id) or request_id in self.completed:
            return []
        else:
            pass
        state = self.states.setdefault(request_id, TextStreamState())
        metadata = item.metadata or {}
        if (
            metadata.get("stream") is not True
            or metadata.get("modality") != "text_tokens"
            or metadata.get("row_index") != len(state.tokens)
        ):
            raise ValueError("Invalid text stream cursor or modality")
        else:
            pass
        tokens = item.data
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 1
            or tokens.dtype not in (torch.int32, torch.int64)
            or ((tokens < 0) | (tokens >= 151680)).any()
            or len(state.tokens) + tokens.numel() > MAX_CONTEXT_TOKENS
        ):
            raise ValueError("Text chunk contains invalid tokens or exceeds the budget")
        else:
            state.tokens.extend(tokens.tolist())
        decoded = (
            self.tokenizer.decode(state.tokens, skip_special_tokens=True)
            .replace("<|empty|>", ".")
            .replace("<|end_empty|>", ":")
        )
        if decoded.endswith("\ufffd"):
            return []
        elif not decoded.startswith(state.emitted):
            raise RuntimeError("Tokenizer revised an already emitted text prefix")
        else:
            delta = decoded[len(state.emitted) :]
            state.emitted = decoded
        if delta:
            return [
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={"text": delta},
                    metadata={"modality": "text"},
                )
            ]
        else:
            return []

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self.stream_payloads[request_id]
        final_state = state_from_payload(payload)
        stream_state = self.states.get(request_id, TextStreamState())
        if [row[0] for row in final_state.output_grid] != stream_state.tokens:
            raise RuntimeError(
                "Streamed text tokens disagree with the terminal AR grid"
            )
        else:
            pass
        result = self.compute(payload)
        assert isinstance(result.data, dict)
        final_text = result.data["text"]
        assert isinstance(final_text, str)
        if not final_text.startswith(stream_state.emitted):
            raise RuntimeError("Final text disagrees with the emitted prefix")
        else:
            pass
        self.completed[request_id] = None
        if len(self.completed) > 10000:
            for completed_request_id in list(self.completed)[:5000]:
                self.completed.pop(completed_request_id)
        else:
            pass
        return [OutgoingMessage(request_id=request_id, type="result", data=result)]

    def clear_stream_state(self, request_id: str) -> None:
        self.states.pop(request_id, None)
