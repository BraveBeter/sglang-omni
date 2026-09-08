# SPDX-License-Identifier: Apache-2.0
"""Model-specific stream adapters and terminal schedulers."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase

from .payload_types import EOSP_TOKEN_ID, MODALITY_PAD_TOKEN_ID
from .request_builders import resolve_output_terminal


def resolve_stream_terminal(request_id: str, payload: StagePayload) -> list[str]:
    if not payload.request.params.get("stream", False):
        return []
    return [resolve_output_terminal(request_id, payload)]


def make_stream_output_builder() -> Any:
    def build(request_id: str, data: Any, req_output: Any) -> list[OutgoingMessage]:
        del req_output
        payload = data.stage_payload
        if not payload.request.params.get("stream", False):
            return []
        if getattr(data.req, "inflight_middle_chunks", 0):
            return []
        state = payload.data
        if not isinstance(state, dict):
            state = state.to_dict()
        audio = state["output_modality"] == "audio"
        start = getattr(data, "stream_grid_cursor", 0)
        rows = data.output_rows[start:]
        data.stream_grid_cursor = len(data.output_rows)
        tokens = []
        for text, code in rows:
            if not audio:
                tokens.append(int(text))
            elif (
                not getattr(data, "stream_audio_ended", False)
                and int(text) == MODALITY_PAD_TOKEN_ID
            ):
                if int(code) == EOSP_TOKEN_ID:
                    data.stream_audio_ended = True
                elif 0 <= int(code) < EOSP_TOKEN_ID:
                    tokens.append(int(code))
                else:
                    raise ValueError("Invalid generated streaming code")
        if not tokens:
            return []
        cursor = getattr(data, "stream_token_cursor", 0)
        metadata = {
            "stream": True,
            "modality": "audio_codes" if audio else "text_tokens",
            "row_index": cursor,
        }
        if audio and cursor == 0:
            metadata.update(
                seed=state["effective_seed"],
                voice_key=state["voice_key"],
                voice={
                    "prompt_token": [state["voice_token_ids"]],
                    "prompt_feat": [state["voice_feat"]],
                    "embedding": [state["voice_embedding"]],
                },
            )
        data.stream_token_cursor = cursor + len(tokens)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target="audio_vocoder" if audio else "text_decode",
                data=torch.tensor(tokens, dtype=torch.int64),
                metadata=metadata,
            )
        ]

    build.flush = lambda rid, data: build(rid, data, None)
    return build


@dataclass
class AudioStreamState:
    codes: list[int] = field(default_factory=list)
    voice_key: str | None = None
    voice_digest: str | None = None
    seed: int | None = None
    session: Any = None
    pending: list[int] = field(default_factory=list)
    chunks: int = 0
    samples: int = 0


class MossSpeechStreamingVocoder(StreamingVocoderBase[AudioStreamState, None]):
    def __init__(self, codec: Any, offline_compute: Any) -> None:
        self.codec = codec
        super().__init__(
            offline_compute, sample_rate=24000, stream_source_hint="MOSS-Speech"
        )

    def create_stream_state(self, request_id: str) -> AudioStreamState:
        return AudioStreamState()

    def latch_stream_contract(
        self, request_id: str, state: AudioStreamState, source: Any, *, origin: str
    ) -> None:
        if origin == "payload":
            from .stages import _state_from_payload

            payload_state = _state_from_payload(source)
            key, seed = payload_state.voice_key, payload_state.effective_seed
            voice = {
                "prompt_token": [payload_state.voice_token_ids],
                "prompt_feat": [payload_state.voice_feat],
                "embedding": [payload_state.voice_embedding],
            }
        else:
            if source.get("stream") is not True or source.get("row_index") != len(
                state.codes
            ):
                raise ValueError(
                    "Noncontiguous streaming code cursor or invalid stream flag"
                )
            key, seed, voice = (
                source.get("voice_key"),
                source.get("seed"),
                source.get("voice"),
            )
            if (
                state.session is not None
                and key is None
                and seed is None
                and voice is None
            ):
                return
        digest = hashlib.sha256(json.dumps(voice, sort_keys=True).encode()).hexdigest()
        if state.session is not None:
            if (
                key != state.voice_key
                or seed != state.seed
                or digest != state.voice_digest
            ):
                raise ValueError("Streaming voice/seed contract changed")
            return
        if key is None or seed is None or not voice:
            raise ValueError("First audio chunk requires voice and seed")
        converted = {
            k: torch.as_tensor(
                v, dtype=torch.int32 if k == "prompt_token" else torch.float32
            )
            for k, v in voice.items()
        }
        self.codec.begin(request_id, converted, seed=int(seed))
        state.voice_digest = digest
        state.voice_key, state.seed, state.session = (
            key,
            int(seed),
            self.codec.sessions[request_id],
        )

    def validate_chunk(
        self, request_id: str, state: AudioStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        if (
            codes.ndim != 1
            or codes.dtype not in (torch.int32, torch.int64)
            or not codes.numel()
        ):
            raise ValueError("Audio code chunk must be a nonempty integer vector")
        if ((codes < 0) | (codes >= EOSP_TOKEN_ID)).any() or len(
            state.codes
        ) + codes.numel() > 512:
            raise ValueError("Audio code chunk exceeds vocabulary or stream budget")
        return codes

    def ingest(
        self, request_id: str, state: AudioStreamState, codes: torch.Tensor
    ) -> None:
        values = codes.tolist()
        state.codes.extend(values)
        state.pending.extend(values)

    def decode_delta(
        self, request_id: str, state: AudioStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        waves = self.codec.push(request_id, state.pending, final=is_final)
        state.pending = []
        if not waves:
            return None
        state.chunks += len(waves)
        state.samples += sum(w.numel() for w in waves)
        return torch.cat(waves)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: AudioStreamState
    ) -> dict[str, Any]:
        from .stages import _state_from_payload, extract_output_codes, terminal_result

        final = _state_from_payload(payload)
        if extract_output_codes(final) != state.codes:
            raise RuntimeError("Streamed codes do not match terminal AR grid")
        data = terminal_result(final)
        # Stream deltas already contain the audio. Never replay an aggregate tail.
        data.pop("audio_data", None)
        data.update(
            stream_chunks=state.chunks,
            stream_samples=state.samples,
            stream_codec_ledger=list(state.session.ledger),
        )
        return data

    def release_stream_resources(
        self, request_id: str, state: AudioStreamState
    ) -> None:
        self.codec.cleanup(request_id)
        state.session = None
        metadata = {
            "session_active": request_id in self.codec.sessions,
            "active_sessions": len(self.codec.sessions),
        }
        if self.codec.device.type == "cuda":
            metadata.update(
                allocated_bytes=torch.cuda.memory_allocated(self.codec.device),
                reserved_bytes=torch.cuda.memory_reserved(self.codec.device),
            )
        _emit_event(
            request_id=request_id,
            stage="audio_vocoder",
            event_name="moss_codec_session_released",
            metadata=metadata,
        )

    def on_serving_stop(self) -> None:
        self.codec.close()


@dataclass
class TextStreamState:
    tokens: list[int] = field(default_factory=list)
    emitted: str = ""


class MossSpeechStreamingText(StreamingSimpleScheduler):
    def __init__(self, tokenizer: Any, offline_compute: Any) -> None:
        self.tokenizer = tokenizer
        self.states: dict[str, TextStreamState] = {}
        self.completed: dict[str, None] = {}
        super().__init__(offline_compute)

    def start(self) -> None:
        try:
            super().start()
        finally:
            self.stop()

    def stop(self) -> None:
        super().stop()
        with self._state_lock:
            self.states.clear()
            self.completed.clear()
            self._stream_payloads.clear()
            self._pending_done.clear()

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return bool(payload.request.params.get("stream", False))

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        if self._is_aborted(request_id) or request_id in self.completed:
            return []
        state = self.states.setdefault(request_id, TextStreamState())
        metadata = item.metadata or {}
        if (
            metadata.get("stream") is not True
            or metadata.get("modality") != "text_tokens"
            or metadata.get("row_index") != len(state.tokens)
        ):
            raise ValueError("Invalid text stream cursor or modality")
        tokens = item.data
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 1
            or tokens.dtype not in (torch.int32, torch.int64)
            or ((tokens < 0) | (tokens >= 151680)).any()
        ):
            raise ValueError("Invalid text token chunk")
        state.tokens.extend(tokens.tolist())
        if len(state.tokens) > 512:
            raise ValueError("Text stream exceeds generation budget")
        decoded = (
            self.tokenizer.decode(state.tokens, skip_special_tokens=True)
            .replace("<|empty|>", ".")
            .replace("<|end_empty|>", ":")
        )
        if decoded.endswith("\ufffd"):
            return []
        if not decoded.startswith(state.emitted):
            raise RuntimeError("Tokenizer revised already emitted text")
        delta = decoded[len(state.emitted) :]
        state.emitted = decoded
        return (
            [
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={"text": delta},
                    metadata={"modality": "text"},
                )
            ]
            if delta
            else []
        )

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self._stream_payloads[request_id]
        result = self._fn(payload)
        data = result.data
        state = self.states.get(request_id, TextStreamState())
        if [row[0] for row in data["output_grid"]] != state.tokens:
            raise RuntimeError("Streamed text tokens do not match terminal AR grid")
        # An output-length stop may leave an incomplete UTF-8 byte sequence.
        # Preserve usage/grid; only emit the complete Unicode prefix.
        text = data["text"].rstrip("\ufffd")
        if not text.startswith(state.emitted):
            raise RuntimeError("Final text disagrees with emitted prefix")
        data["text"] = data["generated_text"] = text
        self.completed[request_id] = None
        if len(self.completed) > 10000:
            for rid in list(self.completed)[:5000]:
                self.completed.pop(rid, None)
        return [OutgoingMessage(request_id=request_id, type="result", data=result)]

    def clear_stream_state(self, request_id: str) -> None:
        self.states.pop(request_id, None)


def create_streaming_text_executor(model_path: str) -> MossSpeechStreamingText:
    from .stages import _load_tokenizer, create_text_decode_executor

    offline = create_text_decode_executor(model_path)
    return MossSpeechStreamingText(_load_tokenizer(model_path), offline._fn)


def create_streaming_audio_executor(
    model_path: str,
    *,
    codec_path: str | None = None,
    gpu_id: int | None = None,
    chunk_size: int = 5,
) -> MossSpeechStreamingVocoder:
    from .components.streaming_codec import MossSpeechStreamingCodec
    from .stages import _resolve_codec_dir, create_audio_vocoder_executor

    codec_dir = _resolve_codec_dir(model_path, codec_path)
    offline = create_audio_vocoder_executor(
        model_path, codec_path=codec_dir, gpu_id=gpu_id
    )
    codec = MossSpeechStreamingCodec(
        codec_dir,
        chunk_size=chunk_size,
        device=f"cuda:{gpu_id}" if gpu_id is not None else "cuda",
    )
    return MossSpeechStreamingVocoder(codec, offline._fn)
