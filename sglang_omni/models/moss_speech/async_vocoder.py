# SPDX-License-Identifier: Apache-2.0
"""Asynchronous MOSS Codec submission and bounded completion pumping."""

import asyncio
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from sglang_omni.models.moss_speech.components.codec_adapter import (
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.codec_async import (
    CodecStepFailure,
    MossSpeechAsyncCodec,
)
from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.streaming import (
    AudioStreamState,
    MossSpeechStreamingVocoder,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage


@dataclass(kw_only=True)
class AsyncAudioStreamState(AudioStreamState):
    finish_requested: bool = False


class MossSpeechAsyncVocoder(MossSpeechStreamingVocoder):
    def __init__(
        self,
        codec: MossSpeechStreamingCodec,
        offline_codec: MossSpeechCodecAdapter,
        *,
        flow_slots: int,
        cfm_graph_mel_length: int | None,
    ) -> None:
        self.flow_slots = flow_slots
        self.cfm_graph_mel_length = cfm_graph_mel_length
        self.async_codec: MossSpeechAsyncCodec | None = None
        self.policy = ExitStack()
        self.last_submitted_request: str | None = None
        super().__init__(codec, offline_codec)
        self.can_batch_stream_chunks = True
        self.pump_on_chunk_batch = False
        self.stream_chunk_batch_max = 64

    def on_serving_start(self) -> None:
        self.policy.enter_context(
            torch.backends.cudnn.flags(benchmark=False, deterministic=True)
        )
        try:
            self.async_codec = MossSpeechAsyncCodec(
                self.codec,
                flow_slots=self.flow_slots,
                cfm_graph_mel_length=self.cfm_graph_mel_length,
            )
        except BaseException:
            self.policy.close()
            raise

    def create_stream_state(self, request_id: str) -> AsyncAudioStreamState:
        return AsyncAudioStreamState()

    def ingest(
        self, request_id: str, state: AudioStreamState, codes: torch.Tensor
    ) -> None:
        assert self.async_codec is not None
        tokens = codes.tolist()
        state.codes.extend(tokens)
        self.async_codec.ingest(request_id, tokens)

    def ready_requests(self) -> list[str]:
        assert self.async_codec is not None
        return [
            request_id
            for request_id, state in self.stream_states.items()
            if isinstance(state, AsyncAudioStreamState)
            and state.session is not None
            and not self.is_aborted(request_id)
            and request_id not in self.async_codec.pending
            and request_id in self.codec.sessions
            and (
                state.finish_requested
                or len(state.session.codes) - state.session.cursor
                >= self.codec.next_hop(state.session)
                + self.codec.flow.pre_lookahead_len
            )
        ]

    def has_ready_work(self) -> bool:
        with self.state_lock:
            return self.async_codec is not None and (
                any(
                    operation.future.done()
                    for operation in self.async_codec.pending.values()
                )
                or (
                    len(self.async_codec.pending) < self.flow_slots
                    and bool(self.ready_requests())
                )
            )

    def handle_message(
        self, message: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        if message.type == "wakeup":
            pass
        else:
            super().handle_message(message, loop)
        self.run_ready_step()

    def on_stream_done(self, request_id: str) -> None:
        assert self.async_codec is not None
        state = self.get_or_create_stream_state(request_id)
        assert isinstance(state, AsyncAudioStreamState)
        state.finish_requested = True
        self.async_codec.ingest(request_id, [], final=True)

    def run_ready_step(self) -> None:
        assert self.async_codec is not None
        failed: list[str] = []
        with self.state_lock:
            for _ in range(self.flow_slots):
                try:
                    waveforms = self.async_codec.collect()
                except CodecStepFailure as error:
                    self.emit_error(error.request_id, error)
                    self.abort_state(error.request_id)
                    failed.append(error.request_id)
                    continue
                for request_id, waveform in waveforms.items():
                    if self.is_aborted(request_id):
                        continue
                    else:
                        state = self.stream_states[request_id]
                    state.chunks += 1
                    state.samples += waveform.numel()
                    self.mark_stream_emitted(request_id)
                    self.outbox.put(self.stream_chunk_message(request_id, waveform))
            ready = self.ready_requests()
            if self.last_submitted_request in ready:
                index = (ready.index(self.last_submitted_request) + 1) % len(ready)
                ready = ready[index:] + ready[:index]
            else:
                pass
            for request_id in ready:
                if self.async_codec.submit(request_id):
                    self.last_submitted_request = request_id
                    self.async_codec.pending[request_id].future.add_done_callback(
                        lambda completed: self.inbox.put(
                            IncomingMessage(
                                request_id="", type="wakeup", data=completed
                            )
                        )
                    )
                else:
                    pass
            for request_id, state in list(self.stream_states.items()):
                if (
                    isinstance(state, AsyncAudioStreamState)
                    and state.finish_requested
                    and request_id not in self.codec.sessions
                    and not self.is_aborted(request_id)
                ):
                    messages = self.terminal_messages(
                        request_id, self.stream_payloads[request_id], state
                    )
                    self.complete_stream_request(request_id, messages)
                else:
                    pass
        for request_id in failed:
            self.cleanup_aborted_request(request_id)

    def release_stream_resources(
        self, request_id: str, state: AudioStreamState
    ) -> None:
        if self.async_codec is not None:
            self.async_codec.cancel(request_id)
        else:
            self.codec.cleanup(request_id)
        state.session = None

    def on_serving_stop(self) -> None:
        try:
            if self.async_codec is not None:
                self.async_codec.close()
            else:
                pass
        finally:
            try:
                super().on_serving_stop()
            finally:
                self.policy.close()
