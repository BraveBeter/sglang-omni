# SPDX-License-Identifier: Apache-2.0
"""Bounded B1 Flow streams with one owner for HiFT and session commits."""

import copy
import logging
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, replace

import torch

from sglang_omni.models.moss_speech.components.cfm_graph import MossSpeechCFMGraph
from sglang_omni.models.moss_speech.components.full_flow_precision import (
    FullFlowBF16Scope,
)
from sglang_omni.models.moss_speech.components.streaming_codec import (
    CodecStreamSession,
    MossSpeechStreamingCodec,
)
from sglang_omni.models.moss_speech.hf_config import MAX_CONTEXT_TOKENS
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.flow import (
    CausalMaskedDiffWithXvec,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class CodecMelResult:
    mel: torch.Tensor
    complete: torch.cuda.Event
    started: torch.cuda.Event


class CodecStepFailure(RuntimeError):
    def __init__(self, request_id: str, error: Exception) -> None:
        super().__init__(str(error))
        self.request_id = request_id


class CodecFlowSlot:
    def __init__(
        self,
        codec: MossSpeechStreamingCodec,
        flow: CausalMaskedDiffWithXvec,
        mel_length: int | None,
    ) -> None:
        self.codec = codec
        self.flow = flow
        self.stream = torch.cuda.Stream(device=codec.device)
        self.graph: MossSpeechCFMGraph | None = None
        self.precision_scope: FullFlowBF16Scope | None = None
        self.completions: int = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        try:
            self.executor.submit(self.prepare, mel_length).result()
        except BaseException:
            self.close()
            raise

    @torch.inference_mode()
    def prepare(self, mel_length: int | None) -> None:
        with torch.cuda.device(self.codec.device), torch.cuda.stream(self.stream):
            if (
                self.codec.flow_precision == "full_flow_bf16"
                and self.flow is not self.codec.flow
            ):
                self.precision_scope = FullFlowBF16Scope(self.flow)
                self.precision_scope.__enter__()
            else:
                pass
            if mel_length is not None:
                self.graph = MossSpeechCFMGraph(
                    self.flow.decoder, mel_length=mel_length
                )
            else:
                pass

    @torch.inference_mode()
    def execute(
        self,
        session: CodecStreamSession,
        available: int,
        final: bool,
        prepared: torch.cuda.Event,
    ) -> CodecMelResult:
        with torch.cuda.device(self.codec.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(prepared)
            started = torch.cuda.Event(enable_timing=True)
            complete = torch.cuda.Event(enable_timing=True)
            started.record()
            mel = self.codec.decode_flow(
                session, available, final=final, flow=self.flow
            )
            complete.record()
            complete.synchronize()
            return CodecMelResult(mel=mel, complete=complete, started=started)

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.stream.synchronize()
        if self.graph is not None:
            self.graph.close()
        else:
            pass
        if self.precision_scope is not None:
            self.precision_scope.__exit__(None, None, None)
        else:
            pass


@dataclass(kw_only=True)
class CodecFlowOperation:
    session: CodecStreamSession
    slot: CodecFlowSlot
    available: int
    final: bool
    hop: int
    future: Future[CodecMelResult]


class MossSpeechAsyncCodec:
    """Keep inputs ingestible while independent Flow slots execute one hop each."""

    def __init__(
        self,
        codec: MossSpeechStreamingCodec,
        *,
        flow_slots: int = 2,
        cfm_graph_mel_length: int | None = 1500,
        record_timing: bool = False,
    ) -> None:
        if (
            codec.device.type != "cuda"
            or type(flow_slots) is not int
            or flow_slots not in (1, 2)
        ):
            raise ValueError("Async Codec requires CUDA and one or two Flow slots")
        elif codec.cfm_graph is not None:
            raise ValueError("Flow graphs must be created by their execution slot")
        elif codec.sessions:
            raise ValueError("Create Flow workers before admitting codec sessions")
        elif codec.flow_precision not in (None, "full_flow_bf16"):
            raise ValueError("Async Codec supports FP32 or native full Flow BF16")
        elif codec.flow_precision is not None and cfm_graph_mel_length is not None:
            raise ValueError("BF16 Flow workers exclude the FP32 CFM graph")
        elif not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
            raise ValueError(
                "Freeze deterministic cuDNN policy before creating workers"
            )
        else:
            pass
        self.codec = codec
        self.owner_thread = threading.get_ident()
        self.pending: dict[str, CodecFlowOperation] = {}
        self.completed_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.record_timing = record_timing
        self.closed = False
        self.has_logged_activation = False
        codec.deactivate_precision()
        try:
            flows = [codec.flow] + [
                copy.deepcopy(codec.flow) for _ in range(flow_slots - 1)
            ]
        finally:
            codec.activate_precision()
        torch.cuda.current_stream(codec.device).synchronize()
        self.slots: list[CodecFlowSlot] = []
        try:
            for flow in flows:
                self.slots.append(CodecFlowSlot(codec, flow, cfm_graph_mel_length))
        except BaseException:
            self.close()
            raise

    def check_owner(self) -> None:
        if threading.get_ident() != self.owner_thread or self.closed:
            raise RuntimeError("Async Codec requires its active commit owner")
        else:
            pass

    def ingest(self, request_id: str, codes: list[int], *, final: bool = False) -> None:
        self.check_owner()
        session = self.codec.sessions[request_id]
        if (
            any(type(code) is not int or not 0 <= code < 16384 for code in codes)
            or len(session.codes) + len(codes) > MAX_CONTEXT_TOKENS
            or (session.final_requested and codes)
        ):
            self.cancel(request_id)
            raise ValueError("Invalid asynchronous Codec codes or final ordering")
        else:
            session.codes.extend(codes)
            session.final_requested = session.final_requested or final

    def submit(self, request_id: str) -> bool:
        self.check_owner()
        if request_id in self.pending:
            return False
        else:
            session = self.codec.sessions[request_id]
        busy = {id(operation.slot) for operation in self.pending.values()}
        slot = next((slot for slot in self.slots if id(slot) not in busy), None)
        hop = self.codec.next_hop(session)
        remaining = len(session.codes) - session.cursor
        if session.final_requested and not session.codes:
            self.codec.cleanup(request_id)
            return False
        elif slot is None or (
            not session.final_requested
            and remaining < hop + self.codec.flow.pre_lookahead_len
        ):
            return False
        else:
            final = (
                session.final_requested
                and remaining < hop + self.codec.flow.pre_lookahead_len
            )
            available = (
                len(session.codes)
                if final
                else session.cursor + hop + self.codec.flow.pre_lookahead_len
            )
            prepared = torch.cuda.Event()
            prepared.record(torch.cuda.current_stream(self.codec.device))
            snapshot = replace(session, codes=list(session.codes[:available]))
            future = slot.executor.submit(
                slot.execute, snapshot, available, final, prepared
            )
            self.pending[request_id] = CodecFlowOperation(
                session=session,
                slot=slot,
                available=available,
                final=final,
                hop=hop,
                future=future,
            )
            return True

    @torch.inference_mode()
    def collect(self, *, wait: bool = False) -> dict[str, torch.Tensor]:
        self.check_owner()
        waveforms: dict[str, torch.Tensor] = {}
        if wait and self.pending:
            wait_futures(
                [operation.future for operation in self.pending.values()],
                return_when=FIRST_COMPLETED,
            )
        else:
            pass
        for request_id, operation in list(self.pending.items()):
            if not operation.future.done():
                continue
            else:
                pass
            release_slot = False
            try:
                result = operation.future.result()
                if not wait and not result.complete.query():
                    continue
                else:
                    result.complete.synchronize()
                    release_slot = True
                if self.record_timing:
                    self.completed_events.append((result.started, result.complete))
                else:
                    pass
                if self.codec.sessions.get(request_id) is not operation.session:
                    continue
                else:
                    pass
                session = operation.session
                result.mel.record_stream(torch.cuda.current_stream(self.codec.device))
                with torch.random.fork_rng(devices=[self.codec.device.index or 0]):
                    torch.set_rng_state(session.cpu_rng)
                    torch.cuda.set_rng_state(session.device_rng, self.codec.device)
                    try:
                        waveform = self.codec.decode_mel(
                            session,
                            operation.available,
                            result.mel,
                            final=operation.final,
                        )
                    finally:
                        session.cpu_rng = torch.get_rng_state()
                        session.device_rng = torch.cuda.get_rng_state(self.codec.device)
                session.cursor = (
                    len(session.codes)
                    if operation.final
                    else session.cursor + operation.hop
                )
                if operation.final:
                    if (
                        session.emitted_samples
                        != len(session.codes) * self.codec.samples_per_code
                    ):
                        raise RuntimeError("Asynchronous Codec sample ledger mismatch")
                    else:
                        self.codec.cleanup(request_id)
                else:
                    pass
                waveforms[request_id] = waveform
                operation.slot.completions += 1
                if not self.has_logged_activation and all(
                    slot.completions for slot in self.slots
                ):
                    logger.info(
                        f"MOSS_CODEC_ASYNC_ACTIVE flow_slots={len(self.slots)} "
                        f"cudnn_enabled={torch.backends.cudnn.enabled} "
                        f"flow_precision={self.codec.flow_precision}"
                    )
                    self.has_logged_activation = True
                else:
                    pass
            except Exception as error:
                release_slot = True
                if self.codec.sessions.get(request_id) is operation.session:
                    self.codec.cleanup(request_id)
                    raise CodecStepFailure(request_id, error) from error
                else:
                    continue
            finally:
                if release_slot:
                    self.pending.pop(request_id)
                else:
                    pass
            break
        return waveforms

    def cancel(self, request_id: str) -> None:
        self.codec.cleanup(request_id)

    def close(self) -> None:
        if self.closed:
            return
        else:
            self.closed = True
        for slot in self.slots:
            slot.close()
        self.pending.clear()
        self.codec.sessions.clear()
