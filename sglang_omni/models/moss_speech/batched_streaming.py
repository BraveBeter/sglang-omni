# SPDX-License-Identifier: Apache-2.0
"""Coalesce matching first hops already present in the vocoder inbox."""

import logging
from threading import get_ident
from typing import NamedTuple

import torch

from sglang_omni.models.moss_speech.first_flow_batch import (
    FirstFlowBatch,
    FirstFlowGraphUnavailable,
)
from sglang_omni.models.moss_speech.flow_history_graph import (
    StreamingPrivateHistoryGraphs,
)
from sglang_omni.models.moss_speech.flow_history_pages import (
    StreamingPagedHistoryGraphs,
)
from sglang_omni.models.moss_speech.flow_prefix_graph import StreamingReferenceGraphs
from sglang_omni.models.moss_speech.streaming import (
    MossSpeechStreamingVocoder,
    MossSpeechStreamState,
)
from sglang_omni.models.moss_speech.streaming_flow_graph import MIN_CAPTURE_FREE_BYTES
from sglang_omni.pipeline.stage.stream_queue import StreamItem

logger = logging.getLogger(__name__)


class ReadyHop(NamedTuple):
    request_id: str
    state: MossSpeechStreamState
    cursor: int
    committed: int
    arrived: int


class FirstHopPreparationError(RuntimeError):
    def __init__(self, hop: ReadyHop, error: Exception) -> None:
        super().__init__(str(error))
        self.hop = hop
        self.error = error


class FirstBatchStreamingVocoder(MossSpeechStreamingVocoder):
    can_batch_stream_chunks = True
    stream_chunk_batch_max = 256
    first_batch_size: int
    first_graph: FirstFlowBatch | None = None
    failed_reference: bytes | None = None

    @torch.inference_mode()
    def group_mels(self, hops: list[ReadyHop]) -> tuple[torch.Tensor, ...] | None:
        assert self.flow is not None
        if (
            self.reference_prefix is None
            or hops[0].state.reference_key == self.failed_reference
        ):
            return None
        else:
            pass
        if (
            self.page_streaming_history
            and self.first_graph is not None
            and self.first_graph.reference.reference_key != hops[0].state.reference_key
        ):
            self.first_graph = None
        else:
            pass
        device = self.codec.device
        conditions = []
        seeds = []
        # note (Codex): The encoder and HiFT each resume the request's own RNG stream.
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=False,
            deterministic=True,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        ):
            for hop in hops:
                state = hop.state
                try:
                    assert state.cpu_rng is not None and state.cuda_rng is not None
                    with torch.random.fork_rng(devices=[device]):
                        torch.set_rng_state(state.cpu_rng)
                        torch.cuda.set_rng_state(state.cuda_rng, device)
                        self.stream_mel(
                            state,
                            hop.committed + self.flow.pre_lookahead_len,
                            False,
                            conditions,
                        )
                        seeds.append(
                            (torch.get_rng_state(), torch.cuda.get_rng_state(device))
                        )
                except torch.OutOfMemoryError:
                    raise
                except Exception as exc:
                    raise FirstHopPreparationError(hop, exc) from exc
            prefix = self.reference_prefix.cached
            if prefix is None or prefix.reference_key != hops[0].state.reference_key:
                return None
            else:
                pass
            decoder = self.flow.decoder.decoder
            production = decoder.forward_estimator
            if isinstance(
                production, (StreamingPagedHistoryGraphs, StreamingPrivateHistoryGraphs)
            ):
                native = production.native
                if (
                    get_ident() != native.owner_thread
                    or torch.cuda.current_stream(device) != native.owner_stream
                ):
                    return None
                elif isinstance(production, StreamingPagedHistoryGraphs):
                    production.prepare_reference(prefix)
                else:
                    pass
                production = native
            else:
                pass
            assert isinstance(production, StreamingReferenceGraphs)
            if production.reference is not prefix:
                production.bind(prefix)
            else:
                pass
            if production.step_buffer is None or production.step_bank is None:
                return None
            else:
                pass
            if self.first_graph is None or self.first_graph.reference is not prefix:
                self.first_graph = None
                needed = (
                    production.step_buffer.numel()
                    * production.step_buffer.element_size()
                    * (production.steps + 1)
                    * len(hops)
                )
                if torch.cuda.mem_get_info(device)[0] < needed + MIN_CAPTURE_FREE_BYTES:
                    self.failed_reference = prefix.reference_key
                    return None
                else:
                    self.first_graph = FirstFlowBatch(production, len(hops))
            else:
                pass
            voice = hops[0].state.voice
            assert voice is not None
            result = self.first_graph.first_mels(
                decoder, conditions, voice["prompt_feat"].shape[1]
            )
        for hop, (cpu, cuda) in zip(hops, seeds, strict=True):
            hop.state.cpu_rng, hop.state.cuda_rng = cpu, cuda
        logger.info("Decoded %d ready first Flow rows together", len(hops))
        return result

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        failed = []
        with self.state_lock:
            ready = []
            cursors = {}
            for rid, item in items:
                if self.is_aborted(rid):
                    continue
                else:
                    pass
                try:
                    state = self.ingest_stream_item(rid, item)
                    if state is None:
                        continue
                    else:
                        pass
                    cursor = cursors.get(rid, state.decoded_codes)
                    arrived = len(state.codes)
                    committed = self.next_committed_codes(state, cursor, arrived)
                    assert self.flow is not None
                    if arrived >= committed + self.flow.pre_lookahead_len:
                        ready.append(ReadyHop(rid, state, cursor, committed, arrived))
                        cursors[rid] = committed
                    else:
                        pass
                except Exception as exc:
                    self.emit_error(rid, exc)
                    self.abort_state(rid)
                    failed.append(rid)
            while ready:
                leader = ready.pop(0)
                if self.is_aborted(leader.request_id):
                    continue
                else:
                    pass
                group = [leader]
                if leader.cursor == 0:
                    partners = [
                        h
                        for h in ready
                        if h.cursor == 0
                        and h.committed == leader.committed
                        and h.state.reference_key == leader.state.reference_key
                        and not self.is_aborted(h.request_id)
                    ]
                    if len(partners) >= self.first_batch_size - 1:
                        group.extend(partners[: self.first_batch_size - 1])
                        selected = {h.request_id for h in group}
                        ready = [
                            h
                            for h in ready
                            if h.cursor != 0 or h.request_id not in selected
                        ]
                    else:
                        pass
                else:
                    pass
                mels = None
                if len(group) > 1:
                    try:
                        mels = self.group_mels(group)
                    except (torch.OutOfMemoryError, FirstFlowGraphUnavailable):
                        self.first_graph = None
                        self.failed_reference = leader.state.reference_key
                        logger.warning(
                            "Using single-row Flow after a first-hop batch allocation or capture failure"
                        )
                    except FirstHopPreparationError as exc:
                        # note (Codex): No row has committed RNG or audio yet. Surviving rows can run alone.
                        failed.extend(self.fail_live_hops([exc.hop], exc.error))
                    except Exception as exc:
                        failed.extend(self.fail_live_hops(group, exc))
                        continue
                else:
                    pass
                for index, hop in enumerate(group):
                    rid = hop.request_id
                    if self.is_aborted(rid):
                        continue
                    else:
                        pass
                    try:
                        wave = self.decode_delta(
                            rid,
                            hop.state,
                            is_final=False,
                            arrived_codes=hop.arrived,
                            prepared_mel=None if mels is None else mels[index],
                        )
                        assert wave is not None
                        if self.is_aborted(rid):
                            continue
                        else:
                            pass
                        message = self.stream_chunk_message(rid, wave)
                        if not self.is_aborted(rid):
                            self.mark_stream_emitted(rid)
                            self.outbox.put(message)
                        else:
                            pass
                    except Exception as exc:
                        failed.extend(self.fail_live_hops([hop], exc))
        for rid in failed:
            self.cleanup_aborted_request(rid)

    def fail_live_hops(self, hops: list[ReadyHop], error: Exception) -> list[str]:
        participants = [
            (h.request_id, h.state) for h in hops if not self.is_aborted(h.request_id)
        ]
        return self.on_step_failure(participants, error) if participants else []

    def on_serving_stop(self) -> None:
        self.first_graph = None
        self.failed_reference = None
        super().on_serving_stop()
