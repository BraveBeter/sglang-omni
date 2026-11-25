# SPDX-License-Identifier: Apache-2.0
"""Admission order for a burst of already queued streaming code packets."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from contextvars import ContextVar
from queue import SimpleQueue
from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_streaming import TestFlow, first_chunk_vocoder

from sglang_omni.models.moss_speech import stages
from sglang_omni.models.moss_speech.batched_streaming import FirstBatchStreamingVocoder
from sglang_omni.models.moss_speech.flow_history_pages import (
    PagedHistory,
    PagedHistoryPrefill,
)
from sglang_omni.models.moss_speech.streaming import MossSpeechStreamState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.message import IncomingMessage


@pytest.mark.parametrize("steps", [None, 2])
def test_serving_thread_loads_only_the_configured_single_grid(
    steps: int | None,
) -> None:
    worker = object.__new__(FirstBatchStreamingVocoder)
    worker.first_hop_flow_steps = steps
    worker.first_flow = None
    single, prefix = Mock(), Mock()
    worker.load_streaming_flow = Mock(return_value=(single, prefix))
    worker.warm_up_serving_thread()
    worker.warm_up_serving_thread()
    if steps is None:
        worker.load_streaming_flow.assert_not_called()
    else:
        worker.load_streaming_flow.assert_called_once_with(2)
        assert worker.first_flow is single and worker.first_reference_prefix is prefix


def test_already_queued_packets_preserve_first_and_refill_boundaries() -> None:
    worker = object.__new__(FirstBatchStreamingVocoder)
    worker.state_lock = RLock()
    worker.first_batch_size = 2
    worker.streaming_chunks_per_decode = 3
    worker.emit_first_closed_chunk = True
    worker.flow = SimpleNamespace(pre_lookahead_len=3, token_mel_ratio=4)
    worker.is_aborted = Mock(return_value=False)
    worker.outbox = SimpleQueue()
    worker.mark_stream_emitted = Mock()
    worker.stream_chunk_message = lambda rid, pcm: (rid, pcm.item())
    states = {
        r: MossSpeechStreamState(
            voice={"prompt_token": torch.zeros(1, 342, dtype=torch.long)},
            reference_key=b"shared",
        )
        for r in ["a", "b", "c"]
    }

    def ingest(rid, item):
        states[rid].codes.append(item)
        return states[rid]

    worker.ingest_stream_item = ingest
    worker.group_mels = Mock(return_value=(torch.tensor(1), torch.tensor(2)))
    seen = []

    def decode(rid, state, *, is_final, arrived_codes, prepared_mel):
        committed = worker.next_committed_codes(
            state, state.decoded_codes, arrived_codes
        )
        seen.append((rid, arrived_codes, committed, prepared_mel is not None))
        state.decoded_codes = committed
        return torch.tensor(committed)

    worker.decode_delta = decode
    packets = [("a", 0)] * 36 + [("b", 0)] * 11 + [("c", 0)] * 11
    worker.on_stream_chunk_batch(packets)
    assert seen == [
        ("a", 11, 8, True),
        ("b", 11, 8, True),
        ("a", 36, 33, False),
        ("c", 11, 8, False),
    ]
    assert worker.group_mels.call_count == 1


@pytest.fixture
def first_hop_worker(monkeypatch):
    codec = Mock(device=torch.device("cuda:0"), flow_dtype=torch.bfloat16)
    monkeypatch.setattr(stages, "MossSpeechCodec", Mock(return_value=codec))
    monkeypatch.setattr(stages.AutoTokenizer, "from_pretrained", Mock())
    worker = stages.create_audio_vocoder_executor(
        "model",
        codec_path="codec",
        device="cpu",
        flow_dtype="bfloat16",
        streaming_flow_steps=3,
        prefill_streaming_reference=True,
        emit_first_closed_chunk=True,
        first_flow_batch_size=2,
    )
    assert isinstance(worker, FirstBatchStreamingVocoder)
    worker.flow = SimpleNamespace(pre_lookahead_len=3, token_mel_ratio=4)
    for rid in ("a", "b", "c"):
        state = worker.get_or_create_stream_state(rid)
        state.voice = {"prompt_token": torch.zeros(1, 342, dtype=torch.long)}
        state.reference_key = b"shared"
    worker.group_mels = Mock(return_value=(torch.ones(1), torch.ones(1)))
    worker.abort_callback = Mock()
    return worker


def send_codes(worker, *requests):
    group = [
        IncomingMessage(
            rid,
            "stream_chunk",
            StreamItem(
                0, torch.zeros(11, dtype=torch.long), "ar_engine", {"stream": True}
            ),
        )
        for rid in requests
    ]
    worker.handle_stream_chunk_batch(group)

    messages = []
    while not worker.outbox.empty():
        message = worker.outbox.get_nowait()
        messages.append((message.request_id, message.type))
    return messages


@pytest.mark.parametrize("stage", ["decode", "publish"])
@pytest.mark.parametrize("failed", ["a", "b"])
def test_first_hop_row_failure_preserves_healthy_request(
    first_hop_worker, stage, failed
):
    worker = first_hop_worker
    publish = worker.stream_chunk_message

    def decode(rid, state, **kwargs):
        if stage == "decode" and rid == failed:
            raise ValueError("failed private HiFT")
        else:
            pass
        state.decoded_codes = 8
        return torch.ones(1)

    def message(rid, wave):
        if stage == "publish" and rid == failed:
            raise ValueError("failed private PCM packaging")
        else:
            pass
        return publish(rid, wave)

    worker.decode_delta = decode
    worker.stream_chunk_message = message
    assert send_codes(worker, "a", "b") == [
        (r, "error" if r == failed else "stream") for r in ("a", "b")
    ]
    healthy = "a" if failed == "b" else "b"
    assert worker.is_aborted(failed) and not worker.is_aborted(healthy)
    assert (
        failed not in worker.stream_states
        and worker.stream_states[healthy].decoded_codes == 8
    )
    assert worker.emitted_stream_ids == {healthy}
    worker.abort_callback.assert_called_once_with(failed)
    assert send_codes(worker, failed) == []


def test_first_hop_shared_failure_aborts_only_batch_participants(first_hop_worker):
    worker = first_hop_worker
    worker.group_mels.side_effect = RuntimeError("failed shared Flow")
    worker.decode_delta = Mock(return_value=torch.ones(1))
    assert send_codes(worker, "a", "b", "c") == [
        ("a", "error"),
        ("b", "error"),
        ("c", "stream"),
    ]
    assert list(worker.stream_states) == ["c"] and worker.emitted_stream_ids == {"c"}
    assert [args.args[0] for args in worker.abort_callback.call_args_list] == ["a", "b"]
    assert worker.decode_delta.call_count == 1


@pytest.mark.parametrize("during", ["flow", "decode", "package"])
def test_concurrent_abort_preserves_batch_partner(
    first_hop_worker, during, monkeypatch
):
    worker = first_hop_worker
    entered = Event()
    marked = Event()
    clear = worker.clear_request_state

    def clear_after_mark(rid, *, keep_aborted=False):
        assert worker.is_aborted(rid)
        marked.set()
        clear(rid, keep_aborted=keep_aborted)

    monkeypatch.setattr(worker, "clear_request_state", clear_after_mark)

    def cancel():
        assert entered.wait(timeout=5)
        worker.abort("a")

    def wait_for_cancel():
        entered.set()
        assert marked.wait(timeout=5)

    if during == "flow":

        def shared(hops):
            wait_for_cancel()
            return (torch.ones(1), torch.ones(1))

        worker.group_mels.side_effect = shared
    else:
        pass
    decoded = []

    def decode(rid, state, **kwargs):
        decoded.append(rid)
        if during == "decode" and rid == "a":
            wait_for_cancel()
        else:
            pass
        return torch.ones(1)

    publish = worker.stream_chunk_message

    def package(rid, wave):
        if during == "package" and rid == "a":
            wait_for_cancel()
        else:
            pass
        return publish(rid, wave)

    worker.decode_delta = decode
    worker.stream_chunk_message = package
    with ThreadPoolExecutor(max_workers=1) as pool:
        aborted = pool.submit(cancel)
        output = send_codes(worker, "a", "b")
        aborted.result(timeout=5)
    assert output == [("b", "stream")] and worker.emitted_stream_ids == {"b"}
    assert decoded == (["b"] if during == "flow" else ["a", "b"])
    assert worker.is_aborted("a") and not worker.is_aborted("b")
    assert "a" not in worker.stream_states and "b" in worker.stream_states
    worker.abort_callback.assert_called_once_with("a")
    assert send_codes(worker, "a") == []


@pytest.mark.parametrize("failed", ["a", "b"])
def test_encoder_failure_retries_only_healthy_rows_without_committing_rng(
    first_hop_worker, monkeypatch, failed
):
    worker = first_hop_worker
    worker.reference_prefix = SimpleNamespace(cached=None)
    worker.group_mels = FirstBatchStreamingVocoder.group_mels.__get__(worker)
    cpu_fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda devices: cpu_fork(devices=[]))
    monkeypatch.setattr(torch.cuda, "set_rng_state", Mock())
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", Mock(return_value=torch.tensor([1]))
    )
    failed_state = worker.stream_states[failed]
    initial = torch.Generator().manual_seed(86).get_state()
    for state in worker.stream_states.values():
        state.cpu_rng = initial.clone()
        state.cuda_rng = torch.tensor([1])

    def collect_row(state, available_codes, is_final, conditions):
        assert len(state.codes) == available_codes == 11
        torch.rand(1)
        if state is failed_state:
            raise ValueError("failed row encoder")
        else:
            pass

    worker.stream_mel = collect_row
    replays = []

    def replay(rid, state, *, arrived_codes, prepared_mel, is_final):
        assert len(state.codes) == arrived_codes == 11 and prepared_mel is None
        assert state.decoded_codes == 0 and torch.equal(state.cpu_rng, initial)
        state.decoded_codes = 8
        replays.append(rid)
        return torch.ones(1)

    worker.decode_delta = replay
    result = send_codes(worker, "a", "b")
    healthy = "b" if failed == "a" else "a"
    assert sorted(result) == sorted([(failed, "error"), (healthy, "stream")])
    assert replays == [healthy] and failed not in worker.stream_states
    assert worker.emitted_stream_ids == {healthy}
    worker.abort_callback.assert_called_once_with(failed)


def test_batch_resources_released_on_stop(first_hop_worker):
    worker = first_hop_worker
    worker.first_graph = Mock()
    worker.failed_reference = b"old"
    worker.on_serving_stop()
    assert worker.first_graph is None and worker.failed_reference is None
    assert worker.flow is None and worker.reference_prefix is None
    worker.codec.close.assert_called_once()


@pytest.mark.parametrize("unavailable", ["prefix", "oom"])
def test_first_hop_unavailable_uses_single_row_fallback(first_hop_worker, unavailable):
    worker = first_hop_worker
    if unavailable == "prefix":
        worker.group_mels = FirstBatchStreamingVocoder.group_mels.__get__(worker)
    else:
        worker.group_mels.side_effect = torch.OutOfMemoryError("batch graph")
    worker.decode_delta = Mock(return_value=torch.ones(1))
    assert send_codes(worker, "a", "b") == [("a", "stream"), ("b", "stream")]
    assert all(
        call.kwargs["prepared_mel"] is None
        for call in worker.decode_delta.call_args_list
    )
    assert worker.failed_reference == (b"shared" if unavailable == "oom" else None)
    worker.abort_callback.assert_not_called()


@pytest.mark.parametrize("batched", [True, False])
def test_first_hop_grids_leave_paged_history_for_the_following_hop(batched):
    worker = first_chunk_vocoder(342, True, 3)
    history = PagedHistoryPrefill.__new__(PagedHistoryPrefill)
    history.transaction = ContextVar("test_flow_history", default=None)
    history.for_voice = Mock(return_value=nullcontext())
    worker.reference_prefix = history
    worker.streaming_flow_steps = 3
    worker.first_hop_flow_steps = 2
    worker.first_flow = TestFlow()
    worker.first_reference_prefix = Mock()
    worker.first_reference_prefix.for_voice.return_value = nullcontext()
    first_inference = worker.first_flow.inference

    def first(**kwargs):
        assert history.transaction.get() is None
        return first_inference(**kwargs)

    worker.first_flow.inference = Mock(side_effect=first)
    native = worker.flow.inference
    pages = PagedHistory((torch.ones(2),))

    def steady(**kwargs):
        tx = history.transaction.get()
        assert tx is not None
        if kwargs["finalize"]:
            assert tx.history is pages
        else:
            assert tx.history is None
            tx.candidate = pages
            tx.written_steps = 3
        return native(**kwargs)

    worker.flow.inference = Mock(side_effect=steady)
    state = worker.create_stream_state("row")
    worker.latch_stream_contract(
        "row", state, {"reference_audio": "test", "effective_seed": 17}, origin="test"
    )
    state.codes = list(range(11))
    prepared = torch.ones(1, 1, 32) if batched else None
    worker.decode_delta("row", state, is_final=False, prepared_mel=prepared)
    assert state.private_history is None
    assert worker.first_flow.inference.call_count == int(not batched)
    worker.flow.inference.assert_not_called()
    state.codes.extend(range(11, 36))
    worker.decode_delta("row", state, is_final=False)
    assert state.private_history is pages
    worker.decode_delta("row", state, is_final=True)
    assert state.private_history is None and state.emitted_samples == 36 * 1920
