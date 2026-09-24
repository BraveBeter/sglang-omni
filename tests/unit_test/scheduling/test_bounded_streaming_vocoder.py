# SPDX-License-Identifier: Apache-2.0
"""Bounded decode scheduling keeps pending work and terminal ordering explicit."""

from __future__ import annotations

import asyncio

import torch

from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.scheduling.test_streaming_vocoder import (
    _drain,
    _FakeStreamingVocoder,
    _item,
    _payload,
)


class BoundedVocoder(_FakeStreamingVocoder):
    def __init__(self, **kwargs):
        super().__init__(defer_stream_decode=True, **kwargs)

    def decode_delta(self, request_id, state, *, is_final):
        self.calls.append(f"decode:{request_id}:{'final' if is_final else 'delta'}")
        if state.decoded_upto == len(state.frames):
            return None
        value = state.frames[state.decoded_upto]
        state.decoded_upto += 1
        return value.to(torch.float32)

    def has_pending_stream_decode(self, request_id, state, *, is_final):
        return state.decoded_upto < len(state.frames)


def tick(scheduler):
    loop = asyncio.new_event_loop()
    try:
        message = scheduler._next_message()
        if message is not None:
            scheduler._handle_message(message, loop)
    finally:
        loop.close()


def drain_pending(scheduler):
    for _ in range(200):
        if not scheduler._deferred_stream_ready and scheduler.inbox.empty():
            return
        tick(scheduler)
    raise AssertionError("Pending stream work did not drain")


def test_single_code_fifo_backlog_yields_to_short_request():
    scheduler = BoundedVocoder()
    for index in range(20):
        scheduler.inbox.put(IncomingMessage("long", "stream_chunk", _item([index])))
    scheduler.inbox.put(IncomingMessage("short", "stream_chunk", _item([99])))
    drain_pending(scheduler)
    messages = _drain(scheduler)
    ids = [message.request_id for message in messages]
    assert ids[:2] == ["long", "short"]
    assert ids.count("long") == 20
    assert ids.count("short") == 1


def test_done_before_payload_drains_without_new_input_and_finishes_once():
    scheduler = BoundedVocoder()
    for value in (1, 2, 3):
        scheduler._on_chunk("r", _item([value]))
    scheduler._on_done("r")
    assert "r" in scheduler._pending_done
    scheduler._on_streaming_new_request("r", _payload())
    assert _drain(scheduler) == []
    tick(scheduler)
    assert [message.type for message in _drain(scheduler)] == ["stream"]
    assert "r" in scheduler._stream_states
    drain_pending(scheduler)
    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "stream", "result"]
    assert not scheduler._stream_states
    assert not scheduler._stream_payloads
    assert not scheduler._deferred_stream_ready
    assert not scheduler._deferred_stream_final
    scheduler._on_chunk("r", _item([9]))
    assert not scheduler._stream_states


def test_abort_pending_work_prevents_decode_and_late_recreation():
    scheduler = BoundedVocoder()
    scheduler._on_chunk("r", _item([1]))
    scheduler._on_done("r")
    scheduler.abort("r")
    scheduler._on_chunk("r", _item([2]))
    assert not scheduler._deferred_stream_ready
    assert not scheduler._deferred_stream_final
    assert not scheduler._stream_states
    assert not any(call.startswith("decode:") for call in scheduler.calls)
    assert _drain(scheduler) == []


def test_failure_aborts_only_current_request_and_drains_other_request():
    class FailingVocoder(BoundedVocoder):
        def decode_delta(self, request_id, state, *, is_final):
            if request_id == "bad":
                raise RuntimeError("decode failed")
            return super().decode_delta(request_id, state, is_final=is_final)

    scheduler = FailingVocoder()
    scheduler._on_chunk("bad", _item([1]))
    scheduler._on_chunk("ok", _item([2]))
    drain_pending(scheduler)
    assert [(message.request_id, message.type) for message in _drain(scheduler)] == [
        ("bad", "error"),
        ("ok", "stream"),
    ]
    assert scheduler._is_aborted("bad")
    assert "bad" not in scheduler._stream_states


def test_stop_clears_deferred_work_and_fallback_waits_until_terminal():
    scheduler = BoundedVocoder(fallback=torch.tensor([9.0]))
    scheduler._on_done("empty")
    scheduler._on_streaming_new_request("empty", _payload("empty"))
    drain_pending(scheduler)
    assert [message.type for message in _drain(scheduler)] == ["stream", "result"]
    scheduler._on_chunk("r", _item([1]))
    scheduler.stop()
    assert not scheduler._stream_states
    assert not scheduler._deferred_stream_ready
    assert not scheduler._deferred_stream_final


def test_ingest_budget_cannot_starve_ready_decode():
    scheduler = BoundedVocoder()
    scheduler._stream_ingest_budget = 2
    for index in range(5):
        scheduler.inbox.put(IncomingMessage("r", "stream_chunk", _item([index])))
    tick(scheduler)
    tick(scheduler)
    tick(scheduler)
    tick(scheduler)
    assert len(_drain(scheduler)) == 1
    assert not scheduler.inbox.empty()


def test_abort_during_hop_suppresses_output_and_followup_work():
    import threading
    import time

    started = threading.Event()
    finish = threading.Event()

    class BlockingVocoder(BoundedVocoder):
        def decode_delta(self, request_id, state, *, is_final):
            started.set()
            assert finish.wait(timeout=3)
            return super().decode_delta(request_id, state, is_final=is_final)

    scheduler = BlockingVocoder()
    scheduler._on_chunk("r", _item([1]))
    scheduler._on_chunk("r", _item([2]))
    decode = threading.Thread(target=scheduler._run_deferred_stream_step)
    abort = threading.Thread(target=scheduler.abort, args=("r",))
    decode.start()
    assert started.wait(timeout=3)
    abort.start()
    deadline = time.monotonic() + 3
    while not scheduler._is_aborted("r") and time.monotonic() < deadline:
        time.sleep(0.001)
    try:
        assert scheduler._is_aborted("r")
    finally:
        finish.set()
        decode.join(timeout=3)
        abort.join(timeout=3)
    assert not decode.is_alive() and not abort.is_alive()
    assert _drain(scheduler) == []
    assert not scheduler._deferred_stream_ready
    assert not scheduler._stream_states
