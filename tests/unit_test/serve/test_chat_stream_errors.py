"""A post-header chat failure must be an explicit SSE error, never a success."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from sglang_omni.serve.openai_api import _chat_stream


class BrokenClient:
    def __init__(self, cancelled=False):
        self.closed = False
        self.cancelled = cancelled

    async def completion_stream(self, *args, **kwargs):
        try:
            yield SimpleNamespace(
                finish_reason=None, modality="text", text="hello", audio_b64=None
            )
            if self.cancelled:
                raise asyncio.CancelledError()
            raise RuntimeError("private internal failure")
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_error_after_first_delta_is_explicit_and_closes_iterator():
    client = BrokenClient()
    events = [
        event
        async for event in _chat_stream(
            client,
            None,
            "r",
            "resp",
            1,
            "model",
            SimpleNamespace(modalities=["text"]),
            "wav",
        )
    ]
    assert client.closed
    parsed = [
        json.loads(e.removeprefix("data: ").strip())
        for e in events
        if "[DONE]" not in e
    ]
    assert parsed[0]["choices"][0]["delta"]["content"] == "hello"
    assert parsed[-1]["error"]["code"] == "stream_error"
    assert "private internal failure" not in events[-1]
    assert not any(c.get("finish_reason") for e in parsed for c in e.get("choices", []))
    assert all("[DONE]" not in event for event in events)


@pytest.mark.asyncio
async def test_disconnect_cancellation_propagates_and_closes_iterator():
    client = BrokenClient(cancelled=True)
    with pytest.raises(asyncio.CancelledError):
        async for _ in _chat_stream(
            client,
            None,
            "r",
            "resp",
            1,
            "model",
            SimpleNamespace(modalities=["text"]),
            "wav",
        ):
            pass
    assert client.closed
