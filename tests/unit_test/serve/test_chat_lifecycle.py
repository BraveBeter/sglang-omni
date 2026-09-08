"""Generic chat preflight, error classification and disconnect ownership."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.admission import QueueFullError
from sglang_omni.serve.openai_api import _await_chat_response, create_app


@pytest.mark.parametrize("mode", ["disconnect", "cancel", "complete", "failure"])
@pytest.mark.asyncio
async def test_chat_waiter_cleanup(mode: str) -> None:
    started = asyncio.Event()
    closed = asyncio.Event()
    aborted = []

    class Client:
        async def completion(self, *args: Any, **kwargs: Any) -> Any:
            started.set()
            try:
                if mode == "complete":
                    return "done"
                if mode == "failure":
                    raise RuntimeError("failed")
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def abort(self, rid: str) -> None:
            aborted.append(rid)

    class Request:
        async def is_disconnected(self) -> bool:
            return mode == "disconnect" and started.is_set()

    task = asyncio.create_task(
        _await_chat_response(
            Request(), Client(), SimpleNamespace(), request_id="rid", audio_format="wav"
        )
    )
    await started.wait()
    if mode == "cancel":
        task.cancel()
    if mode in ("cancel", "disconnect"):
        with pytest.raises(asyncio.CancelledError):
            await task
        assert aborted == ["rid"]
    elif mode == "failure":
        with pytest.raises(RuntimeError, match="failed"):
            await task
        assert not aborted
    else:
        assert await task == "done"
        assert not aborted
    assert closed.is_set()


@pytest.mark.parametrize(
    "error,status",
    [
        (QueueFullError(), 503),
        (RuntimeError("Invalid model request: context length exceeds limit"), 400),
        (RuntimeError("internal failure"), 500),
    ],
)
def test_chat_error_status(error: Exception, status: int) -> None:
    class Client:
        async def completion(self, *args: Any, **kwargs: Any) -> Any:
            raise error

    with TestClient(create_app(Client())) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
    assert response.status_code == status


def test_stream_preflight_precedes_headers() -> None:
    def reject(request: Any) -> None:
        assert request.stream
        raise ValueError("stream unsupported")

    with TestClient(
        create_app(SimpleNamespace(), chat_request_validator=reject)
    ) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        )
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
