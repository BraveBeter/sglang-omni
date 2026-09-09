"""Synchronous model preflight must not block the HTTP event loop."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import CompletionResult
from sglang_omni.serve.openai_api import create_app


@pytest.mark.asyncio
async def test_preflight_keeps_loop_responsive_before_admission():
    entered = threading.Event()
    released = threading.Event()
    observed = []
    owner = threading.get_ident()

    def validate(req):
        entered.set()
        observed.append((threading.get_ident() != owner, released.wait(2)))

    class Client:
        async def completion(self, *args, **kwargs):
            assert released.is_set(), "admission preceded completed validation"
            return CompletionResult(request_id="test", text="ok")

    app = create_app(Client(), chat_request_validator=validate)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        task = asyncio.create_task(
            http.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
        )
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not entered.is_set():
                assert not task.done(), "request ended before preflight"
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.001)
            # This coroutine can only release the validator if the loop is free.
            released.set()
            response = await task
            assert response.status_code == 200
            assert observed == [(True, True)]
        finally:
            released.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "error,status", [(ValueError("invalid"), 400), (RuntimeError("bug"), 500)]
)
def test_preflight_errors_before_backend_or_sse(stream, error, status):
    def validate(req):
        raise error

    app = create_app(SimpleNamespace(), chat_request_validator=validate)
    with TestClient(app, raise_server_exceptions=False) as http:
        result = http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )
    assert result.status_code == status
    assert "text/event-stream" not in result.headers.get("content-type", "")
