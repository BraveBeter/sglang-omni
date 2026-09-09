"""CPU contracts for the public Client wire boundary and HTTP preflight."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client
from sglang_omni.models.moss_speech import request_builders as rb
from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.serve.openai_api import _build_chat_generate_request, create_app
from sglang_omni.serve.protocol import ChatCompletionRequest


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_public_client_wire_preserves_request(modality: str) -> None:
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        modalities=[modality],
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_completion_tokens=13,
        stream=False,
    )
    wire = Client._build_omni_request(_build_chat_generate_request(req))
    state = rb.normalize_and_validate(wire, request_id="http-wire")
    assert (
        state.output_modality,
        state.temperature,
        state.top_p,
        state.effective_seed,
        state.max_new_tokens,
    ) == (modality, 0.0, 1.0, 42, 13)
    assert rb.resolve_terminal_stages(wire) == [
        rb.AUDIO_TERMINAL if modality == "audio" else rb.TEXT_TERMINAL
    ]


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_terminal_public_result(modality: str) -> None:
    from sglang_omni.models.moss_speech.stages import terminal_result

    state = MossSpeechState(
        output_modality=modality,
        generated_text="hello",
        audio_samples=[0.1, 0.2],
        prompt_grid_len=5,
        output_grid=[[10, 512]] * 3,
        finish_reason="length",
    )
    chunk = Client._default_result_builder("rid", terminal_result(state))
    assert chunk.finish_reason == "length"
    assert chunk.usage.prompt_tokens == 5
    assert chunk.usage.completion_tokens == 3
    assert chunk.usage.total_tokens == 8
    if modality == "audio":
        assert chunk.audio_data == [0.1, 0.2]
        assert chunk.sample_rate == 24000
    else:
        assert chunk.text == "hello"


@pytest.mark.parametrize(
    "extra",
    [
        {"stream": True},
        {"modalities": ["text", "audio"]},
        {"min_p": 0.2},
        {"stop": ["x"]},
        {"audio": {"voice": "custom"}},
        {"messages": []},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "bad", "format": "wav"},
                        }
                    ],
                }
            ]
        },
    ],
)
def test_http_preflight_rejects_before_submission(extra: dict[str, Any]) -> None:
    class NeverClient:
        async def completion(self, *args: Any, **kwargs: Any) -> Any:
            pytest.fail("invalid request reached GPU pipeline")

    app = create_app(
        NeverClient(),
        chat_request_validator=MossSpeechPipelineConfig.validate_chat_request,
    )
    body = {"messages": [{"role": "user", "content": "hello"}], **extra}
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400, response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disconnect", "cancel", "complete", "failure"])
async def test_chat_disconnect_cleanup(mode: str) -> None:
    from sglang_omni.serve.openai_api import _await_chat_response

    active = asyncio.Event()
    closed = asyncio.Event()
    aborted = []

    class FakeClient:
        async def completion(self, *args: Any, **kwargs: Any) -> Any:
            active.set()
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
            return mode == "disconnect" and active.is_set()

    task = asyncio.create_task(
        _await_chat_response(
            Request(),
            FakeClient(),
            SimpleNamespace(),
            request_id="rid",
            audio_format="wav",
        )
    )
    await active.wait()
    if mode == "cancel":
        task.cancel()
    if mode in ("cancel", "disconnect"):
        with pytest.raises(asyncio.CancelledError):
            await task
        assert aborted == ["rid"]
    elif mode == "failure":
        with pytest.raises(RuntimeError, match="failed"):
            await task
        assert aborted == []
    else:
        assert await task == "done"
        assert aborted == []
    assert closed.is_set()


@pytest.mark.parametrize(
    "extra",
    [
        {"max_tokens": 0},
        {"max_completion_tokens": 0},
        {"max_tokens": -1},
        {"max_tokens": 4, "max_completion_tokens": 5},
        {"top_k": -2},
        {"seed": 2**64},
    ],
)
def test_invalid_length_and_sampling_preflight(extra: dict[str, Any]) -> None:
    app = create_app(
        SimpleNamespace(),
        chat_request_validator=MossSpeechPipelineConfig.validate_chat_request,
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], **extra},
        )
    assert response.status_code == 400, response.text


def test_public_audios_placeholder_binding() -> None:
    from tests.unit_test.moss_speech.test_request_contract import _b64

    req = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "", "format": "wav"},
                    }
                ],
            }
        ],
        audios=[_b64()],
        stream=False,
    )
    state = rb.normalize_and_validate(
        Client._build_omni_request(_build_chat_generate_request(req)),
        request_id="audios",
    )
    assert len(rb.pop_decoded_audio(state)) == 1


def test_remote_validation_error_maps_to_400() -> None:
    class ErrorClient:
        async def completion(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                str(rb.RequestValidationError("post-encode context too long"))
            )

    with TestClient(create_app(ErrorClient())) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
    assert response.status_code == 400


def test_deployment_limits_reject_before_native() -> None:
    from sglang_omni.models.moss_speech.stages import validate_prompt_length

    validate_prompt_length(512)
    with pytest.raises(rb.RequestValidationError):
        validate_prompt_length(513)
    assert MossSpeechPipelineConfig.generation_admission_defaults() == {
        "max_running_requests": 4,
        "max_queued_requests": 20,
    }
    app = create_app(
        SimpleNamespace(),
        chat_request_validator=MossSpeechPipelineConfig.validate_chat_request,
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 513,
            },
        )
    assert response.status_code == 400


def test_formal_kv_pool_matches_deployment_budget() -> None:
    config = MossSpeechPipelineConfig(model_path="unused")
    ar = next(stage for stage in config.stages if stage.name == "ar_engine")
    overrides = ar.factory_args["server_args_overrides"]
    assert overrides["max_total_tokens"] == 4 * 1024
    assert overrides["max_running_requests"] == 4


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "audio",
    [
        "malformed",
        [1],
        [],
        1,
        0,
        False,
        None,
        {"data": 42},
        {"data": None},
        {"data": []},
    ],
)
def test_malformed_nested_audio_returns_400(stream, audio):
    from sglang_omni.models.moss_speech.config import MossSpeechStreamingPipelineConfig

    app = create_app(
        SimpleNamespace(),
        chat_request_validator=MossSpeechStreamingPipelineConfig.validate_chat_request,
    )
    with TestClient(app, raise_server_exceptions=False) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "input_audio", "input_audio": audio}],
                    }
                ],
                "stream": stream,
            },
        )
    assert response.status_code == 400, response.text
    assert "input_audio" in response.json()["detail"]
