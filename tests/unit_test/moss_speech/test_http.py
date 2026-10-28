# SPDX-License-Identifier: Apache-2.0
"""HTTP and Client results preserve the model's native request contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.models.moss_speech.payload_types import MossSpeechState, OutputModality
from sglang_omni.models.moss_speech.request_builders import terminal_result
from sglang_omni.serve.openai_api import create_app
from tests.unit_test.fixtures.moss_speech_requests import NativeRequestClient


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_terminal_usage_and_finish_reason_reach_public_result(
    modality: OutputModality,
) -> None:
    state = MossSpeechState(
        output_modality=modality,
        generated_text="hello",
        audio_samples=[0.1, 0.2],
        prompt_grid_len=5,
        output_grid=[[1234, 512]] * 3,
        finish_reason="length",
    )
    result = Client.default_result_builder("result", terminal_result(state))
    assert result.finish_reason == "length"
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 8
    if modality == "audio":
        assert result.audio_data == [0.1, 0.2]
        assert result.sample_rate == 24000
    else:
        assert result.text == "hello"


def test_real_http_route_preserves_explicit_sampling_and_reports_model_errors() -> None:
    boundary = NativeRequestClient()
    with TestClient(create_app(boundary, model_name="moss-speech")) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello."}],
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 7,
                "max_completion_tokens": 17,
                "stream": False,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "accepted"
        assert boundary.native is not None
        parameters = boundary.native.params
        assert (
            parameters.temperature,
            parameters.top_p,
            parameters.top_k,
            parameters.max_new_tokens,
        ) == (0.0, 1.0, -1, 17)
        response = http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello."}],
                "min_p": 0.2,
                "stream": False,
            },
        )
        assert response.status_code == 400
        assert "min_p" in response.json()["detail"]


def test_http_malformed_nested_audio_has_model_error_status() -> None:
    with TestClient(
        create_app(NativeRequestClient(), model_name="moss-speech")
    ) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "input_audio", "input_audio": None}],
                    }
                ],
                "stream": False,
            },
        )
    assert response.status_code == 400, response.text
    assert "input_audio" in response.json()["detail"]
