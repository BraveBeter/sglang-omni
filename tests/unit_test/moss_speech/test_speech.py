# SPDX-License-Identifier: Apache-2.0
"""Dedicated speech surfaces share the native MOSS-Speech request contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, Message
from sglang_omni.models.moss_speech.audio_tasks import RequestValidationError
from sglang_omni.models.moss_speech.request_builders import normalize_and_validate
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request
from tests.unit_test.fixtures.moss_speech_requests import (
    NativeRequestClient,
    native_request,
    wav_bytes,
)


@pytest.mark.parametrize("endpoint", ["tts", "asr"])
@pytest.mark.parametrize("explicit", [False, True])
def test_speech_endpoints_preserve_input_and_parameter_presence(
    endpoint: str, explicit: bool
) -> None:
    if endpoint == "tts":
        speech = CreateSpeechRequest(
            input="Hello.",
            temperature=0.2 if explicit else None,
            seed=7 if explicit else None,
            max_new_tokens=23 if explicit else None,
        )
        generation = SpeechRequestValidator(
            default_model="moss"
        ).build_generate_request(speech)
        expected_modality = "audio"
        expected_temperature = 0.2 if explicit else 0.6
    else:
        generation = build_speech_to_text_generate_request(
            audio_bytes=wav_bytes(),
            filename="input.wav",
            content_type="audio/wav",
            model="moss",
            language=None,
            prompt=None,
            temperature=0.2 if explicit else None,
            max_new_tokens=23 if explicit else None,
        )
        expected_modality = "text"
        expected_temperature = 0.2 if explicit else 0.0
    original = generation.to_dict()
    state, native = native_request(Client.build_omni_request(generation), endpoint)
    assert generation.to_dict() == original
    assert state.output_modality == expected_modality
    assert native.params.temperature == expected_temperature
    assert native.params.max_new_tokens == (
        23 if explicit else 40960 - len(native.prompt_rows)
    )
    assert state.turns[0]["role"] == "system"
    if endpoint == "tts":
        assert (
            state.turns[-1]["text"]
            == "Repeat the following text verbatim, without adding anything: Hello."
        )
        if explicit:
            assert native.effective_seed == 7
        else:
            assert state.temperature is None
    else:
        assert state.turns[-1]["kind"] == "audio"


@pytest.mark.parametrize("task", ["tts", "asr"])
def test_chat_metadata_does_not_select_a_speech_endpoint(task: str) -> None:
    generation = GenerateRequest(
        messages=[Message("user", "Hello.")], metadata={"task": task}, stream=False
    )
    state = normalize_and_validate(
        Client.build_omni_request(generation), request_id="chat"
    )
    assert state.turns == [{"role": "user", "kind": "text", "text": "Hello."}]
    assert state.output_modality == "text"


@pytest.mark.parametrize(
    "endpoint,options",
    [
        ("tts", {"voice": "alice"}),
        ("tts", {"speed": 1.5}),
        ("tts", {"language": "English"}),
        ("tts", {"instructions": "whisper"}),
        ("tts", {"max_new_tokens": 40960}),
        ("asr", {"language": "en"}),
        ("asr", {"prompt": "hint"}),
        ("asr", {"task": "translate"}),
        ("asr", {"detect_language": True}),
        ("asr", {"segment_timestamps": True}),
    ],
)
def test_speech_endpoints_reject_unqualified_controls(
    endpoint: str, options: dict[str, object]
) -> None:
    if endpoint == "tts":
        generation = SpeechRequestValidator(
            default_model="moss"
        ).build_generate_request(CreateSpeechRequest(input="Hello.", **options))
    else:
        arguments = {
            "audio_bytes": wav_bytes(),
            "filename": "input.wav",
            "content_type": "audio/wav",
            "model": "moss",
            "language": None,
            "prompt": None,
            "temperature": None,
        }
        generation = build_speech_to_text_generate_request(**(arguments | options))
    with pytest.raises(RequestValidationError):
        normalize_and_validate(
            Client.build_omni_request(generation), request_id="speech-options"
        )


@pytest.mark.parametrize("endpoint", ["tts", "asr"])
def test_real_speech_http_routes_preserve_parameters_and_model_errors(
    endpoint: str,
) -> None:
    boundary = NativeRequestClient()
    with TestClient(create_app(boundary, model_name="moss-speech")) as http:
        if endpoint == "tts":
            body = {
                "input": "Hello.",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 17,
            }
            response = http.post("/v1/audio/speech", json=body)
            assert response.status_code == 200, response.text
            assert response.content[:4] == b"RIFF"
            assert boundary.native is not None
            assert boundary.native.params.top_p == 1.0
            invalid_response = http.post("/v1/audio/speech", json=body | {"speed": 1.5})
        else:
            form = {
                "model": "moss-speech",
                "temperature": "0.0",
                "max_new_tokens": "17",
            }
            files = {"file": ("input.wav", wav_bytes(), "audio/wav")}
            response = http.post("/v1/audio/transcriptions", data=form, files=files)
            assert response.status_code == 200, response.text
            assert response.json()["text"] == "accepted"
            invalid_response = http.post(
                "/v1/audio/transcriptions", data=form | {"language": "en"}, files=files
            )
        assert boundary.native is not None
        assert boundary.native.params.temperature == 0.0
        assert boundary.native.params.max_new_tokens == 17
        assert invalid_response.status_code == 400, invalid_response.text
