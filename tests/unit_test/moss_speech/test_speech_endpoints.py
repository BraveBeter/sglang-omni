# SPDX-License-Identifier: Apache-2.0
"""Dedicated endpoint lowering reuses the model's canonical chat contract."""
import io

import numpy as np
import pytest
import soundfile as sf

from sglang_omni.client import Client
from sglang_omni.models.moss_speech.request_builders import normalize_and_validate
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_service import SpeechRequestValidator
from sglang_omni.serve.speech_to_text import build_speech_to_text_generate_request


def speech(**kwargs):
    service = SpeechRequestValidator(default_model="moss")
    request = CreateSpeechRequest(input="Hello.", **kwargs)
    return Client._build_omni_request(service.build_generate_request(request))


def audio_bytes():
    data = io.BytesIO()
    sf.write(data, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return data.getvalue()


def test_speech_wire_keeps_explicit_parameters_and_original_input():
    wire = speech(temperature=0.0, top_p=1.0, seed=123, max_new_tokens=17)
    state = normalize_and_validate(wire, request_id="speech")
    assert state.output_modality == "audio"
    from sglang_omni.models.moss_speech.audio_tasks import TTS_INPUT_PREFIX

    assert state.turns[-1] == {
        "role": "user",
        "kind": "text",
        "text": TTS_INPUT_PREFIX + "Hello.",
    }
    assert state.turns[0]["role"] == "system"
    assert (
        state.temperature,
        state.top_p,
        state.effective_seed,
        state.max_new_tokens,
    ) == (0.0, 1.0, 123, 17)
    assert {"temperature", "top_p", "seed", "max_new_tokens"} <= set(
        state.explicit_params
    )


def test_speech_endpoint_defaults_do_not_become_explicit_model_parameters():
    state = normalize_and_validate(speech(), request_id="defaults")
    assert not state.explicit_params


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voice": "alice"},
        {"speed": 1.5},
        {"language": "English"},
        {"instructions": "whisper"},
        {"max_new_tokens": 513},
    ],
)
def test_speech_rejects_unqualified_options(kwargs):
    with pytest.raises(ValueError):
        normalize_and_validate(speech(**kwargs), request_id="bad")


def test_transcription_wire_normalizes_audio_and_greedy_default():
    generate = build_speech_to_text_generate_request(
        audio_bytes=audio_bytes(),
        filename="input.wav",
        content_type="audio/wav",
        model="moss",
        language=None,
        prompt=None,
        temperature=None,
        max_new_tokens=23,
    )
    wire = Client._build_omni_request(generate)
    state = normalize_and_validate(wire, request_id="asr")
    assert state.output_modality == "text"
    assert [turn["kind"] for turn in state.turns] == ["text", "audio"]
    assert state.turns[0]["role"] == "system"
    assert state.temperature == 0.0 and "temperature" in state.explicit_params
    assert state.max_new_tokens == 23


@pytest.mark.parametrize(
    "extra",
    [
        {"language": "en"},
        {"prompt": "hidden hints"},
        {"task": "translate"},
        {"detect_language": True},
        {"segment_timestamps": True},
    ],
)
def test_transcription_rejects_unqualified_options(extra):
    args = dict(
        audio_bytes=audio_bytes(),
        filename="a.wav",
        content_type="audio/wav",
        model="moss",
        language=None,
        prompt=None,
        temperature=None,
    )
    generate = build_speech_to_text_generate_request(**{**args, **extra})
    with pytest.raises(ValueError):
        normalize_and_validate(
            Client._build_omni_request(generate), request_id="bad-asr"
        )


@pytest.mark.parametrize("task", ["tts", "asr"])
@pytest.mark.parametrize("wire_form", [False, True])
def test_chat_metadata_task_does_not_change_chat_input(task, wire_form):
    from sglang_omni.client import GenerateRequest, Message
    from sglang_omni.proto.request import OmniRequest

    generate = GenerateRequest(
        messages=[Message(role="user", content="Hello.")],
        stream=False,
        metadata={"task": task},
    )
    request = (
        Client._build_omni_request(generate)
        if wire_form
        else OmniRequest(inputs=generate)
    )
    state = normalize_and_validate(request, request_id="ordinary-chat")
    assert state.turns == [{"role": "user", "kind": "text", "text": "Hello."}]
    assert state.output_modality == "text"
