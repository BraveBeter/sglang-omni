# SPDX-License-Identifier: Apache-2.0
"""Public request parameters, modality routing, and input validation."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, Message, SamplingParams
from sglang_omni.models.moss_speech.audio_tasks import RequestValidationError
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.models.moss_speech.request_builders import (
    normalize_and_validate,
    pop_decoded_audio,
    resolve_output_terminal,
    resolve_terminal_stages,
)
from sglang_omni.models.moss_speech.request_data import build_sglang_moss_request
from sglang_omni.models.moss_speech.streaming import (
    make_stream_output_builder,
    resolve_stream_terminal,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_api import build_chat_generate_request
from sglang_omni.serve.openai_errors import generation_error_status_code
from sglang_omni.serve.protocol import ChatCompletionRequest
from tests.unit_test.fixtures.moss_speech_requests import (
    canonical_fixture,
    native_request,
    wav_bytes,
)


@pytest.mark.parametrize("entry", ["typed", "serialized", "client", "http"])
@pytest.mark.parametrize(
    "sampling,expected",
    [
        (SamplingParams(), (0.6, 0.95, 20, 1.1, 200)),
        (
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_new_tokens=17,
            ),
            (0.0, 1.0, -1, 1.0, 17),
        ),
        (
            SamplingParams(
                temperature=0.2,
                top_p=0.8,
                top_k=7,
                repetition_penalty=1.3,
                max_new_tokens=19,
            ),
            (0.2, 0.8, 7, 1.3, 19),
        ),
        (
            SamplingParams(temperature=1.0, top_p=1.0, top_k=-1),
            (1.0, 1.0, -1, 1.1, 200),
        ),
        (
            SamplingParams(top_k=0, max_new_tokens=1),
            (0.6, 0.95, 0, 1.1, 1),
        ),
        (
            SamplingParams(top_k=1),
            (0.6, 0.95, 1, 1.1, 200),
        ),
    ],
)
def test_sampling_presence_reaches_native_parameters(
    entry: str,
    sampling: SamplingParams,
    expected: tuple[float, float, int, float, int],
) -> None:
    """Unmarked explicit scalar parameters previously disappeared at native admission."""
    generation = GenerateRequest(
        messages=[Message("user", "Hello.")], sampling=sampling, stream=False
    )
    original_sampling = sampling.to_dict()
    if entry == "typed":
        request = OmniRequest(inputs=generation)
    elif entry == "serialized":
        request = OmniRequest(inputs=generation.to_dict())
    elif entry == "client":
        request = Client.build_omni_request(generation)
    else:
        chat = ChatCompletionRequest(
            model="moss-speech",
            messages=[{"role": "user", "content": "Hello."}],
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            repetition_penalty=sampling.repetition_penalty,
            max_tokens=sampling.max_new_tokens,
        )
        request = Client.build_omni_request(build_chat_generate_request(chat))
    state, native = native_request(request)
    assert state.output_modality == "text"
    assert state.stop == []
    actual = native.params
    assert (
        actual.temperature,
        actual.top_p,
        actual.top_k,
        actual.repetition_penalty,
        actual.max_new_tokens,
    ) == expected
    assert actual.do_sample is (expected[0] > 0 and expected[2] != 1)
    assert native.req.sampling_params.max_new_tokens == expected[4]
    assert native.max_new_tokens == expected[4]
    assert native.enforce_request_limits
    assert list(native.req.origin_input_ids) == [151644]
    assert sampling.to_dict() == original_sampling


@pytest.mark.parametrize(
    "case_name", ["t2t_short", "t2s_cn", "s2t_cn", "s2s_cn", "mixed_multiturn"]
)
def test_four_modes_and_mixed_history_bind_one_terminal(case_name: str) -> None:
    case = canonical_fixture().cases[case_name]
    encoded = base64.b64encode(wav_bytes()).decode("ascii")
    messages = [
        Message(
            turn["role"],
            (
                turn["text"]
                if turn["kind"] == "text"
                else [{"type": "input_audio", "input_audio": {"data": encoded}}]
            ),
        )
        for turn in case.turns
    ]
    request = Client.build_omni_request(
        GenerateRequest(
            messages=messages, output_modalities=[case.output_modality], stream=False
        )
    )
    state = normalize_and_validate(request, request_id=case_name)
    assert state.turns == case.turns
    assert state.output_modality == case.output_modality
    decoded = pop_decoded_audio(state)
    assert len(decoded) == len(case.audio_codes)
    assert all(
        sample_rate == 16000 and waveform.numel() == 1600
        for waveform, sample_rate, _ in decoded
    )
    assert pop_decoded_audio(state) == []
    assert "decoded_audio" not in state.to_dict()
    terminal = "audio_vocoder" if case.output_modality == "audio" else "text_decode"
    assert resolve_terminal_stages(request) == [terminal]
    for payload_state in (state, state.to_dict()):
        assert (
            resolve_output_terminal(
                case_name, StagePayload(case_name, request, payload_state)
            )
            == terminal
        )


def test_audio_placeholders_bind_sources_in_order_and_reject_ambiguous_sources(
    tmp_path: Path,
) -> None:
    sources = []
    for index, sample_rate in enumerate([16000, 24000]):
        path = tmp_path / f"audio-{index}.wav"
        path.write_bytes(wav_bytes(sample_rate=sample_rate))
        sources.append(str(path))
    generation = build_chat_generate_request(
        ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": ""}}],
                }
            ]
            * 2,
            audios=sources,
            stream=False,
        )
    )
    state = normalize_and_validate(
        Client.build_omni_request(generation), request_id="sources"
    )
    assert [sample_rate for _, sample_rate, _ in pop_decoded_audio(state)] == [
        16000,
        24000,
    ]
    generation.messages = [Message("user", "no audio placeholder")]
    with pytest.raises(RequestValidationError, match="audio turns"):
        normalize_and_validate(
            Client.build_omni_request(generation), request_id="mismatch"
        )
    generation.messages = [
        Message(
            "user",
            [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(wav_bytes()).decode("ascii")
                    },
                }
            ],
        )
    ]
    generation.metadata = {"audios": sources[:1]}
    with pytest.raises(RequestValidationError, match="not both"):
        normalize_and_validate(
            Client.build_omni_request(generation), request_id="ambiguous"
        )


@pytest.mark.parametrize("entry", ["typed", "serialized"])
@pytest.mark.parametrize(
    "embedded,override,effective",
    [(True, None, True), (True, False, False), (False, True, True)],
)
def test_stream_presence_controls_routing_and_output(
    entry: str, embedded: bool, override: bool | None, effective: bool
) -> None:
    """Typed streaming requests previously normalized but failed to emit downstream chunks."""
    generation = GenerateRequest(messages=[Message("user", "Hello.")], stream=embedded)
    request = OmniRequest(
        inputs=generation if entry == "typed" else generation.to_dict(),
        params={} if override is None else {"stream": override},
    )
    _, native = native_request(request, "stream")
    assert generation.stream is embedded
    assert request.params["stream"] is effective
    native.output_rows = [(1234, 512)]
    messages = make_stream_output_builder().flush("stream", native)
    assert resolve_stream_terminal("stream", native.stage_payload) == (
        ["text_decode"] if effective else []
    )
    if effective:
        assert len(messages) == 1
        assert messages[0].target == "text_decode"
        assert messages[0].data.tolist() == [1234]
    else:
        assert messages == []
        assert native.stream_grid_cursor == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"messages": []},
        {"messages": [Message("tool", "x")]},
        {"messages": [Message("user", None)]},
        {
            "messages": [
                Message("user", [{"type": "image_url", "image_url": {"url": "x"}}])
            ]
        },
        {
            "messages": [
                Message(
                    "user",
                    [
                        {"type": "text", "text": "hi"},
                        {"type": "input_audio", "input_audio": {"data": ""}},
                    ],
                )
            ]
        },
        {
            "messages": [
                Message(
                    "user", [{"type": "input_audio", "input_audio": {"data": ""}}] * 2
                )
            ]
        },
        {"output_modalities": ["text", "audio"]},
        {"output_modalities": ["audio", "audio"]},
        {"output_modalities": ["video"]},
        {"output_modalities": ["TEXT"]},
        {"extra_params": {"ignore_eos": True}},
        {"stage_params": {"ar_engine": {"temperature": 0.4}}},
    ],
)
def test_unsupported_chat_shapes_have_http_400_surface(
    changes: dict[str, object]
) -> None:
    generation = GenerateRequest(
        **({"messages": [Message("user", "Hello.")], "stream": False} | changes)
    )
    with pytest.raises(RequestValidationError) as caught:
        normalize_and_validate(OmniRequest(inputs=generation), request_id="invalid")
    assert generation_error_status_code(caught.value) == 400


@pytest.mark.parametrize("stream", [False, True])
def test_chat_request_cannot_replace_configured_voice(stream: bool) -> None:
    generation = GenerateRequest(
        messages=[Message("user", "Hello.")],
        output_modalities=["audio"],
        metadata={"audio_config": {"voice": "custom", "format": "wav"}},
        stream=stream,
    )
    with pytest.raises(RequestValidationError, match="voice/audio configuration"):
        normalize_and_validate(
            OmniRequest(inputs=generation), request_id="voice", allow_streaming=True
        )


@pytest.mark.parametrize(
    "sampling",
    [
        SamplingParams(temperature=float("inf")),
        SamplingParams(repetition_penalty=float("nan")),
        SamplingParams(repetition_penalty=float("inf")),
        SamplingParams(top_p=0.0),
        SamplingParams(top_k=-2),
        SamplingParams(min_p=0.2),
        SamplingParams(stop=["stop"]),
        SamplingParams(stop_token_ids=[1]),
        SamplingParams(max_new_tokens=0),
        SamplingParams(max_new_tokens=513),
        SamplingParams(seed=2**64),
    ],
)
def test_unsupported_sampling_is_rejected_before_admission(
    sampling: SamplingParams,
) -> None:
    generation = GenerateRequest(
        messages=[Message("user", "Hello.")], sampling=sampling, stream=False
    )
    with pytest.raises(RequestValidationError):
        normalize_and_validate(OmniRequest(inputs=generation), request_id="sampling")


@pytest.mark.parametrize(
    "failure", ["corrupt", "nonfinite", "duration", "sample_rate", "url"]
)
def test_invalid_audio_is_rejected(failure: str) -> None:
    if failure == "url":
        encoded = "https://example.invalid/audio.wav"
    else:
        audio = {
            "corrupt": b"not audio",
            "nonfinite": wav_bytes(nonfinite=True),
            "duration": wav_bytes(sample_rate=8000, sample_count=248001),
            "sample_rate": wav_bytes(sample_rate=4000),
        }[failure]
        encoded = base64.b64encode(audio).decode("ascii")
    generation = GenerateRequest(
        messages=[
            Message("user", [{"type": "input_audio", "input_audio": {"data": encoded}}])
        ],
        stream=False,
    )
    with pytest.raises(RequestValidationError) as caught:
        normalize_and_validate(OmniRequest(inputs=generation), request_id="audio")
    assert generation_error_status_code(caught.value) == 400


def test_seed_presence_is_preserved_and_fallback_is_request_owned() -> None:
    generation = GenerateRequest(messages=[Message("user", "Hello.")], stream=False)
    _, first = native_request(OmniRequest(inputs=generation), "first")
    _, retry = native_request(OmniRequest(inputs=generation), "first")
    _, other = native_request(OmniRequest(inputs=generation), "other")
    assert first.effective_seed == retry.effective_seed
    assert first.effective_seed != other.effective_seed
    for seed in [-(2**63), -1, 0, 2**64 - 1]:
        generation.sampling.seed = seed
        _, native = native_request(OmniRequest(inputs=generation), "first")
        assert native.effective_seed == seed


def test_native_grid_removes_left_padding_and_rejects_bad_ids() -> None:
    request = OmniRequest(inputs=None)
    state = MossSpeechState(
        input_grid=[[999999, 999999], [151667, 8], [1234, 512]],
        attention_mask=[0, 1, 1],
    )
    payload = StagePayload("grid", request, state.to_dict())
    native = build_sglang_moss_request(state, payload=payload)
    assert native.prompt_rows.tolist() == [[151667, 8], [1234, 512]]
    assert list(native.req.origin_input_ids) == [8, 1234]
    assert native.mode == 0
    state.input_grid = [[151644, 512], [151646, 512], [151667, 500]]
    state.attention_mask = []
    payload.data = state.to_dict()
    native = build_sglang_moss_request(state, payload=payload)
    assert native.mode == 1
    assert list(native.req.origin_input_ids) == [151644, 151646, 500]
    assert native.req.eos_token_ids == {151643, 151645}
    assert native.stage_payload is payload
    for bad_grid in [[[151680, 512]], [[1234, 16512]], [[-1, 512]]]:
        state.input_grid, state.attention_mask = bad_grid, []
        with pytest.raises(RequestValidationError, match="out-of-vocabulary"):
            build_sglang_moss_request(state, payload=payload)
