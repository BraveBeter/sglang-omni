# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint MOSS-Speech chat smoke tests with retained audio artifacts."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import requests
import soundfile as sf

from benchmarks.benchmarker.utils import managed_omni_server
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.moss_speech.config import (
    MossSpeechPipelineConfig,
    MossSpeechPreprocessingStage,
)
from sglang_omni.serve.protocol import (
    ChatCompletionAudio,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    UsageResponse,
)
from sglang_omni.utils import find_available_port

pytestmark = pytest.mark.accelerator
Language = Literal["en", "zh"]
Modality = Literal["text", "audio"]
SAMPLE_RATE_HZ = 24000
SAMPLES_PER_CODE = 1920
REQUEST_TIMEOUT_SECONDS = 300
TEXT_PROMPTS = {
    "en": "Answer briefly: what do people drink when they are thirsty?",
    "zh": "请简短回答：人们口渴时通常喝什么？",
}


@dataclass
class SmokeResult:
    text: str = ""
    audio_bytes: bytes = b""
    chunk_samples: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    usage: UsageResponse | None = None
    elapsed_seconds: float = 0.0
    first_text_seconds: float | None = None
    first_audio_seconds: float | None = None


@dataclass
class MossSpeechServer:
    base_url: str
    session: requests.Session
    output_directory: Path
    input_audio: dict[Language, str]

    def payload(
        self,
        language: Language = "en",
        input_modality: Modality = "text",
        output_modality: Modality = "text",
        streaming: bool = False,
        max_tokens: int = 256,
    ) -> dict[str, object]:
        content: str | list[dict[str, object]]
        if input_modality == "audio":
            content = [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": self.input_audio[language],
                        "format": "wav",
                    },
                }
            ]
        else:
            content = TEXT_PROMPTS[language]
        return {
            "messages": [{"role": "user", "content": content}],
            "modalities": [output_modality],
            "stream": streaming,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 42,
            "max_tokens": max_tokens,
            **({"audio": {"format": "pcm"}} if streaming else {}),
        }

    def complete(self, case: str, payload: dict[str, object]) -> SmokeResult:
        result = SmokeResult()
        streaming = payload["stream"] is True
        started_seconds = time.perf_counter()
        with (
            self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=streaming,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response,
            (self.output_directory / f"{case}.response.txt").open("w") as response_file,
        ):
            response_file.write(f"HTTP {response.status_code}\n")
            if not streaming or response.status_code != 200:
                response_file.write(response.text)
            else:
                pass
            assert response.status_code == 200, response.text
            if streaming:
                terminal_events = 0
                saw_done = False
                for line in response.iter_lines(chunk_size=None, decode_unicode=True):
                    if not line:
                        continue
                    else:
                        pass
                    response_file.write(f"{line}\n")
                    response_file.flush()
                    assert not saw_done, "Payload followed DONE sentinel"
                    assert line.startswith("data: "), line
                    event_json = line.removeprefix("data: ")
                    if event_json == "[DONE]":
                        saw_done = True
                        continue
                    else:
                        pass
                    event = ChatCompletionStreamResponse.model_validate_json(event_json)
                    assert len(event.choices) == 1
                    choice = event.choices[0]
                    if choice.finish_reason is not None:
                        assert not choice.delta.content and choice.delta.audio is None
                        terminal_events += 1
                        result.finish_reason = choice.finish_reason
                        result.usage = event.usage
                    else:
                        assert terminal_events == 0, "Payload followed terminal event"
                    if choice.delta.content:
                        result.text += choice.delta.content
                        if result.first_text_seconds is None:
                            result.first_text_seconds = (
                                time.perf_counter() - started_seconds
                            )
                        else:
                            pass
                    else:
                        pass
                    if choice.delta.audio is not None:
                        pcm_bytes = base64.b64decode(
                            choice.delta.audio.data, validate=True
                        )
                        assert len(pcm_bytes) > 0 and len(pcm_bytes) % 2 == 0
                        result.audio_bytes += pcm_bytes
                        result.chunk_samples.append(len(pcm_bytes) // 2)
                        if result.first_audio_seconds is None:
                            result.first_audio_seconds = (
                                time.perf_counter() - started_seconds
                            )
                        else:
                            pass
                    else:
                        pass
                assert terminal_events == 1 and saw_done
            else:
                completion = ChatCompletionResponse.model_validate_json(response.text)
                assert len(completion.choices) == 1
                choice = completion.choices[0]
                result.finish_reason = choice.finish_reason
                result.usage = completion.usage
                text = choice.message.get("content", "")
                assert isinstance(text, str)
                result.text = text
                if "audio" in choice.message:
                    audio = ChatCompletionAudio.model_validate(choice.message["audio"])
                    result.audio_bytes = base64.b64decode(audio.data, validate=True)
                else:
                    pass
        result.elapsed_seconds = time.perf_counter() - started_seconds
        assert result.finish_reason in ("stop", "length")
        assert result.usage is not None
        assert 0 < result.usage.completion_tokens <= payload["max_tokens"]
        assert result.usage.prompt_tokens > 0
        assert result.usage.total_tokens == (
            result.usage.prompt_tokens + result.usage.completion_tokens
        )
        audio_duration_seconds = None
        if result.audio_bytes:
            if streaming:
                audio_samples = np.frombuffer(result.audio_bytes, dtype="<i2")
                sample_rate_hz = SAMPLE_RATE_HZ
                with wave.open(
                    str(self.output_directory / f"{case}.wav"), "wb"
                ) as audio_file:
                    audio_file.setparams(
                        (1, 2, SAMPLE_RATE_HZ, 0, "NONE", "not compressed")
                    )
                    audio_file.writeframes(result.audio_bytes)
                assert sum(result.chunk_samples) == audio_samples.size
                assert audio_samples.size % SAMPLES_PER_CODE == 0
                assert (
                    0
                    < audio_samples.size // SAMPLES_PER_CODE
                    <= result.usage.completion_tokens
                )
            else:
                (self.output_directory / f"{case}.wav").write_bytes(result.audio_bytes)
                audio_samples, sample_rate_hz = sf.read(io.BytesIO(result.audio_bytes))
            assert sample_rate_hz == SAMPLE_RATE_HZ and audio_samples.ndim == 1
            assert audio_samples.size > 0 and np.isfinite(audio_samples).all()
            assert np.any(audio_samples != 0), "Generated audio is entirely silent"
            audio_duration_seconds = audio_samples.size / sample_rate_hz
        else:
            pass
        summary = {
            "request_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
            "finish_reason": result.finish_reason,
            "usage": result.usage.model_dump(),
            "text": result.text,
            "elapsed_seconds": result.elapsed_seconds,
            "ttft_seconds": result.first_text_seconds,
            "ttfa_first_pcm_seconds": result.first_audio_seconds,
            "audio_duration_seconds": audio_duration_seconds,
            "rtf": (
                result.elapsed_seconds / audio_duration_seconds
                if audio_duration_seconds
                else None
            ),
            "chunk_samples": result.chunk_samples,
        }
        (self.output_directory / f"{case}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        return result


@pytest.fixture(scope="module")
def moss_speech_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[MossSpeechServer]:
    config_filename = os.environ.get("MOSS_SPEECH_CI_CONFIG")
    if not config_filename:
        pytest.skip(
            "Set MOSS_SPEECH_CI_CONFIG to an allocated real-checkpoint configuration"
        )
    else:
        pass
    config_path = Path(config_filename).resolve(strict=True)
    configuration = ConfigManager.from_file(str(config_path)).config
    assert isinstance(configuration, MossSpeechPipelineConfig)
    assert configuration.model_path, "Configure the real MOSS-Speech checkpoint"
    preprocessing = next(
        stage for stage in configuration.stages if stage.name == "preprocessing"
    )
    assert isinstance(preprocessing, MossSpeechPreprocessingStage)
    assert preprocessing.factory.voice_wav, "Configure the actual fixed reference WAV"
    voice_path = Path(preprocessing.factory.voice_wav).resolve(strict=True)
    output_directory = Path(
        os.environ.get(
            "MOSS_SPEECH_CI_OUTPUT_DIR", str(tmp_path_factory.mktemp("moss_speech_ci"))
        )
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    input_audio: dict[Language, str] = {}
    asset_hashes = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "voice_sha256": hashlib.sha256(voice_path.read_bytes()).hexdigest(),
    }
    for language in ("en", "zh"):
        input_filename = os.environ.get(f"MOSS_SPEECH_CI_AUDIO_{language.upper()}")
        assert (
            input_filename
        ), f"Set MOSS_SPEECH_CI_AUDIO_{language.upper()} to real {language} speech"
        input_bytes = Path(input_filename).read_bytes()
        audio_information = sf.info(io.BytesIO(input_bytes))
        assert audio_information.frames > 0 and audio_information.channels > 0
        assert 8000 <= audio_information.samplerate <= 48000
        assert audio_information.frames / audio_information.samplerate <= 30, (
            f"Input {language} speech exceeds the model's 30-second request limit; "
            "prepare a bounded speech input before starting the server"
        )
        input_audio[language] = base64.b64encode(input_bytes).decode("ascii")
        asset_hashes[f"input_{language}_sha256"] = hashlib.sha256(
            input_bytes
        ).hexdigest()
    (output_directory / "assets.json").write_text(
        json.dumps(asset_hashes, indent=2) + "\n"
    )
    (output_directory / "configuration.json").write_text(
        json.dumps(configuration.model_dump(mode="json"), indent=2) + "\n"
    )
    session = requests.Session()
    session.trust_env = False
    port = find_available_port()
    try:
        with managed_omni_server(
            model_path=configuration.model_path,
            server_config=str(config_path),
            port=port,
            host="127.0.0.1",
            log_file=output_directory / "server.log",
            timeout=900,
            wait_for_gpu_release=False,
        ):
            yield MossSpeechServer(
                f"http://127.0.0.1:{port}", session, output_directory, input_audio
            )
    finally:
        session.close()


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("input_modality", ["text", "audio"])
@pytest.mark.parametrize("output_modality", ["text", "audio"])
@pytest.mark.parametrize("streaming", [False, True], ids=["nonstream", "stream"])
def test_four_modes(
    moss_speech_server: MossSpeechServer,
    language: Language,
    input_modality: Modality,
    output_modality: Modality,
    streaming: bool,
) -> None:
    case = f"{language}-{input_modality}-to-{output_modality}-{'stream' if streaming else 'nonstream'}"
    result = moss_speech_server.complete(
        case,
        moss_speech_server.payload(
            language, input_modality, output_modality, streaming
        ),
    )
    if output_modality == "text":
        assert result.text.strip() and not result.audio_bytes
    else:
        assert result.audio_bytes


def test_natural_eos(moss_speech_server: MossSpeechServer) -> None:
    payload = moss_speech_server.payload(max_tokens=512)
    payload["messages"] = [
        {"role": "user", "content": "Reply with exactly one word: hello."}
    ]
    result = moss_speech_server.complete("natural-eos", payload)
    assert result.text.strip() and result.finish_reason == "stop"
    assert result.usage is not None and result.usage.completion_tokens < 512


def test_length_limit_flushes_audio_tail(moss_speech_server: MossSpeechServer) -> None:
    payload = moss_speech_server.payload(
        output_modality="audio", streaming=True, max_tokens=64
    )
    payload["messages"] = [
        {
            "role": "user",
            "content": "Count aloud from one to one hundred, without skipping any numbers.",
        }
    ]
    result = moss_speech_server.complete("length-audio-tail", payload)
    assert result.finish_reason == "length", "A natural stop cannot verify truncation"
    assert result.usage is not None and result.usage.completion_tokens == 64
    assert result.audio_bytes and result.chunk_samples[-1] > 0


def test_disconnect_then_followup(moss_speech_server: MossSpeechServer) -> None:
    payload = moss_speech_server.payload(
        output_modality="audio", streaming=True, max_tokens=512
    )
    with (
        moss_speech_server.session.post(
            f"{moss_speech_server.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response,
        (moss_speech_server.output_directory / "disconnect.response.txt").open(
            "w"
        ) as response_file,
    ):
        response_file.write(f"HTTP {response.status_code}\n")
        assert response.status_code == 200, response.text
        saw_audio = False
        for line in response.iter_lines(chunk_size=None, decode_unicode=True):
            if not line:
                continue
            else:
                pass
            response_file.write(f"{line}\n")
            response_file.flush()
            assert line.startswith("data: ") and line != "data: [DONE]", line
            event = ChatCompletionStreamResponse.model_validate_json(
                line.removeprefix("data: ")
            )
            assert event.choices[0].finish_reason is None
            if event.choices[0].delta.audio is not None:
                saw_audio = True
                break
            else:
                pass
        assert saw_audio
    (moss_speech_server.output_directory / "disconnect.json").write_text(
        json.dumps(
            {
                "client_closed_after_audio_delta": True,
                "server_cancellation_acknowledged": None,
            },
            indent=2,
        )
        + "\n"
    )
    result = moss_speech_server.complete(
        "after-disconnect", moss_speech_server.payload()
    )
    assert result.text.strip()


def test_invalid_request_then_followup(moss_speech_server: MossSpeechServer) -> None:
    payload = moss_speech_server.payload()
    payload["min_p"] = 0.2
    with moss_speech_server.session.post(
        f"{moss_speech_server.base_url}/v1/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        (
            moss_speech_server.output_directory / "invalid-request.response.txt"
        ).write_text(f"HTTP {response.status_code}\n{response.text}\n")
        assert response.status_code == 400 and "min_p" in response.text
    result = moss_speech_server.complete(
        "after-invalid-request", moss_speech_server.payload()
    )
    assert result.text.strip()
