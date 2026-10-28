# SPDX-License-Identifier: Apache-2.0
"""Admission, cancellation and real request-owned generation cleanup."""

import asyncio
import base64
import gc
import threading
import weakref
from collections.abc import Iterator
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
import soundfile
import torch

import sglang_omni.models.moss_speech.stages as stages_module
from sglang_omni.client.types import GenerateRequest, Message
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.payload_types import MossSpeechState, OutputModality
from sglang_omni.models.moss_speech.request_builders import (
    make_moss_speech_scheduler_adapters,
    normalize_and_validate,
    terminal_result,
)
from sglang_omni.models.moss_speech.request_data import (
    MossSpeechSGLangRequestData,
    build_sglang_moss_request,
)
from sglang_omni.models.moss_speech.request_lifecycle import MossSpeechRequestLifecycle
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.types import DeferredAdmission
from tests.unit_test.fixtures.moss_speech_codec import encoder_adapter
from tests.unit_test.fixtures.moss_speech_model import native_runtime, tiny_model
from tests.unit_test.fixtures.moss_speech_requests import (
    FixtureTokenizer,
    canonical_fixture,
)


@pytest.fixture
def lifecycle() -> Iterator[MossSpeechRequestLifecycle]:
    with native_runtime(torch.device("cpu")):
        owner = MossSpeechRequestLifecycle(tiny_model(torch.device("cpu")))
        yield owner
        for request_id in list(owner.requests):
            owner.release(request_id)


def request_data(
    request_id: str, *, modality: OutputModality = "text"
) -> MossSpeechSGLangRequestData:
    state = MossSpeechState(
        input_grid=[[12, 512], [13, 512]],
        attention_mask=[1, 1],
        prompt_grid_len=2,
        effective_seed=123,
        output_modality=modality,
    )
    payload = StagePayload(request_id, OmniRequest(inputs=None), state.to_dict())
    return build_sglang_moss_request(state, payload=payload)


def test_abort_releases_only_matching_actual_request_and_model_logits(
    lifecycle: MossSpeechRequestLifecycle,
) -> None:
    aborted, kept = request_data("aborted"), request_data("kept")
    lifecycle.admit(aborted, "text")
    lifecycle.admit(kept, "text")
    aborted.output_rows = [(97, 512)]
    aborted.output_ids = [97]
    aborted.pending_feedback_queue.append(torch.ones(128))
    aborted.rng_generator = torch.Generator().manual_seed(1)
    lifecycle.model.dual_logits_by_request_id.update(
        {
            ("aborted", "prefill"): (torch.ones(1), torch.ones(1)),
            ("aborted", "decode"): (torch.ones(1), torch.ones(1)),
            ("kept", "prefill"): (torch.zeros(1), torch.zeros(1)),
        }
    )
    lifecycle.release("aborted", aborted=True)
    lifecycle.release("aborted", aborted=True)
    assert aborted.finished and not aborted.output_rows and not aborted.output_ids
    assert not aborted.pending_feedback_queue and aborted.rng_generator is None
    assert set(lifecycle.requests) == {"kept"}
    assert set(lifecycle.model.dual_logits_by_request_id) == {("kept", "prefill")}
    assert torch.equal(kept.prompt_rows, torch.tensor([[12, 512], [13, 512]]))


def test_duplicate_live_id_cannot_replace_existing_request(
    lifecycle: MossSpeechRequestLifecycle,
) -> None:
    original = request_data("duplicate")
    lifecycle.admit(original, "text")
    with pytest.raises(ValueError, match="already exists"):
        lifecycle.admit(request_data("duplicate"), "text")
    assert lifecycle.requests["duplicate"] is original
    lifecycle.release("duplicate")
    replacement = request_data("duplicate")
    assert lifecycle.admit(replacement, "text") is replacement


def test_modalities_admit_in_fifo_groups_and_cancel_pending_without_starvation(
    lifecycle: MossSpeechRequestLifecycle,
) -> None:
    initial = request_data("first")
    lifecycle.admit(initial, "text")
    audio = lifecycle.admit(request_data("audio", modality="audio"), "audio")
    cancelled = lifecycle.admit(request_data("cancelled", modality="audio"), "audio")
    text_first = lifecycle.admit(request_data("text-first"), "text")
    text_second = lifecycle.admit(request_data("text-second"), "text")
    assert all(
        isinstance(admission, DeferredAdmission)
        for admission in (audio, cancelled, text_first, text_second)
    )
    assert not audio.ready.done() and not text_first.ready.done()
    lifecycle.release("cancelled", aborted=True)
    assert cancelled.ready.cancelled()
    lifecycle.release("first")
    assert audio.ready.result(timeout=0) is None
    assert not text_first.ready.done() and not text_second.ready.done()
    lifecycle.release("audio")
    assert (
        text_first.ready.result(timeout=0) is None
        and text_second.ready.result(timeout=0) is None
    )
    delayed_audio = lifecycle.admit(
        request_data("last-audio", modality="audio"), "audio"
    )
    assert (
        isinstance(delayed_audio, DeferredAdmission) and not delayed_audio.ready.done()
    )
    lifecycle.release("text-first")
    assert not delayed_audio.ready.done()
    lifecycle.release("text-second")
    assert delayed_audio.ready.result(timeout=0) is None


def test_cancel_before_late_builder_admission_leaves_no_live_resources(
    lifecycle: MossSpeechRequestLifecycle,
) -> None:
    """A builder finishing after cancellation must not resurrect its request."""
    reached_admission = threading.Event()
    continue_admission = threading.Event()
    built = request_data("late")

    def late_builder() -> (
        MossSpeechSGLangRequestData | DeferredAdmission[MossSpeechSGLangRequestData]
    ):
        reached_admission.set()
        assert continue_admission.wait(timeout=10)
        return lifecycle.admit(built, "text")

    with ThreadPoolExecutor(max_workers=1) as builders:
        future = builders.submit(late_builder)
        assert reached_admission.wait(timeout=10)
        lifecycle.release("late", aborted=True)
        continue_admission.set()
        with pytest.raises(CancelledError):
            future.result(timeout=10)
    assert lifecycle.snapshot() == {"requests": 0, "active": 0, "pending": 0}
    assert not lifecycle.model.dual_logits_by_request_id


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_result_adapter_copies_grid_and_finish_reason_before_cleanup(
    lifecycle: MossSpeechRequestLifecycle, finish_reason: str
) -> None:
    build, adapt = make_moss_speech_scheduler_adapters(lifecycle=lifecycle)
    source = request_data("finished")
    assert source.stage_payload is not None
    built = build(source.stage_payload)
    assert isinstance(built, MossSpeechSGLangRequestData)
    generated_rows = [(97, 512), (98, 512)]
    built.output_rows = list(generated_rows)
    built.finish_reason = finish_reason
    built.rng_generator = torch.Generator().manual_seed(1)
    lifecycle.model.dual_logits_by_request_id[("finished", "prefill")] = (
        torch.ones(1),
        torch.ones(1),
    )
    result = adapt(built)
    restored = MossSpeechState.from_dict(result.data)
    assert restored.output_grid == [list(row) for row in generated_rows]
    assert restored.finish_reason == finish_reason
    assert terminal_result(restored)["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
    }
    assert not lifecycle.requests and not lifecycle.model.dual_logits_by_request_id
    assert built.finished and not built.output_rows and built.rng_generator is None


def test_result_adapter_failure_still_releases_its_actual_owner(
    lifecycle: MossSpeechRequestLifecycle,
) -> None:
    _, adapt = make_moss_speech_scheduler_adapters(lifecycle=lifecycle)
    built = request_data("invalid-result")
    lifecycle.admit(built, "text")
    assert built.stage_payload is not None
    built.stage_payload.data = "invalid wire state"
    lifecycle.model.dual_logits_by_request_id[("invalid-result", "decode")] = (
        torch.ones(1),
        torch.ones(1),
    )
    with pytest.raises(TypeError, match="stage data"):
        adapt(built)
    assert not lifecycle.requests and not lifecycle.model.dual_logits_by_request_id
    assert built.finished


def test_retry_keeps_prompt_seed_and_restarts_generated_history(
    lifecycle: MossSpeechRequestLifecycle,
) -> None:
    original = request_data("retry")
    lifecycle.admit(original, "text")
    expected = torch.rand(
        6, generator=MossSpeechModelRunner.generator_for(original, torch.device("cpu"))
    )
    original.output_rows.append((97, 512))
    original.stream_grid_cursor = original.stream_token_cursor = 1
    original.pending_feedback_queue.append(torch.ones(128))
    lifecycle.release("retry")
    retry = request_data("retry")
    lifecycle.admit(retry, "text")
    actual = torch.rand(
        6, generator=MossSpeechModelRunner.generator_for(retry, torch.device("cpu"))
    )
    assert torch.equal(actual, expected)
    assert torch.equal(original.prompt_rows, retry.prompt_rows)
    assert original.effective_seed == retry.effective_seed == 123
    assert not original.output_rows and not original.pending_feedback_queue
    assert original.stream_grid_cursor == original.stream_token_cursor == 0


@pytest.mark.parametrize("outcome", ["success", "failure", "abort"])
def test_preprocessing_enrolls_voice_once_consumes_audio_and_releases_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Decoded waveforms and per-request state must not survive preprocessing."""
    adapter = encoder_adapter(tmp_path / "codec", monkeypatch)
    voice_path = tmp_path / "voice.wav"
    soundfile.write(
        voice_path,
        torch.sin(torch.arange(8000, dtype=torch.float32) / 20).numpy(),
        16000,
        subtype="FLOAT",
    )
    enroll_voice = Mock(wraps=adapter.encode_voice_ref)
    monkeypatch.setattr(adapter, "encode_voice_ref", enroll_voice)
    monkeypatch.setattr(
        stages_module,
        "load_tokenizer",
        Mock(return_value=FixtureTokenizer(canonical_fixture().text_encodings)),
    )
    monkeypatch.setattr(
        stages_module, "MossSpeechCodecAdapter", Mock(return_value=adapter)
    )
    state_references: list[weakref.ReferenceType[MossSpeechState]] = []
    waveform_references: list[weakref.ReferenceType[torch.Tensor]] = []

    def observe(
        request: OmniRequest, *, request_id: str, allow_streaming: bool
    ) -> MossSpeechState:
        state = normalize_and_validate(
            request, request_id=request_id, allow_streaming=allow_streaming
        )
        state_references.append(weakref.ref(state))
        waveform_references.extend(
            weakref.ref(waveform) for waveform, _, _ in state.decoded_audio
        )
        if outcome == "abort":
            scheduler.abort(request_id)
        else:
            pass
        return state

    monkeypatch.setattr(stages_module, "normalize_and_validate", observe)
    scheduler = stages_module.create_preprocessing_executor(
        str(tmp_path),
        codec_path=str(tmp_path / "codec"),
        voice_wav=str(voice_path),
        encode_batch_size=2,
        context_limit=1 if outcome == "failure" else 512,
        device="cpu",
    )
    encoded = base64.b64encode(voice_path.read_bytes()).decode("ascii")
    loop = asyncio.new_event_loop()
    try:
        for request_index in range(2):
            request_id = f"preprocess-{request_index}"
            request = OmniRequest(
                inputs=GenerateRequest(
                    messages=[
                        Message(
                            "user",
                            [{"type": "input_audio", "input_audio": {"data": encoded}}],
                        )
                    ],
                    output_modalities=["audio"],
                    stream=False,
                )
            )
            incoming = IncomingMessage(
                request_id, "new_request", StagePayload(request_id, request, None)
            )
            if outcome == "failure":
                with pytest.raises(ValueError, match="Canonical prompt"):
                    scheduler.run_single(incoming, loop)
            else:
                scheduler.run_single(incoming, loop)
            if outcome == "success":
                result = scheduler.outbox.get_nowait()
                assert result.type == "result"
                assert "decoded_audio" not in result.data.data
                assert result.data.data["audio_codes"][0]
                assert isinstance(
                    result.data.data["voice_payload"]["embedding_bytes"], bytes
                )
            else:
                assert scheduler.outbox.empty()
    finally:
        scheduler.stop()
        loop.close()
    gc.collect()
    assert len(state_references) == len(waveform_references) == 2
    assert all(
        reference() is None for reference in state_references + waveform_references
    )
    assert enroll_voice.call_count == 1
    assert (
        adapter.is_closed
        and adapter.encoder is None
        and adapter.speaker_encoder is None
    )
