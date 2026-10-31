# SPDX-License-Identifier: Apache-2.0
"""Real terminal scheduler boundaries, protocol validation and tail ownership."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from sglang_omni.models.moss_speech.payload_types import (
    EOSP_TOKEN_ID,
    MODALITY_PAD_TOKEN_ID,
    MossSpeechState,
)
from sglang_omni.models.moss_speech.request_builders import extract_output_codes
from sglang_omni.models.moss_speech.request_data import build_sglang_moss_request
from sglang_omni.models.moss_speech.streaming import (
    MossSpeechStreamingText,
    MossSpeechStreamOutputBuilder,
    resolve_stream_terminal,
)
from sglang_omni.models.moss_speech.voice_transport import (
    pack_voice_payload,
    voice_payload_digest,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.fixtures.moss_speech_codec import (
    TinyHiFT,
    TinyOfflineDecoder,
    Utf8Tokenizer,
    audio_chunk,
    audio_payload,
    drain_messages,
    vocoder,
    voice_tensors,
)


@pytest.mark.parametrize("modality", ["text", "audio"])
@pytest.mark.parametrize("stream", [False, True])
def test_stream_completion_routes_only_to_selected_terminal(
    modality: str, stream: bool
) -> None:
    payload = StagePayload(
        "route",
        OmniRequest(inputs=None, params={"stream": stream}),
        MossSpeechState(output_modality=modality).to_dict(),
    )
    assert resolve_stream_terminal("route", payload) == (
        ["audio_vocoder" if modality == "audio" else "text_decode"] if stream else []
    )


def test_audio_output_cursor_ignores_inactive_eos_and_never_replays_rows() -> None:
    packed_voice = pack_voice_payload(voice_tensors())
    state = MossSpeechState(
        output_modality="audio",
        input_grid=[[12, 512]],
        attention_mask=[1],
        voice_payload=packed_voice,
        voice_key=voice_payload_digest(packed_voice),
    )
    payload = StagePayload(
        "audio", OmniRequest(inputs=None, params={"stream": True}), state.to_dict()
    )
    request = build_sglang_moss_request(state, payload=payload)
    builder = MossSpeechStreamOutputBuilder()
    request.output_rows = [(12, EOSP_TOKEN_ID), (MODALITY_PAD_TOKEN_ID, 3)]
    first = builder.flush("audio", request)
    assert first[0].target == "audio_vocoder" and first[0].data.tolist() == [3]
    assert first[0].metadata["row_index"] == 0
    assert first[0].metadata["voice_payload"] == packed_voice
    assert builder.flush("audio", request) == []
    request.output_rows.extend(
        [
            (MODALITY_PAD_TOKEN_ID, 4),
            (MODALITY_PAD_TOKEN_ID, EOSP_TOKEN_ID),
            (MODALITY_PAD_TOKEN_ID, 5),
        ]
    )
    second = builder.flush("audio", request)
    assert second[0].data.tolist() == [4] and second[0].metadata["row_index"] == 1
    assert "voice_payload" not in second[0].metadata
    assert builder.flush("audio", request) == []


def test_text_output_keeps_both_rows_on_text_terminal() -> None:
    state = MossSpeechState(input_grid=[[12, 512]], attention_mask=[1])
    payload = StagePayload(
        "text", OmniRequest(inputs=None, params={"stream": True}), state.to_dict()
    )
    request = build_sglang_moss_request(state, payload=payload)
    request.output_rows = [(97, 512), (98, EOSP_TOKEN_ID)]
    output = MossSpeechStreamOutputBuilder().flush("text", request)
    assert output[0].target == "text_decode" and output[0].data.tolist() == [97, 98]
    assert output[0].metadata["modality"] == "text_tokens"
    assert "voice_payload" not in output[0].metadata


def test_terminal_code_extraction_ignores_inactive_eos_and_rejects_invalid_active_code() -> (
    None
):
    state = MossSpeechState(
        output_modality="audio",
        output_grid=[
            [97, EOSP_TOKEN_ID],
            [MODALITY_PAD_TOKEN_ID, 3],
            [MODALITY_PAD_TOKEN_ID, EOSP_TOKEN_ID],
            [MODALITY_PAD_TOKEN_ID, 4],
        ],
    )
    assert extract_output_codes(state) == [3]
    state.output_grid[1][1] = EOSP_TOKEN_ID + 1
    with pytest.raises(ValueError, match="generated audio code"):
        extract_output_codes(state)


def test_vocoder_done_before_payload_flushes_once_and_drops_late_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = vocoder(tmp_path, monkeypatch)
    try:
        scheduler.handle_stream_chunk("audio", audio_chunk([3, 4]))
        scheduler.handle_stream_done("audio")
        assert drain_messages(scheduler) == []
        scheduler.handle_streaming_new_request("audio", audio_payload("audio", [3, 4]))
        messages = drain_messages(scheduler)
        assert [message.type for message in messages] == ["stream", "result"]
        assert messages[-1].data.data["stream_samples"] == 3840
        assert "audio_data" not in messages[-1].data.data
        scheduler.handle_stream_chunk("audio", audio_chunk([3, 4]))
        assert drain_messages(scheduler) == []
        assert not scheduler.codec.sessions and not scheduler.stream_states
    finally:
        scheduler.stop()


@pytest.mark.parametrize("terminal", ["audio", "text"])
def test_duplicate_eos_does_not_restore_completed_pending_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str
) -> None:
    """A duplicate EOS after completion must not retain a phantom pending request."""
    if terminal == "audio":
        scheduler = vocoder(tmp_path, monkeypatch)
        chunk = audio_chunk([97])
        payload = audio_payload("completed", [97])
    else:
        scheduler = MossSpeechStreamingText(Utf8Tokenizer())
        chunk = StreamItem(
            0,
            torch.tensor([97]),
            "ar_engine",
            {"stream": True, "modality": "text_tokens", "row_index": 0},
        )
        payload = StagePayload(
            "completed",
            OmniRequest(inputs=None, params={"stream": True}),
            MossSpeechState(output_grid=[[97, 512]]).to_dict(),
        )
    try:
        scheduler.handle_stream_chunk("completed", chunk)
        scheduler.handle_streaming_new_request("completed", payload)
        scheduler.handle_stream_done("completed")
        completed_messages = drain_messages(scheduler)
        assert sum(message.type == "result" for message in completed_messages) == 1
        scheduler.handle_stream_done("completed")
        assert drain_messages(scheduler) == []
        assert "completed" not in scheduler.pending_done
        assert "completed" not in scheduler.stream_payloads
        scheduler.handle_streaming_new_request("completed", payload)
        scheduler.handle_stream_chunk("completed", chunk)
        scheduler.handle_stream_done("completed")
        reused_messages = drain_messages(scheduler)
        assert sum(message.type == "result" for message in reused_messages) == 1
        assert "completed" not in scheduler.pending_done
    finally:
        scheduler.stop()


@pytest.mark.parametrize(
    "finish_reason,code_count",
    [("stop", 511), ("length", 512), ("stop", 999), ("length", 1000)],
)
def test_terminal_audio_preserves_natural_eos_and_length_code_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish_reason: str, code_count: int
) -> None:
    scheduler = vocoder(tmp_path, monkeypatch)
    codes = list(range(code_count))
    try:
        scheduler.handle_streaming_new_request(
            "budget", audio_payload("budget", codes, finish_reason=finish_reason)
        )
        scheduler.handle_stream_chunk("budget", audio_chunk(codes))
        scheduler.handle_stream_done("budget")
        messages = drain_messages(scheduler)
        terminal = messages[-1].data.data
        assert messages[-1].type == "result"
        assert terminal["finish_reason"] == finish_reason
        assert terminal["stream_samples"] == code_count * 1920
        assert terminal["usage"]["completion_tokens"] == code_count + (
            finish_reason == "stop"
        )
        assert terminal["stream_codec_ledger"][-1]["final"] is True
        assert not scheduler.codec.sessions and not scheduler.stream_states
    finally:
        scheduler.stop()


@pytest.mark.parametrize("should_fail", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("caller_enabled,caller_tf32", [(True, False), (False, True)])
def test_offline_decode_restores_cudnn_flags_after_success_or_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    should_fail: bool,
    caller_enabled: bool,
    caller_tf32: bool,
) -> None:
    scheduler = vocoder(tmp_path, monkeypatch)
    decoder = scheduler.offline_codec.decoder
    assert isinstance(decoder, TinyOfflineDecoder)
    decoder.should_fail = should_fail
    original_decode = decoder.decode
    observed_flags: list[tuple[bool, bool, bool, bool]] = []

    def observe_decode(
        audio_token: torch.Tensor, voice: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        observed_flags.append(
            (
                torch.backends.cudnn.enabled,
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32,
            )
        )
        return original_decode(audio_token, voice)

    monkeypatch.setattr(decoder, "decode", observe_decode)
    try:
        with torch.backends.cudnn.flags(
            enabled=caller_enabled,
            benchmark=True,
            deterministic=False,
            allow_tf32=caller_tf32,
        ):
            payload = audio_payload("offline-flags", [3, 4], stream=False)
            if should_fail:
                with pytest.raises(
                    RuntimeError, match="Injected offline decoder failure"
                ):
                    scheduler.compute(payload)
            else:
                scheduler.compute(payload)
            assert observed_flags == [(caller_enabled, False, True, caller_tf32)]
            assert (
                torch.backends.cudnn.enabled,
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32,
            ) == (caller_enabled, True, False, caller_tf32)
    finally:
        scheduler.stop()


@pytest.mark.parametrize("changed_field", ["seed", "voice_key", "voice_payload"])
def test_stream_contract_rejects_voice_or_seed_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_field: str
) -> None:
    scheduler = vocoder(tmp_path, monkeypatch)
    try:
        first = audio_chunk([3, 4])
        scheduler.handle_stream_chunk("audio", first)
        metadata = copy.deepcopy(first.metadata)
        metadata["row_index"] = 2
        if changed_field == "seed":
            metadata["seed"] = 8
        elif changed_field == "voice_key":
            metadata["voice_key"] = "changed"
        else:
            changed_voice = voice_tensors()
            changed_voice["embedding"][0, 0] = 99
            metadata["voice_payload"] = pack_voice_payload(changed_voice)
        with pytest.raises(ValueError, match="changed"):
            scheduler.handle_stream_chunk(
                "audio", StreamItem(1, torch.tensor([5]), "ar_engine", metadata)
            )
        scheduler.abort("audio")
        assert not scheduler.codec.sessions and not scheduler.stream_states
    finally:
        scheduler.stop()


@pytest.mark.parametrize(
    "invalid_kind", ["cursor", "modality", "code", "dtype", "budget"]
)
def test_stream_rejects_malformed_chunks_before_consuming_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_kind: str
) -> None:
    scheduler = vocoder(tmp_path, monkeypatch)
    try:
        chunk = audio_chunk([3])
        if invalid_kind == "cursor":
            chunk.metadata["row_index"] = 1
        elif invalid_kind == "modality":
            chunk.metadata["modality"] = "text_tokens"
        elif invalid_kind == "code":
            chunk.data = torch.tensor([EOSP_TOKEN_ID])
        elif invalid_kind == "dtype":
            chunk.data = torch.tensor([3.0])
        else:
            chunk.data = torch.zeros(1001, dtype=torch.long)
        with pytest.raises(ValueError):
            scheduler.handle_stream_chunk("bad", chunk)
        scheduler.abort("bad")
        assert not scheduler.codec.sessions and not scheduler.stream_states
    finally:
        scheduler.stop()


def test_abort_releases_only_matching_codec_session_and_rejects_late_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = vocoder(tmp_path, monkeypatch)
    scheduler.handle_stream_chunk("aborted", audio_chunk([1, 2]))
    scheduler.handle_stream_chunk("kept", audio_chunk([3, 4]))
    scheduler.abort("aborted")
    scheduler.handle_stream_chunk("aborted", audio_chunk([1, 2]))
    assert set(scheduler.codec.sessions) == {"kept"}
    assert set(scheduler.stream_states) == {"kept"}
    assert drain_messages(scheduler) == []
    scheduler.stop()
    assert not scheduler.codec.sessions and not scheduler.stream_states
    assert scheduler.offline_codec.is_closed


def test_abort_during_hop_suppresses_output_and_followup_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed neural hop must not publish audio after its request was aborted."""
    scheduler = vocoder(tmp_path, monkeypatch)
    hop_started, finish_hop, abort_marked = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    assert isinstance(scheduler.codec.hift, TinyHiFT)
    original_inference = scheduler.codec.hift.inference
    original_record_abort = scheduler.record_aborted_request_id

    def blocked_inference(
        *, speech_feat: torch.Tensor, cache_source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hop_started.set()
        assert finish_hop.wait(timeout=10)
        return original_inference(speech_feat=speech_feat, cache_source=cache_source)

    def observe_abort(request_id: str) -> None:
        original_record_abort(request_id)
        abort_marked.set()

    monkeypatch.setattr(scheduler.codec.hift, "inference", blocked_inference)
    monkeypatch.setattr(scheduler, "record_aborted_request_id", observe_abort)
    try:
        with ThreadPoolExecutor(max_workers=2) as execution:
            decode = execution.submit(
                scheduler.handle_stream_chunk,
                "during-hop",
                audio_chunk(list(range(51))),
            )
            try:
                assert hop_started.wait(timeout=10)
                abort = execution.submit(scheduler.abort, "during-hop")
                assert abort_marked.wait(timeout=10)
            finally:
                finish_hop.set()
            decode.result(timeout=10)
            abort.result(timeout=10)
        assert scheduler.is_aborted("during-hop")
        assert drain_messages(scheduler) == []
        assert not scheduler.codec.sessions and not scheduler.stream_states
        monkeypatch.setattr(scheduler.codec.hift, "inference", original_inference)
        scheduler.handle_streaming_new_request(
            "followup", audio_payload("followup", [3, 4])
        )
        scheduler.handle_stream_chunk("followup", audio_chunk([3, 4]))
        scheduler.handle_stream_done("followup")
        messages = drain_messages(scheduler)
        assert [message.type for message in messages] == ["stream", "result"]
        assert messages[-1].data.data["stream_samples"] == 3840
        assert not scheduler.codec.sessions and not scheduler.stream_states
    finally:
        finish_hop.set()
        scheduler.stop()


def test_decoder_failure_emits_error_and_followup_request_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = vocoder(tmp_path, monkeypatch)
    assert isinstance(scheduler.codec.hift, TinyHiFT)
    scheduler.codec.hift.should_fail = True
    serving_thread = threading.Thread(target=scheduler.start)
    serving_thread.start()
    try:
        scheduler.inbox.put(
            IncomingMessage("bad", "new_request", audio_payload("bad", [3, 4]))
        )
        scheduler.inbox.put(IncomingMessage("bad", "stream_chunk", audio_chunk([3, 4])))
        scheduler.inbox.put(IncomingMessage("bad", "stream_done"))
        failure = scheduler.outbox.get(timeout=10)
        assert failure.type == "error" and failure.request_id == "bad"
        scheduler.codec.hift.should_fail = False
        scheduler.inbox.put(
            IncomingMessage("good", "new_request", audio_payload("good", [3, 4]))
        )
        scheduler.inbox.put(
            IncomingMessage("good", "stream_chunk", audio_chunk([3, 4]))
        )
        scheduler.inbox.put(IncomingMessage("good", "stream_done"))
        success = [scheduler.outbox.get(timeout=10), scheduler.outbox.get(timeout=10)]
        assert [message.type for message in success] == ["stream", "result"]
        assert all(message.request_id == "good" for message in success)
        assert success[-1].data.data["stream_samples"] == 3840
        assert (
            "bad" not in scheduler.codec.sessions
            and "bad" not in scheduler.stream_states
        )
    finally:
        scheduler.stop()
        serving_thread.join(timeout=10)
        assert not serving_thread.is_alive()
    assert not scheduler.codec.sessions and not scheduler.stream_states
