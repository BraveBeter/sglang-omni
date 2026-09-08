"""Streaming routes and delta accounting before GPU integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_speech.config import MossSpeechStreamingPipelineConfig
from sglang_omni.models.moss_speech.streaming import (
    make_stream_output_builder,
    resolve_stream_terminal,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


def payload(modality="audio", stream=True):
    return StagePayload(
        request_id="r",
        request=OmniRequest(inputs={}, params={"stream": stream}),
        data={
            "output_modality": modality,
            "effective_seed": 7,
            "voice_key": "fixed",
            "voice_token_ids": [1, 2],
            "voice_feat": [[0.0] * 80] * 8,
            "voice_embedding": [0.0] * 192,
        },
    )


def test_only_selected_terminal_gets_stream_done():
    assert resolve_stream_terminal("r", payload()) == ["audio_vocoder"]
    assert resolve_stream_terminal("r", payload("text")) == ["text_decode"]
    assert resolve_stream_terminal("r", payload(stream=False)) == []


def test_config_explicitly_opts_in_without_changing_offline():
    from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig

    offline = MossSpeechPipelineConfig(model_path="local")
    streamed = MossSpeechStreamingPipelineConfig(model_path="local")
    ar = next(s for s in streamed.stages if s.name == "ar_engine")
    assert set(ar.stream_to) == {"text_decode", "audio_vocoder"}
    assert ar.stream_done_to_fn
    assert not next(s for s in offline.stages if s.name == "ar_engine").stream_to
    assert all(
        s.can_accept_stream_before_payload for s in streamed.stages if s.terminal
    )
    assert next(s for s in streamed.stages if s.name == "audio_vocoder").env == {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }
    assert all(not s.env for s in streamed.stages if s.name != "audio_vocoder")
    assert all(not s.env for s in offline.stages)


def test_audio_cursor_ignores_inactive_eosp_and_never_repeats_rows():
    build = make_stream_output_builder()
    data = SimpleNamespace(
        stage_payload=payload(),
        req=SimpleNamespace(inflight_middle_chunks=0),
        output_rows=[[42, 16384], [151667, 3], [151667, 4]],
    )
    first = build("r", data, None)
    assert len(first) == 1 and first[0].data.tolist() == [3, 4]
    assert first[0].metadata["row_index"] == 0
    assert "voice" in first[0].metadata
    assert build("r", data, None) == []
    data.output_rows.extend([[151667, 5], [151667, 16384], [151667, 6]])
    second = build("r", data, None)
    assert second[0].data.tolist() == [5]
    assert second[0].metadata["row_index"] == 2
    assert "voice" not in second[0].metadata
    assert build.flush("r", data) == []


def test_text_rows_stay_on_text_terminal():
    build = make_stream_output_builder()
    data = SimpleNamespace(
        stage_payload=payload("text"),
        req=SimpleNamespace(inflight_middle_chunks=0),
        output_rows=[[42, 7], [8, 16384]],
    )
    msg = build("r", data, None)[0]
    assert msg.target == "text_decode" and msg.data.tolist() == [42, 8]
    assert msg.metadata["modality"] == "text_tokens"


def test_streaming_request_validation_preserves_capability_boundary():
    from sglang_omni.models.moss_speech.config import MossSpeechPipelineConfig
    from sglang_omni.serve.protocol import ChatCompletionRequest

    req = ChatCompletionRequest(
        model="moss",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        modalities=["audio"],
        audio={"format": "pcm"},
    )
    with pytest.raises(ValueError, match="streaming"):
        MossSpeechPipelineConfig.validate_chat_request(req)
    MossSpeechStreamingPipelineConfig.validate_chat_request(req)
    req.audio = {"voice": "uploaded"}
    with pytest.raises(ValueError, match="audio|voice"):
        MossSpeechStreamingPipelineConfig.validate_chat_request(req)


def audio_item(codes, index=0, first=True):
    from sglang_omni.pipeline.stage.stream_queue import StreamItem

    metadata = {"stream": True, "modality": "audio_codes", "row_index": index}
    if first:
        p = payload().data
        metadata.update(
            voice_key=p["voice_key"],
            seed=p["effective_seed"],
            voice={
                "prompt_token": [p["voice_token_ids"]],
                "prompt_feat": [p["voice_feat"]],
                "embedding": [p["voice_embedding"]],
            },
        )
    return StreamItem(index, torch.tensor(codes), "ar_engine", metadata)


def test_vocoder_done_before_payload_flushes_once_and_drops_late_chunks():
    import numpy as np

    from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingVocoder
    from tests.unit_test.moss_speech.test_streaming_codec import codec

    c = codec()
    s = MossSpeechStreamingVocoder(c, None)
    s._on_chunk("r", audio_item([3] * 28))
    s._on_done("r")
    p = payload()
    p.data["output_grid"] = [[151667, 3]] * 28 + [[151667, 16384]]
    s._handle_streaming_new_request("r", p)
    out = []
    while not s.outbox.empty():
        out.append(s.outbox.get_nowait())
    streams = [m for m in out if m.type == "stream"]
    results = [m for m in out if m.type == "result"]
    assert (
        sum(
            np.frombuffer(m.data["audio_waveform"], dtype="float32").size
            for m in streams
        )
        == 28 * 1920
    )
    assert len(results) == 1 and "audio_data" not in results[0].data.data
    assert not s._stream_states and not c.sessions and not s._pending_done
    assert s.on_stream_chunk("r", audio_item([4], index=28, first=False)) == []
    assert not c.sessions


def test_abort_cleans_codec_and_rejects_late_work(monkeypatch):
    from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingVocoder
    from tests.unit_test.moss_speech.test_streaming_codec import codec

    receipts = []
    monkeypatch.setattr(
        "sglang_omni.models.moss_speech.streaming._emit_event",
        lambda **event: receipts.append(event),
    )
    c = codec()
    s = MossSpeechStreamingVocoder(c, None)
    s._on_chunk("r", audio_item([3] * 16))
    assert c.sessions
    s.abort("r")
    s.abort("r")
    assert not c.sessions and not s._stream_states
    assert len(receipts) == 1
    assert receipts[0]["event_name"] == "moss_codec_session_released"
    assert receipts[0]["metadata"]["session_active"] is False
    assert s.on_stream_chunk("r", audio_item([3] * 16)) == []
    s._on_chunk("new", audio_item([4] * 16))
    assert "new" in c.sessions
    s.stop()
    assert not c.sessions


def test_repeated_voice_metadata_must_be_immutable():
    from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingVocoder
    from tests.unit_test.moss_speech.test_streaming_codec import codec

    s = MossSpeechStreamingVocoder(codec(), None)
    s.on_stream_chunk("r", audio_item([3]))
    changed = audio_item([4], index=1)
    changed.metadata["voice"]["embedding"][0][0] = 1.0
    with pytest.raises(ValueError, match="voice|Voice"):
        s.on_stream_chunk("r", changed)
    s.abort("r")


def test_text_utf8_hold_final_prefix_and_shutdown_cleanup():
    from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingText
    from sglang_omni.pipeline.stage.stream_queue import StreamItem

    tokenizer = SimpleNamespace(
        decode=lambda tokens, **kw: {1: "你\ufffd", 2: "你好", 3: "你好!"}[len(tokens)]
    )

    def offline(p):
        p.data.update(
            text=tokenizer.decode(p.data["output_grid"]),
            generated_text="",
            usage={"completion_tokens": len(p.data["output_grid"])},
        )
        return p

    s = MossSpeechStreamingText(tokenizer, offline)

    def item(tokens, index):
        return StreamItem(
            index,
            torch.tensor(tokens),
            "ar_engine",
            {"stream": True, "modality": "text_tokens", "row_index": index},
        )

    assert s.on_stream_chunk("r", item([1], 0)) == []
    assert s.on_stream_chunk("r", item([2], 1))[0].data["text"] == "你好"
    assert s.on_stream_chunk("r", item([3], 2))[0].data["text"] == "!"
    p = payload("text")
    p.data["output_grid"] = [[1, 512], [2, 512], [3, 512]]
    s._stream_payloads["r"] = p
    assert s.on_stream_done("r")[0].data.data["text"] == "你好!"
    s.clear_stream_state("r")
    assert s.on_stream_chunk("partial", item([1], 0)) == []
    p = payload("text")
    p.data["output_grid"] = [[1, 512]]
    s._stream_payloads["partial"] = p
    result = s.on_stream_done("partial")[0].data.data
    assert result["text"] == "你" and result["usage"]["completion_tokens"] == 1
    s.stop()
    assert not s.states
