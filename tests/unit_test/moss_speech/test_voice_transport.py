"""Exact compact voice transport and legacy streaming contract compatibility."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import msgpack
import pytest
import torch

from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.models.moss_speech.voice_transport import (
    pack_voice_payload,
    state_voice_payload,
    unpack_voice_payload,
    voice_payload_digest,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.moss_speech.test_streaming_codec import codec
from tests.unit_test.moss_speech.test_streaming_pipeline import audio_item, payload


def voice():
    feature = torch.linspace(-1.0, 1.0, 640, dtype=torch.float32).reshape(1, 8, 80)
    feature.flatten()[:4] = torch.tensor(
        [0, -2147483648, 1065353217, 8388609], dtype=torch.int32
    ).view(torch.float32)
    return dict(
        prompt_token=torch.tensor([[1, 16383]], dtype=torch.int32),
        prompt_feat=feature,
        embedding=torch.linspace(-2.0, 2.0, 192).reshape(1, 192),
    )


def state_from_voice(value):
    return MossSpeechState(
        output_modality="audio",
        voice_key="fixed",
        effective_seed=7,
        voice_token_ids=value["prompt_token"][0].tolist(),
        voice_feat=value["prompt_feat"][0].tolist(),
        voice_embedding=value["embedding"][0].tolist(),
    )


def test_compact_roundtrip_preserves_all_fp32_bits_and_control_plane():
    original = voice()
    packed = pack_voice_payload(original)
    assert set(packed) == {
        f"{name}_{suffix}"
        for name in original
        for suffix in ("bytes", "shape", "dtype")
    }
    assert all(isinstance(packed[f"{name}_bytes"], bytes) for name in original)
    transported = msgpack.unpackb(msgpack.packb(packed, use_bin_type=True), raw=False)
    restored = unpack_voice_payload(transported)
    for name, value in original.items():
        assert torch.equal(
            value.contiguous().view(torch.uint8),
            restored[name].contiguous().view(torch.uint8),
        )
    restored["prompt_feat"].zero_()
    assert torch.equal(
        unpack_voice_payload(packed)["prompt_feat"].view(torch.int32),
        original["prompt_feat"].view(torch.int32),
    )


def test_state_forwards_bytes_without_decoding_and_legacy_is_equivalent():
    original = voice()
    legacy = state_from_voice(original)
    compact = MossSpeechState(
        output_modality="audio",
        voice_key="fixed",
        effective_seed=7,
        voice_payload=pack_voice_payload(original),
    )
    stored = compact.to_dict()
    restored = MossSpeechState.from_dict(stored)
    assert restored.voice_payload is compact.voice_payload
    assert restored.voice_feat == []
    assert restored.voice_token_ids == []
    assert restored.voice_embedding == []
    assert voice_payload_digest(state_voice_payload(legacy)) == voice_payload_digest(
        state_voice_payload(compact)
    )
    assert (
        msgpack.unpackb(msgpack.packb(stored, use_bin_type=True), raw=False)[
            "voice_payload"
        ]
        == compact.voice_payload
    )


@pytest.mark.parametrize("change", ["truncated", "dtype", "shape", "missing", "extra"])
def test_malformed_typed_voice_is_rejected(change):
    packed = pack_voice_payload(voice())
    if change == "truncated":
        packed["prompt_feat_bytes"] = packed["prompt_feat_bytes"][:-1]
    elif change == "dtype":
        packed["prompt_feat_dtype"] = "float64"
    elif change == "shape":
        packed["prompt_feat_shape"] = [1, 80, 8]
    elif change == "missing":
        packed.pop("embedding_shape")
    else:
        packed["unexpected"] = "value"
    with pytest.raises((ValueError, TypeError), match="voice|Voice"):
        unpack_voice_payload(packed)


def test_preprocessing_packs_fixed_voice_once(monkeypatch):
    from sglang_omni.models.moss_speech import stages

    value = voice()
    encoded = SimpleNamespace(**value, meta={"revision": "test"})
    monkeypatch.setattr(stages, "_resolve_codec_dir", lambda *args: ".")
    monkeypatch.setattr(stages, "_load_tokenizer", lambda *args: None)
    monkeypatch.setattr(stages.os.path, "isfile", lambda *args: True)
    monkeypatch.setattr(
        stages,
        "MossSpeechCodecAdapter",
        lambda *args, **kwargs: SimpleNamespace(encode_voice_ref=lambda _: encoded),
    )
    monkeypatch.setattr(
        stages,
        "MossSpeechGridProcessor",
        lambda _: SimpleNamespace(
            build=lambda *args: SimpleNamespace(
                grid=torch.tensor([[[1, 512]]]),
                attention_mask=torch.ones(1, 1, dtype=torch.long),
                prompt_len=1,
            )
        ),
    )
    monkeypatch.setattr(
        stages,
        "normalize_and_validate",
        lambda *args, **kwargs: MossSpeechState(output_modality="audio"),
    )
    monkeypatch.setattr(stages, "pop_decoded_audio", lambda state: [])
    calls = []
    original = stages.pack_voice_payload

    def pack(value):
        calls.append(1)
        return original(value)

    monkeypatch.setattr(stages, "pack_voice_payload", pack)
    scheduler = stages.create_preprocessing_executor(".", voice_wav="voice.wav")
    outputs = [
        scheduler._fn(StagePayload(str(i), OmniRequest(inputs={}), {})).data
        for i in range(2)
    ]
    assert len(calls) == 1
    assert outputs[0]["voice_payload"] is outputs[1]["voice_payload"]
    assert all(
        out["voice_feat"] == []
        and out["voice_token_ids"] == []
        and out["voice_embedding"] == []
        for out in outputs
    )


def test_stream_builder_passes_compact_bytes_only_on_first_chunk():
    from sglang_omni.models.moss_speech.streaming import make_stream_output_builder

    state = MossSpeechState(
        output_modality="audio",
        voice_key="fixed",
        effective_seed=7,
        voice_payload=pack_voice_payload(voice()),
    )
    data = SimpleNamespace(
        stage_payload=StagePayload(
            "r", OmniRequest(inputs={}, params={"stream": True}), state.to_dict()
        ),
        req=SimpleNamespace(inflight_middle_chunks=0),
        output_rows=[[151667, 3]],
    )
    build = make_stream_output_builder()
    first = build("r", data, None)[0]
    assert first.metadata["voice_payload"] is state.voice_payload
    assert "voice" not in first.metadata
    data.output_rows.append([151667, 4])
    second = build("r", data, None)[0]
    assert "voice_payload" not in second.metadata and "voice" not in second.metadata


def compact_item(codes, index=0, value=None):
    item = audio_item(codes, index=index)
    legacy = item.metadata.pop("voice")
    item.metadata["voice_payload"] = pack_voice_payload(
        value
        or {
            key: torch.tensor(
                value, dtype=torch.int32 if key == "prompt_token" else torch.float32
            )
            for key, value in legacy.items()
        }
    )
    return item


@pytest.mark.parametrize("changed", ["voice", "seed", "key"])
def test_stream_compact_voice_or_seed_change_is_rejected(changed):
    from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingVocoder

    scheduler = MossSpeechStreamingVocoder(codec(), None)
    scheduler.on_stream_chunk("r", compact_item([3]))
    next_item = compact_item([4], index=1)
    if changed == "voice":
        packed = next_item.metadata["voice_payload"]
        raw = packed["embedding_bytes"]
        packed["embedding_bytes"] = bytes([raw[0] ^ 1]) + raw[1:]
    elif changed == "seed":
        next_item.metadata["seed"] += 1
    else:
        next_item.metadata["voice_key"] = "other"
    with pytest.raises(ValueError, match="voice|Voice|seed"):
        scheduler.on_stream_chunk("r", next_item)
    scheduler.abort("r")
    assert not scheduler.codec.sessions and not scheduler._stream_states


def test_compact_and_legacy_contracts_match_and_cancel_releases():
    from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingVocoder

    scheduler = MossSpeechStreamingVocoder(codec(), None)
    scheduler.on_stream_chunk("r", compact_item([3]))
    scheduler.on_stream_chunk("r", audio_item([4], index=1))
    state = scheduler._stream_states["r"]
    scheduler.latch_stream_contract("r", state, payload(), origin="payload")
    compact = payload()
    compact.data["voice_payload"] = copy.deepcopy(
        compact_item([3]).metadata["voice_payload"]
    )
    for name in ("voice_token_ids", "voice_feat", "voice_embedding"):
        compact.data[name] = []
    scheduler.latch_stream_contract("r", state, compact, origin="payload")
    scheduler.abort("r")
    assert not scheduler.codec.sessions and not scheduler._stream_states
    assert scheduler.on_stream_chunk("r", compact_item([5], index=2)) == []


@pytest.mark.parametrize("first_compact", [False, True])
def test_legacy_sub_fp32_voice_change_is_still_rejected(first_compact):
    from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingVocoder

    scheduler = MossSpeechStreamingVocoder(codec(), None)
    first = compact_item([3]) if first_compact else audio_item([3])
    scheduler.on_stream_chunk("r", first)
    changed = audio_item([4], index=1)
    # The old JSON contract distinguished values even if FP32 rounds them alike.
    changed.metadata["voice"]["embedding"][0][0] = 1e-50
    with pytest.raises(ValueError, match="voice|Voice"):
        scheduler.on_stream_chunk("r", changed)
    scheduler.abort("r")
