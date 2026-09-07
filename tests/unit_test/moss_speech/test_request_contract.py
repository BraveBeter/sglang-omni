"""HTTP->canonical contract tests for MOSS-Speech (T2.3, GPU-free).

Drives the REAL chat lowering (`_build_chat_generate_request`) and the model
normalize/validate entry. Includes a no-codec-import spy: a poisoned alias
for the codec adapter module proves rejected/normalized requests never touch
codec code paths.
"""

from __future__ import annotations

import base64
import io
import sys
import types
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from sglang_omni.models.moss_speech import request_builders as rb
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest


def _wav_bytes(seconds: float = 0.5, sr: int = 16000, nan: bool = False) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    wav = 0.1 * np.sin(2 * np.pi * 220 * t)
    if nan:
        wav[len(wav) // 2] = np.nan
    buf = io.BytesIO()
    # FLOAT subtype keeps NaN through the round trip (PCM quantization would
    # silently sanitize it), exercising the non-finite guard.
    subtype = "FLOAT" if nan else "PCM_16"
    sf.write(buf, wav.astype(np.float32), sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def _b64(seconds: float = 0.5, sr: int = 16000, nan: bool = False) -> str:
    return base64.b64encode(_wav_bytes(seconds, sr, nan)).decode()


def _audio_part(data_b64: str) -> dict[str, Any]:
    return {"type": "input_audio", "input_audio": {"data": data_b64, "format": "wav"}}


def _lower(**kwargs) -> OmniRequest:
    kwargs.setdefault("messages", [{"role": "user", "content": "hello"}])
    req = ChatCompletionRequest(model="moss-speech", **kwargs)
    gen = _build_chat_generate_request(req)
    return OmniRequest(inputs=gen)


def _normalize(omni: OmniRequest, request_id: str = "req-test") -> MossSpeechState:
    return rb.normalize_and_validate(omni, request_id=request_id)


# ------------------------------------------------------------------- valid
def test_valid_text_request_defaults_to_text_modality() -> None:
    state = _normalize(_lower())
    assert state.output_modality == "text"
    assert state.turns == [{"role": "user", "kind": "text", "text": "hello"}]
    assert state.stop == []


def test_valid_audio_request_with_inline_part() -> None:
    omni = _lower(modalities=["audio"], messages=[
        {"role": "user", "content": [_audio_part(_b64())]}
    ])
    state = _normalize(omni)
    assert state.output_modality == "audio"
    assert state.turns[0]["kind"] == "audio"
    audio = rb.pop_decoded_audio(state)
    assert len(audio) == 1 and audio[0][1] == 16000


def test_valid_mixed_multiturn_prefers_inline_parts() -> None:
    omni = _lower(messages=[
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": [_audio_part(_b64(0.3))]},
    ])
    state = _normalize(omni)
    assert [t["kind"] for t in state.turns] == ["text", "text", "audio"]


def test_audios_list_binds_one_to_one_in_order(tmp_path) -> None:
    p1 = tmp_path / "a1.wav"
    p2 = tmp_path / "a2.wav"
    p1.write_bytes(_wav_bytes(0.2, sr=24000))
    p2.write_bytes(_wav_bytes(0.2, sr=44100))
    omni = _lower(messages=[
        {"role": "user", "content": [_audio_part(_b64(0.1))]},  # placeholder replaced below
    ])
    # rebuild with audios[] instead of inline parts
    gen = _build_chat_generate_request(ChatCompletionRequest(
        model="moss-speech",
        messages=[
            {"role": "user", "content": "q1"},
            {"role": "user", "content": "q2"},  # no audio turns -> mismatch
        ],
        audios=[str(p1)],
    ))
    with pytest.raises(rb.RequestValidationError, match="audio turns"):
        _normalize(OmniRequest(inputs=gen))

    gen2 = _build_chat_generate_request(ChatCompletionRequest(
        model="moss-speech",
        messages=[
            {"role": "user", "content": [_audio_part(_b64(0.1))]},
            {"role": "user", "content": [_audio_part(_b64(0.1))]},
        ],
        audios=[str(p1), str(p2)],
    ))
    with pytest.raises(rb.RequestValidationError, match="not both"):
        _normalize(OmniRequest(inputs=gen2))


# ------------------------------------------------------------- modality rules
@pytest.mark.parametrize("modalities,ok", [
    (["text"], True), (["audio"], True),
    (["text", "audio"], False), (["audio", "audio"], False),
    (["video"], False), (["TEXT"], False), (None, True),
])
def test_modality_matrix(modalities, ok) -> None:
    kwargs = {} if modalities is None else {"modalities": modalities}
    omni = _lower(**kwargs)
    if ok:
        state = _normalize(omni)
        assert state.output_modality in ("text", "audio")
    else:
        with pytest.raises(rb.RequestValidationError, match="modalities"):
            _normalize(omni)


# --------------------------------------------------------- V1 rejections
def test_stream_rejected() -> None:
    with pytest.raises(rb.RequestValidationError, match="streaming"):
        _normalize(_lower(stream=True))


def test_request_voice_rejected() -> None:
    with pytest.raises(rb.RequestValidationError, match="voice"):
        _normalize(_lower(audio={"voice": "alloy", "format": "wav"}))


def test_image_parts_rejected() -> None:
    omni = _lower(messages=[{"role": "user", "content": [
        {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "x"}}
    ]}])
    with pytest.raises(rb.RequestValidationError, match="not supported in V1"):
        _normalize(omni)


def test_mixed_modality_in_single_turn_rejected() -> None:
    omni = _lower(messages=[{"role": "user", "content": [
        {"type": "text", "text": "hi"}, _audio_part(_b64(0.1))
    ]}])
    with pytest.raises(rb.RequestValidationError, match="single modality"):
        _normalize(omni)


def test_two_audio_parts_in_one_message_rejected() -> None:
    omni = _lower(messages=[{"role": "user", "content": [
        _audio_part(_b64(0.1)), _audio_part(_b64(0.1))
    ]}])
    with pytest.raises(rb.RequestValidationError, match="multiple input_audio"):
        _normalize(omni)


def test_unknown_role_rejected() -> None:
    gen = _build_chat_generate_request(ChatCompletionRequest(
        model="m", messages=[{"role": "tool", "content": "x"}]
    ))
    with pytest.raises(rb.RequestValidationError, match="role"):
        _normalize(OmniRequest(inputs=gen))


# ------------------------------------------------------------- audio validity
def test_undecodable_audio_rejected() -> None:
    omni = _lower(messages=[{"role": "user", "content": [
        _audio_part(base64.b64encode(b"not-audio").decode())
    ]}])
    with pytest.raises(rb.RequestValidationError, match="could not be decoded"):
        _normalize(omni)


def test_non_finite_audio_rejected() -> None:
    omni = _lower(messages=[{"role": "user", "content": [_audio_part(_b64(nan=True))]}])
    with pytest.raises(rb.RequestValidationError, match="non-finite"):
        _normalize(omni)


def test_oversize_duration_rejected() -> None:
    omni = _lower(messages=[{"role": "user", "content": [_audio_part(_b64(seconds=31.0))]}])
    with pytest.raises(rb.RequestValidationError, match="duration"):
        _normalize(omni)


def test_url_audio_rejected() -> None:
    omni = _lower(messages=[{"role": "user", "content": [
        _audio_part("https://example.com/x.wav")]}])
    with pytest.raises(rb.RequestValidationError, match="URL audio"):
        _normalize(omni)


# ------------------------------------------------------------- params & seed
def test_explicit_seed_wins_and_absent_seed_is_stable() -> None:
    s1 = _normalize(_lower(seed=42), request_id="rid-A")
    assert s1.effective_seed == 42
    s2 = _normalize(_lower(), request_id="rid-A")
    s3 = _normalize(_lower(), request_id="rid-A")
    s4 = _normalize(_lower(), request_id="rid-B")
    assert s2.effective_seed == s3.effective_seed  # stable per request_id
    assert s2.effective_seed != s4.effective_seed  # differs across ids
    assert s2.effective_seed != s1.effective_seed


def test_explicit_params_flow_through_gap1() -> None:
    omni = _lower(max_tokens=17, temperature=0.9)
    state = _normalize(omni)
    assert "max_tokens" in state.explicit_params
    assert "temperature" in state.explicit_params
    assert state.max_new_tokens == 17


def test_endpoint_defaults_not_marked_explicit() -> None:
    state = _normalize(_lower())  # nothing set by the user
    assert state.explicit_params == []
    assert state.temperature == 1.0  # endpoint fill-in value, unmarked


# ------------------------------------------------------------------ no-GPU spy
def test_normalize_never_imports_codec_adapter(monkeypatch) -> None:
    """Poison the codec adapter module alias; normalize must not import it."""
    def _booby_trap(name):
        raise AssertionError(f"codec module {name} imported during normalize")

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def guarded_import(name, *a, **k):
        if name.startswith("sglang_omni.models.moss_speech.components.codec_adapter"):
            _booby_trap(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    _normalize(_lower(modalities=["audio"], messages=[
        {"role": "user", "content": [_audio_part(_b64())]}
    ]))  # must not raise


# -------------------------------------------------------------------- routing
def test_route_fn_selects_exactly_one_terminal() -> None:
    st = MossSpeechState(output_modality="text")
    assert rb.resolve_output_terminal("r1", types.SimpleNamespace(data=st)) == "text_decode"
    st2 = MossSpeechState(output_modality="audio")
    assert rb.resolve_output_terminal("r2", types.SimpleNamespace(data=st2)) == "audio_vocoder"
    assert rb.resolve_output_terminal("r3", types.SimpleNamespace(data={"output_modality": "audio"})) == "audio_vocoder"
    with pytest.raises(rb.RequestValidationError):
        rb.resolve_output_terminal("r4", types.SimpleNamespace(data=MossSpeechState(output_modality="both")))


# -------------------------------------------------------------------- cleanup
def test_cleanup_idempotent_and_stage_scoped() -> None:
    rb.remember_prepared("p1", MossSpeechState())
    rb.remember_vocoder_session("p1")
    rb.cleanup_preprocessing_state("p1")
    rb.cleanup_preprocessing_state("p1")  # idempotent
    assert "p1" not in rb._PREPARED_REQUESTS
    assert "p1" in rb._VOCODER_SESSIONS  # preprocessing cleanup does not touch vocoder scope
    rb.cleanup_vocoder_state("p1")
    rb.cleanup_vocoder_state("p1")
    assert "p1" not in rb._VOCODER_SESSIONS
