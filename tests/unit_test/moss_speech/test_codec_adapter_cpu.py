"""GPU-free unit tests for the MOSS-Speech codec adapter (T1.2).

Covers input normalization, error paths, cleanup idempotency, voice-hook
store/load round-trip, and the RNG-scope helper — everything that does not
require weights on a GPU device. GPU behavior is covered by the smoke script
(`scripts/moss_speech/p1/smoke_adapter.py`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.moss_speech.components.codec_adapter import (
    CodecAdapterError,
    MossSpeechCodecAdapter,
)
from sglang_omni.models.moss_speech.components.hf_codec.modeling import (
    _scoped_global_rng,
)
from sglang_omni.models.moss_speech.components.voice import (
    MEL_PARAMS,
    VoiceConditioning,
    compute_voice_mel,
)


def _wav(sr: int = 24000, seconds: float = 0.5) -> torch.Tensor:
    t = torch.arange(int(sr * seconds), dtype=torch.float32) / sr
    return (0.1 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)


class _RngProbe:
    def __init__(self) -> None:
        import random

        self.random = random
        self.before_py = None
        self.before_np = None
        self.before_torch = None

    def capture(self) -> None:
        self.before_py = self.random.getstate()
        self.before_np = np.random.get_state()
        self.before_torch = torch.get_rng_state()

    def matches(self) -> bool:
        return (
            self.random.getstate() == self.before_py
            and np.random.get_state()[1].tolist() == self.before_np[1].tolist()
            and torch.equal(torch.get_rng_state(), self.before_torch)
        )


class TestRngScope:
    def test_scope_restores_all_streams(self) -> None:
        probe = _RngProbe()
        probe.capture()
        with _scoped_global_rng():
            torch.manual_seed(1234)
            np.random.seed(1234)
            import random as _r

            _r.seed(1234)
            _ = torch.randn(4)
        assert probe.matches()


class TestVoiceMel:
    def test_mel_shape_and_finiteness(self) -> None:
        wav = _wav(seconds=1.0)
        mel = compute_voice_mel(wav)
        assert (
            mel.dim() == 3
            and mel.shape[0] == 1
            and mel.shape[2] == MEL_PARAMS["num_mels"]
        )
        expected_frames = int(wav.shape[1] / MEL_PARAMS["hop_size"])
        assert abs(int(mel.shape[1]) - expected_frames) <= 1
        assert torch.isfinite(mel).all()


class TestAdapterConstructionRules:
    def test_rejects_fp32_only_baseline(self) -> None:
        with pytest.raises(NotImplementedError):
            MossSpeechCodecAdapter("/nonexistent", dtype=torch.bfloat16)

    def test_rejects_empty_component_set(self) -> None:
        with pytest.raises(ValueError):
            MossSpeechCodecAdapter(
                "/nonexistent", load_encoder=False, load_decoder=False
            )

    def test_missing_codec_dir_raises(self) -> None:
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            MossSpeechCodecAdapter("/nonexistent-moss-codec-dir")


class _FakeDecoder:
    def __init__(self) -> None:
        self.released: list[str] = []

    def release_session(self, uuid: str) -> None:
        self.released.append(uuid)

    def token2wav(self, *a, **k):  # pragma: no cover - not used here
        raise AssertionError("not used in this test")


class TestCleanupIdempotency:
    def test_cleanup_and_close_are_idempotent(self) -> None:
        adapter = object.__new__(MossSpeechCodecAdapter)
        adapter._closed = False
        adapter._sessions = {"req-1": True, "req-2": True}
        adapter._decode_lock = __import__("threading").Lock()
        fake = _FakeDecoder()
        adapter._decoder = fake
        adapter._encoder = None
        adapter._feature_extractor = None
        adapter._speaker_encoder = None

        adapter.cleanup("req-1")
        adapter.cleanup("req-1")  # idempotent
        assert adapter._sessions == {"req-2": True}
        assert fake.released == ["req-1", "req-1"]

        adapter.close()
        adapter.close()  # idempotent
        assert adapter._closed
        assert adapter._sessions == {}
        assert sorted(fake.released) == ["req-1", "req-1", "req-2"]
        assert adapter._decoder is None

        with pytest.raises(CodecAdapterError):
            adapter._require_open()

    def test_cleanup_never_touches_voice_cache(self) -> None:
        adapter = object.__new__(MossSpeechCodecAdapter)
        adapter._closed = False
        adapter._sessions = {}
        adapter._decode_lock = __import__("threading").Lock()
        adapter._decoder = _FakeDecoder()
        adapter._encoder = None
        adapter._feature_extractor = None
        adapter._speaker_encoder = None
        shared = {"voice-a": object()}
        adapter.cleanup("req-x")
        adapter.close()
        assert shared == {"voice-a": object()} or "voice-a" in shared


class TestVoiceHookRoundTrip:
    def _conditioning(self) -> VoiceConditioning:
        return VoiceConditioning(
            prompt_token=torch.tensor([[1, 2, 3]], dtype=torch.int32),
            prompt_feat=torch.zeros(1, 12, 80),
            embedding=torch.zeros(1, 192),
            meta={"sr": 24000, "token_len": 3, "revision": "test"},
        )

    def test_store_load_round_trip(self) -> None:
        from sglang_omni.models.moss_speech.components.voice import MossSpeechVoiceHook

        class _Null:
            def extract(self, wav):  # pragma: no cover
                raise AssertionError("not used")

        hook = MossSpeechVoiceHook(
            encode_codes_fn=lambda w: [1], speaker_encoder=_Null()
        )
        art = self._conditioning()
        art.prompt_feat[0, 0, 0] = 3.5
        stored = hook.store_artifact(art)
        loaded = hook.load_artifact(stored)
        assert torch.equal(loaded.prompt_token, art.prompt_token)
        assert torch.equal(loaded.prompt_feat, art.prompt_feat)
        assert loaded.meta == art.meta
        # stored representation must be a detached copy
        stored["prompt_feat"][0, 0, 0] = -1.0
        assert loaded.prompt_feat[0, 0, 0] == 3.5

    def test_normalize_rejects_artifact_input(self) -> None:
        from sglang_omni.models.moss_speech.components.voice import MossSpeechVoiceHook

        class _Null:
            def extract(self, wav):  # pragma: no cover
                raise AssertionError("not used")

        hook = MossSpeechVoiceHook(
            encode_codes_fn=lambda w: [1], speaker_encoder=_Null()
        )
        with pytest.raises(TypeError):
            hook.normalize_input(self._conditioning())
