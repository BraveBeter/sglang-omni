"""Client request types must be ready before accepting the first request."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_speech import stages


def test_factory_prepares_client_types_without_a_request(monkeypatch):
    events = []
    original = importlib.import_module

    def imported(name, *args, **kwargs):
        result = original(name, *args, **kwargs)
        if name == "sglang_omni.client.types":
            assert all(
                hasattr(result, k)
                for k in ("SamplingParams", "Message", "GenerateRequest")
            )
            events.append("client_types_ready")
        return result

    voice = SimpleNamespace(
        meta={"revision": "cpu"},
        prompt_token=torch.zeros(1, 2, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 8, 80),
        embedding=torch.zeros(1, 192),
    )
    monkeypatch.setattr(importlib, "import_module", imported)
    monkeypatch.setattr(stages, "_resolve_codec_dir", lambda *a: ".")
    monkeypatch.setattr(stages.os.path, "isfile", lambda *a: True)
    monkeypatch.setattr(
        stages,
        "MossSpeechCodecAdapter",
        lambda *a, **k: SimpleNamespace(encode_voice_ref=lambda _: voice),
    )
    monkeypatch.setattr(stages, "_load_tokenizer", lambda *a: object())
    monkeypatch.setattr(stages, "MossSpeechGridProcessor", lambda *a: object())
    before = torch.random.get_rng_state().clone()
    scheduler = stages.create_preprocessing_executor(".", voice_wav="cpu.wav")
    assert events == ["client_types_ready"]
    assert torch.equal(before, torch.random.get_rng_state())
    assert scheduler is not None


def test_missing_voice_still_fails_before_preparation(monkeypatch):
    monkeypatch.setattr(stages, "_resolve_codec_dir", lambda *a: ".")
    monkeypatch.setattr(stages.os.path, "isfile", lambda *a: False)
    monkeypatch.setattr(
        importlib, "import_module", lambda *a, **k: pytest.fail("premature import")
    )
    with pytest.raises(FileNotFoundError):
        stages.create_preprocessing_executor(".", voice_wav="missing.wav")
