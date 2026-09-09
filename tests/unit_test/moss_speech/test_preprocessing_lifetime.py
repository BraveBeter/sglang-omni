"""Request state ownership through the real preprocessing scheduler."""

import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_speech import stages
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest


@pytest.mark.parametrize("outcome", ["success", "failure", "abort"])
def test_preprocessing_releases_state(monkeypatch, outcome):
    class Tokenizer:
        pad_token_id = 151643

        def __call__(self, value, **kwargs):
            return {"input_ids": [10, 11, 12]}

    voice = SimpleNamespace(
        meta={"revision": "test"},
        prompt_token=torch.zeros(1, 2, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 8, 80),
        embedding=torch.zeros(1, 192),
    )
    monkeypatch.setattr(stages, "_resolve_codec_dir", lambda *a: ".")
    monkeypatch.setattr(stages, "_load_tokenizer", lambda *a: Tokenizer())
    monkeypatch.setattr(stages.os.path, "isfile", lambda *a: True)
    monkeypatch.setattr(
        stages,
        "MossSpeechCodecAdapter",
        lambda *a, **k: SimpleNamespace(encode_voice_ref=lambda _: voice),
    )
    scheduler = stages.create_preprocessing_executor(".", voice_wav="fixture.wav")
    original = stages.normalize_and_validate
    refs = []

    def observe(*args, **kwargs):
        state = original(*args, **kwargs)
        refs.append(weakref.ref(state))
        if outcome == "abort":
            scheduler._aborted.add(kwargs["request_id"])
        return state

    monkeypatch.setattr(stages, "normalize_and_validate", observe)
    if outcome == "failure":

        def fail(*args):
            raise ValueError("post-encode failure")

        monkeypatch.setattr(stages, "validate_prompt_length", fail)
    loop = asyncio.new_event_loop()
    try:
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}], modalities=["audio"]
        )
        for i in range(3):
            rid = f"request-{i}"
            payload = StagePayload(
                request_id=rid,
                request=OmniRequest(inputs=_build_chat_generate_request(req)),
                data={},
            )
            msg = IncomingMessage(rid, "new_request", payload)
            if outcome == "failure":
                with pytest.raises(ValueError, match="post-encode failure"):
                    scheduler._run_single(msg, loop)
            else:
                scheduler._run_single(msg, loop)
            if outcome == "success":
                assert scheduler.outbox.get_nowait().type == "result"
            else:
                assert scheduler.outbox.empty()
        scheduler.stop()
    finally:
        loop.close()
    gc.collect()
    assert len(refs) == 3
    assert all(ref() is None for ref in refs)
