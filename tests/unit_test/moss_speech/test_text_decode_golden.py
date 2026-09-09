"""Real tokenizer text limits; checkpoint assets are supplied explicitly."""

import os
from types import SimpleNamespace

import pytest

from scripts.moss_speech.p5.quality import reference_text_fields
from sglang_omni.models.moss_speech import stages
from sglang_omni.models.moss_speech.payload_types import MossSpeechState
from sglang_omni.proto.request import StagePayload


@pytest.mark.parametrize("text", ["hello world", "你好世界", "🙂"])
def test_real_tokenizer_all_prefixes(text):
    model_dir = os.environ.get("MOSS_SPEECH_MODEL_DIR")
    if not model_dir:
        pytest.skip("Set MOSS_SPEECH_MODEL_DIR for tokenizer golden checks")
    tokenizer = stages._load_tokenizer(model_dir)
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    executor = stages.create_text_decode_executor(model_dir)
    for cap in range(1, len(tokens) + 2):
        ids = (tokens + [151645])[:cap]
        grid = [[token, 512] for token in ids]
        reference = reference_text_fields(tokenizer, grid, audio_output=False)
        state = MossSpeechState(
            output_modality="text",
            output_grid=grid,
            finish_reason="stop" if cap > len(tokens) else "length",
        )
        result = executor._fn(
            StagePayload(
                request_id="length", request=SimpleNamespace(), data=state.to_dict()
            )
        ).data
        assert result["text"] == reference["service_text"]
        assert result["usage"]["completion_tokens"] == len(grid)
        if cap >= len(tokens):
            assert result["text"] == text
        if cap == len(tokens) and len(tokens) > 1:
            assert reference["text"] != text
