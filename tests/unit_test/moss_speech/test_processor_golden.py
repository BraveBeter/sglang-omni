"""Golden grid tests for the MOSS-Speech canonical processor (T2.3).

Rebuilds the P0 fixtures' canonical inputs from (conversation, per-turn
codes) and asserts exact token-grid and attention-mask equality with the
reference processor output saved in P0. Requires:
  MOSS_SPEECH_MODEL_DIR    — locked checkpoint dir (tokenizer)
  MOSS_SPEECH_FIXTURES_DIR — repo tests/fixtures/moss_speech
  MOSS_P1_ALIGNMENT_DIR    — workspace artifacts/p1/alignment (reference codes)
Skipped when the env vars are absent (pure-repo CI runs).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from sglang_omni.models.moss_speech.components.processor import MossSpeechGridProcessor
from sglang_omni.models.moss_speech.payload_types import MossSpeechState

SYSTEM_TEXT = "You are a helpful assistant. Answer the user's questions with text."
SYSTEM_SPEECH = "You are a helpful voice assistant. Answer the user's questions with spoken responses."


def _env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value and Path(value).is_dir() else None


@pytest.fixture(scope="module")
def tokenizer():
    model_dir = _env("MOSS_SPEECH_MODEL_DIR")
    if model_dir is None:
        pytest.skip("MOSS_SPEECH_MODEL_DIR not set")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)


@pytest.fixture(scope="module")
def reference_codes():
    align = _env("MOSS_P1_ALIGNMENT_DIR")
    if align is None:
        pytest.skip("MOSS_P1_ALIGNMENT_DIR not set")
    export = json.loads((align / "reference" / "reference_export.json").read_text())
    return {"cn": export["encode"]["single_cn_r0"], "en": export["encode"]["single_en_r0"]}


def _cases(fixtures: Path):
    return {
        "t2t_short": (
            "text", SYSTEM_TEXT,
            [{"role": "user", "kind": "text", "text": "Introduce yourself in one sentence."}],
            [],
        ),
        "t2s_cn": (
            "audio", SYSTEM_SPEECH,
            [{"role": "user", "kind": "text", "text": "用中文介绍一下上海的三到四个著名景点。"}],
            [],
        ),
        "s2t_cn": (
            "text", SYSTEM_TEXT,
            [{"role": "user", "kind": "audio", "text": None}],
            ["cn"],
        ),
        "s2s_cn": (
            "audio", SYSTEM_SPEECH,
            [{"role": "user", "kind": "audio", "text": None}],
            ["cn"],
        ),
        "mixed_multiturn": (
            "text", SYSTEM_TEXT,
            [
                {"role": "user", "kind": "text", "text": "My name is Alice and I like hiking."},
                {"role": "assistant", "kind": "text", "text": "Nice to meet you, Alice! Hiking is a great way to enjoy nature."},
                {"role": "user", "kind": "audio", "text": None},
            ],
            ["cn"],
        ),
    }


def test_golden_grids_match_reference(tokenizer, reference_codes) -> None:
    fixtures = _env("MOSS_SPEECH_FIXTURES_DIR")
    if fixtures is None:
        pytest.skip("MOSS_SPEECH_FIXTURES_DIR not set")
    proc = MossSpeechGridProcessor(tokenizer)
    for case, (modality, system, turns, audio_assets) in _cases(fixtures).items():
        canon = json.loads((fixtures / case / "canonical_input.json").read_text())
        expected_grid = torch.tensor(canon["input_ids"], dtype=torch.long)
        expected_mask = torch.tensor(canon["attention_mask"], dtype=torch.long)
        full_turns = [{"role": "system", "kind": "text", "text": system}] + turns
        codes = [reference_codes[a] for a in audio_assets]
        built = proc.build(full_turns, codes, modality)
        assert torch.equal(built.grid[0], expected_grid), (
            f"{case}: grid mismatch (built {tuple(built.grid.shape)} vs expected {tuple(expected_grid.shape)})"
        )
        assert torch.equal(built.attention_mask[0], expected_mask), case


def test_default_system_appended_at_end_not_first(tokenizer) -> None:
    proc = MossSpeechGridProcessor(tokenizer)
    turns = [{"role": "user", "kind": "text", "text": "hi"}]
    built = proc.build(turns, [], "text")
    ids = built.grid[0, :, 0].tolist()
    # reference quirk: the default system turn is appended AFTER the user turn.
    # Locate via role tokens (im_start ids repeat across turns, so search the
    # distinctive system/user role tokens instead).
    user_tok = tokenizer("user", add_special_tokens=False)["input_ids"][0]
    sys_tok = tokenizer("system", add_special_tokens=False)["input_ids"][0]
    first_user_pos = ids.index(user_tok)
    system_pos = ids.index(sys_tok)
    assert system_pos > first_user_pos


def test_collate_left_padding_matches_reference_semantics(tokenizer) -> None:
    proc = MossSpeechGridProcessor(tokenizer)
    a = proc.build([{"role": "user", "kind": "text", "text": "short"}], [], "text")
    b = proc.build(
        [{"role": "user", "kind": "text", "text": "a considerably longer user message for padding"}],
        [], "text",
    )
    batch = proc.collate([a, b])
    assert batch.grid.shape == (2, max(a.prompt_len, b.prompt_len), 2)
    shorter = min(a.prompt_len, b.prompt_len)
    pad_len = batch.grid.shape[1] - shorter
    # left padding: text channel padded with tokenizer pad, audio channel with 512
    assert (batch.grid[0, :pad_len, 0] == tokenizer.pad_token_id).all()
    assert (batch.grid[0, :pad_len, 1] == 512).all()
    assert (batch.attention_mask[0, :pad_len] == 0).all()
    assert (batch.attention_mask[1] == 1).all()


def test_state_grid_round_trip_wire(tokenizer) -> None:
    """Grid survives the wire codec round trip (cross-process contract)."""
    proc = MossSpeechGridProcessor(tokenizer)
    built = proc.build([{"role": "user", "kind": "text", "text": "round trip"}], [], "text")
    state = MossSpeechState(
        input_grid=built.grid[0].tolist(),
        attention_mask=built.attention_mask[0].tolist(),
        prompt_grid_len=built.prompt_len,
    )
    dumped = state.to_dict()
    restored = MossSpeechState.from_dict(dumped)
    assert restored.input_grid == state.input_grid
    assert restored.attention_mask == state.attention_mask
    assert torch.equal(torch.tensor(restored.input_grid), built.grid[0])
