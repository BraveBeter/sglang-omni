# SPDX-License-Identifier: Apache-2.0
"""Canonical grid, audio-boundary, and padding contracts."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_speech.components.processor import (
    ContractViolation,
    MossSpeechGridProcessor,
)
from sglang_omni.models.moss_speech.payload_types import MossSpeechState, OutputModality
from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingText
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.fixtures.moss_speech_codec import Utf8Tokenizer, drain_messages
from tests.unit_test.fixtures.moss_speech_requests import (
    FixtureTokenizer,
    canonical_fixture,
)


@pytest.mark.parametrize(
    "case_name", ["t2t_short", "t2s_cn", "s2t_cn", "s2s_cn", "mixed_multiturn"]
)
def test_reference_grids_and_wire_round_trip(case_name: str) -> None:
    reference = canonical_fixture()
    case = reference.cases[case_name]
    processor = MossSpeechGridProcessor(FixtureTokenizer(reference.text_encodings))
    built = processor.build(case.turns, case.audio_codes, case.output_modality)
    assert built.grid[0].tolist() == case.input_ids
    assert built.attention_mask[0].tolist() == case.attention_mask
    state = MossSpeechState(
        input_grid=built.grid[0].tolist(),
        attention_mask=built.attention_mask[0].tolist(),
        prompt_grid_len=built.prompt_len,
    )
    restored = MossSpeechState.from_dict(state.to_dict())
    assert restored.input_grid == case.input_ids
    assert restored.attention_mask == case.attention_mask
    assert restored.prompt_grid_len == len(case.input_ids)


@pytest.mark.parametrize(
    "modality,system_text,assistant_prefix",
    [
        (
            "text",
            "You are a helpful assistant. Respond with text outputs.",
            [151644, 77091, 198],
        ),
        (
            "audio",
            "You are a helpful assistant. Respond with spoken outputs.",
            [151644, 77091, 198, 151646],
        ),
    ],
)
def test_default_system_follows_conversation(
    modality: OutputModality, system_text: str, assistant_prefix: list[int]
) -> None:
    reference = canonical_fixture()
    processor = MossSpeechGridProcessor(FixtureTokenizer(reference.text_encodings))
    built = processor.build(
        [{"role": "user", "kind": "text", "text": "hi"}], [], modality
    )
    expected_text = (
        [151644, 872, 198, 1104, 1105, 151645, 198, 151644, 8948, 198]
        + [1000 + byte for byte in system_text.encode("utf-8")]
        + [151645, 198]
        + assistant_prefix
    )
    assert built.grid[0, :, 0].tolist() == expected_text
    assert built.grid[0, :, 1].tolist() == [512] * len(expected_text)


def test_collate_left_pads_without_changing_samples() -> None:
    reference = canonical_fixture()
    processor = MossSpeechGridProcessor(FixtureTokenizer(reference.text_encodings))
    short = processor.build([{"role": "user", "kind": "text", "text": "a"}], [], "text")
    long = processor.build(
        [{"role": "user", "kind": "text", "text": "你好"}], [], "audio"
    )
    short_before, long_before = short.grid.clone(), long.grid.clone()
    batch = processor.collate([short, long])
    padding_length = long.prompt_len - short.prompt_len
    assert padding_length > 0
    assert batch.grid.shape == (2, long.prompt_len, 2)
    assert batch.grid[0, :padding_length].tolist() == [[151643, 512]] * padding_length
    assert (
        batch.attention_mask[0].tolist()
        == [0] * padding_length + [1] * short.prompt_len
    )
    torch.testing.assert_close(batch.grid[0, padding_length:], short_before[0])
    torch.testing.assert_close(batch.grid[1], long_before[0])
    torch.testing.assert_close(short.grid, short_before)
    torch.testing.assert_close(long.grid, long_before)


def test_multiple_audio_segments_keep_order_and_code_boundaries() -> None:
    reference = canonical_fixture()
    processor = MossSpeechGridProcessor(FixtureTokenizer(reference.text_encodings))
    built = processor.build(
        [
            {"role": "user", "kind": "audio", "text": None},
            {"role": "assistant", "kind": "text", "text": "ok"},
            {"role": "user", "kind": "audio", "text": None},
        ],
        [[0, 512, 16383], [7]],
        "audio",
    )
    audio_rows = [row for row in built.grid[0].tolist() if row[0] == 151667]
    assert [row[1] for row in audio_rows] == [0, 512, 16383, 16384, 7, 16384]
    assert built.grid[0, -1].tolist() == [151646, 512]


@pytest.mark.parametrize("audio_codes", [[], [[]]])
def test_missing_or_empty_audio_codes_are_rejected(
    audio_codes: list[list[int]],
) -> None:
    reference = canonical_fixture()
    processor = MossSpeechGridProcessor(FixtureTokenizer(reference.text_encodings))
    with pytest.raises(ContractViolation, match="missing codes|zero codes"):
        processor.build(
            [{"role": "user", "kind": "audio", "text": None}], audio_codes, "text"
        )


def test_empty_collation_is_rejected() -> None:
    reference = canonical_fixture()
    processor = MossSpeechGridProcessor(FixtureTokenizer(reference.text_encodings))
    with pytest.raises(ContractViolation, match="empty batch"):
        processor.collate([])


def test_text_utf8_hold_final_prefix_and_shutdown_cleanup() -> None:
    scheduler = MossSpeechStreamingText(Utf8Tokenizer())
    try:
        scheduler.handle_stream_chunk(
            "utf8",
            StreamItem(
                0,
                torch.tensor([228, 189]),
                "ar_engine",
                {"stream": True, "modality": "text_tokens", "row_index": 0},
            ),
        )
        assert drain_messages(scheduler) == []
        scheduler.handle_stream_chunk(
            "utf8",
            StreamItem(
                1,
                torch.tensor([160, 228]),
                "ar_engine",
                {"stream": True, "modality": "text_tokens", "row_index": 2},
            ),
        )
        assert drain_messages(scheduler) == []
        state = MossSpeechState(
            output_grid=[[token, 512] for token in [228, 189, 160, 228]],
            finish_reason="length",
        )
        scheduler.handle_stream_done("utf8")
        scheduler.handle_streaming_new_request(
            "utf8",
            StagePayload(
                "utf8",
                OmniRequest(inputs=None, params={"stream": True}),
                state.to_dict(),
            ),
        )
        messages = drain_messages(scheduler)
        assert messages[-1].data.data["text"] == "你"
        assert messages[-1].data.data["finish_reason"] == "length"
        scheduler.handle_stream_chunk(
            "late",
            StreamItem(
                0,
                torch.tensor([97]),
                "ar_engine",
                {"stream": True, "modality": "text_tokens", "row_index": 0},
            ),
        )
    finally:
        scheduler.stop()
    assert (
        not scheduler.states
        and not scheduler.stream_payloads
        and not scheduler.pending_done
    )
