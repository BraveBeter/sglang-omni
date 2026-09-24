"""Explicit small-first profile and bounded startup-only prefix cache contracts."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_speech.components.streaming_codec import (
    MossSpeechStreamingCodec,
)


class FakeFlow:
    pre_lookahead_len = 3
    token_mel_ratio = 4

    def inference(self, *, token, finalize, **kwargs):
        count = token.shape[1] - (0 if finalize else 3)
        mel = torch.ones(1, 80, count * 4)
        return mel, mel


class FakeHift:
    def inference(self, *, speech_feat, cache_source):
        size = speech_feat.shape[-1] * 480
        noise = torch.rand(1, size) * 0.1
        return noise, noise.unsqueeze(1)


def make_codec(**kwargs):
    return MossSpeechStreamingCodec(
        None, device="cpu", flow=FakeFlow(), hift=FakeHift(), **kwargs
    )


def voice(length=342):
    return dict(
        prompt_token=torch.zeros(1, length, dtype=torch.int32),
        prompt_feat=torch.zeros(1, length * 4, 80),
        embedding=torch.zeros(1, 192),
    )


def emit(codec, codes):
    codec.begin("r", voice(), seed=7)
    state = codec.sessions["r"]
    waves = []
    for code in codes:
        waves.extend(codec.push("r", [code]))
    waves.extend(codec.push("r", [], final=True))
    return waves, state


def test_default_small_first_flag_preserves_original_pcm_ledger_and_rng():
    original = make_codec(chunk_size=25)
    candidate = make_codec(chunk_size=25, small_first_chunk=False)
    left, left_state = emit(original, [2] * 61)
    right, right_state = emit(candidate, [2] * 61)
    assert len(left) == len(right)
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(left, right)
    )
    assert left_state.ledger == right_state.ledger
    assert torch.equal(left_state.cpu_rng, right_state.cpu_rng)


@pytest.mark.parametrize("prompt,expected", [(342, 8), (300, 25), (348, 27), (349, 26)])
def test_small_first_profile_aligns_prompt_and_keeps_positive_overlap(prompt, expected):
    codec = make_codec(chunk_size=25, small_first_chunk=True)
    codec.begin("r", voice(prompt), seed=7)
    state = codec.sessions["r"]
    assert codec.next_hop(state) == expected
    assert expected * codec.samples_per_code > codec.source_cache_len
    state.cursor = expected
    assert codec.next_hop(state) == 25


def test_small_first_profile_rejects_chunk5():
    with pytest.raises(ValueError, match="25"):
        make_codec(chunk_size=5, small_first_chunk=True)


def test_first8_emission_and_final_total_samples():
    codec = make_codec(chunk_size=25, small_first_chunk=True)
    codec.begin("r", voice(), seed=7)
    assert not codec.push("r", [2] * 10)
    first = codec.push("r", [2])
    assert len(first) == 1 and first[0].numel() == 11520
    rest = codec.push("r", [2] * 20, final=True)
    assert sum(w.numel() for w in first + rest) == 31 * 1920
    assert not codec.sessions
