"""Streaming sample accounting and request isolation without model assets."""

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
        n = speech_feat.shape[-1] * 480
        noise = torch.rand(1, n) * 0.1
        return noise, noise.unsqueeze(1)


def codec():
    return MossSpeechStreamingCodec(
        None, chunk_size=5, device="cpu", flow=FakeFlow(), hift=FakeHift()
    )


def voice():
    return {
        "prompt_token": torch.zeros(1, 2, dtype=torch.int32),
        "prompt_feat": torch.zeros(1, 8, 80),
        "embedding": torch.zeros(1, 192),
    }


@pytest.mark.parametrize("length", [1, 4, 5, 8, 10, 11, 13, 16, 21, 28])
def test_sample_accounting_short_boundary_and_tail(length):
    c = codec()
    c.begin("r", voice(), seed=3)
    waves = []
    for _ in range(length):
        waves.extend(c.push("r", [100]))
    waves.extend(c.push("r", [], final=True))
    assert sum(w.numel() for w in waves) == length * 1920
    assert all(torch.isfinite(w).all() and w.numel() > 0 for w in waves)
    assert not c.sessions
    with pytest.raises(ValueError, match="session"):
        c.push("r", [], final=True)


def test_rng_isolated_across_interleaving_and_abort():
    c = codec()
    before = torch.get_rng_state().clone()
    flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    c.begin("serial", voice(), seed=7)
    expected = torch.cat(c.push("serial", [2] * 28, final=True))
    c.begin("a", voice(), seed=7)
    c.begin("b", voice(), seed=19)
    actual = []
    for _ in range(28):
        actual.extend(c.push("a", [2]))
        c.push("b", [3])
    c.cleanup("b")
    c.cleanup("b")
    actual.extend(c.push("a", [], final=True))
    assert torch.equal(expected, torch.cat(actual))
    assert torch.equal(before, torch.get_rng_state())
    assert flags == (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    assert not c.sessions


def test_invalid_code_and_decoder_error_release_session():
    c = codec()
    c.begin("bad", voice(), seed=0)
    with pytest.raises(ValueError, match="code"):
        c.push("bad", [16384])
    assert not c.sessions
    c.begin("error", voice(), seed=0)

    def fail(**kwargs):
        raise RuntimeError("codec failure")

    c.hift.inference = fail
    with pytest.raises(RuntimeError, match="codec failure"):
        c.push("error", [5] * 30)
    assert not c.sessions


def test_duplicate_begin_and_empty_final_are_rejected():
    c = codec()
    c.begin("r", voice(), seed=0)
    with pytest.raises(ValueError, match="active"):
        c.begin("r", voice(), seed=1)
    with pytest.raises(ValueError, match="empty"):
        c.push("r", [], final=True)
    assert not c.sessions


def test_weight_loader_allows_only_training_metadata():
    from sglang_omni.models.moss_speech.components.streaming_codec import (
        load_flow_weights,
    )

    layer = torch.nn.Linear(2, 1)
    state = dict(layer.state_dict(), epoch=12, step=100)
    load_flow_weights(layer, state)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_flow_weights(layer, dict(state, unknown=torch.zeros(1)))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_flow_weights(layer, {"epoch": 12, "step": 100})


@pytest.mark.parametrize("length", [1, 4, 5, 8, 10, 11, 13, 16, 21, 28, 512])
def test_bounded_hops_preserve_samples_ledger_and_rng(length):
    c = codec()
    before = torch.get_rng_state().clone()
    c.begin("eager", voice(), seed=7)
    eager_state = c.sessions["eager"]
    expected = c.push("eager", [2] * length, final=True)
    c.begin("bounded", voice(), seed=7)
    bounded_state = c.sessions["bounded"]
    actual = c.push("bounded", [2] * length, final=True, max_hops=1)
    assert len(actual) == 1
    while "bounded" in c.sessions:
        waves = c.push("bounded", [], max_hops=1)
        assert len(waves) == 1
        actual.extend(waves)
    assert len(expected) == len(actual)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert eager_state.ledger == bounded_state.ledger
    assert torch.equal(eager_state.cpu_rng, bounded_state.cpu_rng)
    assert torch.equal(before, torch.get_rng_state())


def test_bounded_hops_interleave_seeds_and_abort_without_rng_leak():
    c = codec()
    before = torch.get_rng_state().clone()
    c.begin("eager", voice(), seed=7)
    expected = c.push("eager", [2] * 28, final=True)
    c.begin("a", voice(), seed=7)
    c.begin("b", voice(), seed=19)
    actual = c.push("a", [2] * 28, final=True, max_hops=1)
    c.push("b", [3] * 28, final=True, max_hops=1)
    c.cleanup("b")
    while "a" in c.sessions:
        actual.extend(c.push("a", [], max_hops=1))
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert torch.equal(before, torch.get_rng_state())
    assert not c.sessions


def test_bounded_hop_validation_and_pending_final_reject_new_codes():
    c = codec()
    c.begin("bad_limit", voice(), seed=0)
    with pytest.raises(ValueError, match="max_hops"):
        c.push("bad_limit", [2], max_hops=0)
    assert not c.sessions
    c.begin("r", voice(), seed=0)
    c.push("r", [2] * 28, final=True, max_hops=1)
    with pytest.raises(ValueError, match="final"):
        c.push("r", [3], max_hops=1)
    assert not c.sessions
