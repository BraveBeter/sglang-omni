"""Actual native types and dynamic tensor contracts before CUDA qualification."""

import pytest
import torch

from sglang_omni.models.moss_speech.components import cfm_block_tail as api
from sglang_omni.models.moss_speech.components.matcha_components.transformer import (
    BasicTransformerBlock,
)


def native_block():
    return BasicTransformerBlock(256, 8, 64, activation_fn="gelu").eval()


def test_real_native_types_and_inactive_lora_are_supported():
    api.validate_native_block(native_block())


@pytest.mark.parametrize(
    "change",
    [
        "block_forward",
        "norm_forward",
        "projection_forward",
        "lora",
        "dropout",
        "subclass",
    ],
)
def test_unknown_native_body_is_rejected(change):
    block = native_block()
    if change == "block_forward":
        block.forward = lambda **kwargs: kwargs["hidden_states"]
    elif change == "norm_forward":
        block.norm1.forward = lambda value: value
    elif change == "projection_forward":
        block.attn1.to_q.forward = lambda value: value
    elif change == "lora":
        block.ff.net[2].lora_layer = torch.nn.Identity()
    elif change == "dropout":
        block.ff.net[1].p = 0.1
    else:

        class Other(torch.nn.LayerNorm):
            pass

        block.norm3 = Other(256)
    with pytest.raises(RuntimeError):
        api.validate_native_block(block)


@pytest.mark.parametrize("length", [1392, 1400, 1424, 1500])
def test_dynamic_tail_keeps_full_qkv_layout_prefix_and_owned_result(
    monkeypatch, length
):
    block = native_block()
    tail = length - 1280
    qslot = torch.zeros(2, length, 512)
    kslot, vslot = torch.full_like(qslot, 20), torch.full_like(qslot, 30)
    prefix = torch.full((2, 1280, 256), 40.0)
    shared_projection = torch.empty(2, tail, 512)
    calls = []

    def project(kind, value):
        calls.append(kind)
        assert value.is_contiguous()
        if kind in ("q", "k", "v"):
            shared_projection.fill_({"q": 1, "k": 2, "v": 3}[kind])
            return shared_projection
        if kind == "ff_first":
            return value.repeat_interleave(4, -1)
        return value[..., :256].clone()

    plans = {
        kind: (lambda value, op=kind: project(op, value))
        for kind in ("q", "k", "v", "to_out", "ff_first", "ff")
    }
    key = (length, 0, "layer.attn1")
    body = api.TailBlock(block, key, qslot, kslot, vslot, prefix, plans)
    mask = torch.zeros(1).expand(2, 8, length, length)
    monkeypatch.setattr(block.attn1, "prepare_attention_mask", lambda mask, n, b: mask)
    monkeypatch.setattr(
        torch,
        "_fused_sdp_choice",
        lambda *args: int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
    )

    def sdpa(q, k, v, **kwargs):
        assert q.stride() == (length * 512, 64, 512, 1)
        assert q.storage_offset() == 1280 * 512
        assert (
            torch.all(q == 1)
            and torch.all(k[..., 1280:, :] == 2)
            and torch.all(v[..., 1280:, :] == 3)
        )
        assert torch.all(k[..., :1280, :] == 20) and torch.all(v[..., :1280, :] == 30)
        assert kwargs["attn_mask"].shape == (2, 8, tail, length)
        return torch.zeros(2, tail, 8, 64).transpose(1, 2)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)
    first = body.run(torch.zeros(2, length, 256), mask, key=key)
    saved = first.clone()
    second = body.run(torch.ones(2, length, 256), mask, key=key)
    assert not torch.equal(first[:, 1280:], second[:, 1280:])
    assert torch.equal(first, saved) and first.data_ptr() != second.data_ptr()
    assert (
        torch.all(first[:, :1280] == 40) and torch.count_nonzero(qslot[:, :1280]) == 0
    )
    assert torch.all(kslot[:, :1280] == 20) and torch.all(vslot[:, :1280] == 30)
    assert calls == ["q", "k", "v", "to_out", "ff_first", "ff"] * 2
    assert body.counts() == (2, 2, 12)
    with pytest.raises(RuntimeError, match="identity"):
        body.run(torch.zeros(2, length, 256), mask, key=(length, 1, "layer.attn1"))
    assert body.counts() == (2, 2, 12)


def test_intermediate_gate_preserves_signed_zero_and_counts_only_complete():
    reference = {name: torch.zeros(2, 1400, 256) for name in api.STAGES}
    check = api.IntermediateGate(reference, 1400)
    for name in api.STAGES[:-1]:
        check(name, torch.zeros(2, 120, 256))
    with pytest.raises(RuntimeError, match="complete"):
        check.finish()
    wrong = torch.zeros(2, 1400, 256)
    wrong[0, 0, 0] = -0.0
    with pytest.raises(RuntimeError, match="output"):
        check("output", wrong)
    assert check.count == 12


def test_actual_native_capture_restores_inner_hooks_on_success_and_failure():
    block = native_block()
    modules = tuple(block.modules())
    before = [(dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in modules]
    result, captured = api.capture_native(
        block,
        block.forward,
        (),
        dict(hidden_states=torch.zeros(2, 4, 256), attention_mask=None),
    )
    assert set(captured) == set(api.STAGES)
    assert api.raw_equal(captured["output"], result)
    assert [
        (dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in modules
    ] == before

    def failing(**kwargs):
        block.norm1(kwargs["hidden_states"])
        raise RuntimeError("original failed")

    with pytest.raises(RuntimeError, match="original failed"):
        api.capture_native(
            block, failing, (), dict(hidden_states=torch.zeros(2, 4, 256))
        )
    assert [
        (dict(m._forward_hooks), dict(m._forward_pre_hooks)) for m in modules
    ] == before
