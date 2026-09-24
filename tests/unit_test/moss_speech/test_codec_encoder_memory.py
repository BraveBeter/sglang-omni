"""Attention storage reuse and exact inference contracts."""

from __future__ import annotations

import copy
import weakref
from typing import Any
from unittest.mock import patch

import pytest
import torch

from sglang_omni.models.moss_speech.components.cosyvoice_flow.transformer.attention import (
    RelPositionMultiHeadedAttention,
)


@pytest.mark.parametrize(
    "enabled,training,grad",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_score_storage_reused_only_for_optimized_inference(
    enabled: bool, training: bool, grad: bool
) -> None:
    module = RelPositionMultiHeadedAttention(2, 16, 0.0)
    module.train(training)
    module.inference_optimizations = enabled
    x, pos = torch.randn(1, 12, 16), torch.randn(1, 23, 16)
    references = []
    original = torch.matmul

    def matmul(*args: Any, **kwargs: Any) -> torch.Tensor:
        result = original(*args, **kwargs)
        references.append(weakref.ref(result))
        return result

    def attention(
        value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        assert len(references) == 2
        should_reuse = enabled and not training and not grad
        if should_reuse:
            assert references[0]() is scores
            assert references[1]() is None
        else:
            assert references[0]() is not None
            assert references[0]() is not scores
        return value.transpose(1, 2).reshape(1, 12, 16)

    with (
        torch.set_grad_enabled(grad),
        patch.object(torch, "matmul", side_effect=matmul),
        patch.object(module, "forward_attention", side_effect=attention),
    ):
        module(x, x, x, pos_emb=pos)


@pytest.mark.parametrize("cached", [0, 3])
@pytest.mark.parametrize("mask_kind", ["empty", "padding", "chunks"])
def test_encoder_memory_preserves_values_and_cache(cached: int, mask_kind: str) -> None:
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(19)
        module = RelPositionMultiHeadedAttention(2, 16, 0.0).eval()
        x = torch.randn(1, 12, 16)
        pos = torch.randn(1, 2 * (12 + cached) - 1, 16)
        cache = torch.randn(1, 2, cached, 16)
        if mask_kind == "empty":
            mask = torch.ones(0, 0, 0, dtype=torch.bool)
        elif mask_kind == "padding":
            mask = torch.ones(1, 1, 12 + cached, dtype=torch.bool)
            mask[..., -2:] = False
        else:
            mask = torch.tril(
                torch.ones(1, 12, 12 + cached, dtype=torch.bool), diagonal=cached
            )
        module.inference_optimizations = False
        expected = module(x, x, x, mask, pos, cache)
        module.inference_optimizations = True
        actual = module(x, x, x, mask, pos, cache)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("memory", [False, True])
def test_decoder_preparation_and_memory_options_are_independent(
    prepare: bool, memory: bool
) -> None:
    from sglang_omni.models.moss_speech.components.hf_codec import modeling

    flow = torch.nn.Identity()
    flow.input_frame_rate = 12.5
    with (
        patch.object(modeling, "_build_flow_decoder", return_value=flow) as builder,
        patch.object(modeling, "_build_hift", return_value=torch.nn.Identity()),
        patch.object(modeling.torch, "load", return_value={}),
    ):
        modeling.AudioDecoder(
            "flow",
            "hift",
            "speaker",
            device="cpu",
            load_speaker_encoder=False,
            codec_optimizations=prepare,
            codec_memory_optimizations=memory,
        )
    expected = {}
    if prepare:
        expected["inference_optimizations"] = True
    if memory:
        expected["memory_optimizations"] = True
    builder.assert_called_once_with(**expected)


@pytest.mark.parametrize("size", [1, 2, 3, 12, 17])
@pytest.mark.parametrize("batch,heads", [(1, 2), (2, 3)])
def test_relative_shift_reuses_storage_without_changing_elements(
    size: int, batch: int, heads: int
) -> None:
    module = RelPositionMultiHeadedAttention(2, 16, 0.0).eval()
    backing = torch.arange(batch * heads * size * (2 * size - 1) + 7)
    x = backing[7:].view(batch, heads, size, 2 * size - 1)
    expected = module.rel_shift(x)
    module.inference_optimizations = True
    with torch.inference_mode():
        actual = module.rel_shift(x)
    assert torch.equal(actual, expected)
    assert actual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


def test_single_frame_shift_handles_contiguous_zero_strides() -> None:
    module = RelPositionMultiHeadedAttention(2, 16, 0.0).eval()
    x = torch.arange(6).as_strided((2, 3, 1, 1), (3, 1, 0, 0))
    assert x.is_contiguous()
    expected = module.rel_shift(x)
    module.inference_optimizations = True
    with torch.inference_mode():
        actual = module.rel_shift(x)
    assert torch.equal(actual, expected)
    assert actual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


def test_noncontiguous_shift_keeps_the_copy_fallback() -> None:
    module = RelPositionMultiHeadedAttention(2, 16, 0.0).eval()
    x = torch.arange(2 * 3 * 12 * 46).view(2, 3, 12, 46)[..., ::2]
    assert not x.is_contiguous()
    expected = module.rel_shift(x)
    module.inference_optimizations = True
    with torch.inference_mode():
        actual = module.rel_shift(x)
    assert torch.equal(actual, expected)
    assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    "enabled,training,grad",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_shift_storage_reuse_is_inference_only(
    enabled: bool, training: bool, grad: bool
) -> None:
    module = RelPositionMultiHeadedAttention(2, 16, 0.0)
    module.train(training)
    module.inference_optimizations = enabled
    x = torch.randn(1, 2, 3, 5, requires_grad=grad)
    with torch.set_grad_enabled(grad):
        actual = module.rel_shift(x)
    same_storage = actual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert same_storage == (enabled and not training and not grad)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("cached", [0, 3])
def test_memory_option_preserves_input_and_parameter_gradients(
    training: bool, cached: int
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        reference = RelPositionMultiHeadedAttention(2, 16, 0.0)
        reference.train(training)
        candidate = copy.deepcopy(reference)
        candidate.inference_optimizations = True
        x = torch.randn(1, 12, 16, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        pos = torch.randn(1, 2 * (12 + cached) - 1, 16)
        cache = torch.randn(1, 2, cached, 16)
        expected = reference(x, x, x, pos_emb=pos, cache=cache)
        actual = candidate(y, y, y, pos_emb=pos, cache=cache)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        expected[0].sum().backward()
        actual[0].sum().backward()
        assert torch.equal(x.grad, y.grad)
        for left, right in zip(reference.parameters(), candidate.parameters()):
            if left.grad is None:
                assert right.grad is None
            else:
                assert torch.equal(left.grad, right.grad)
