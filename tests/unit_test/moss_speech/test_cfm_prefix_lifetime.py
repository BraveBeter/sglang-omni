"""Cache lifetime and ordering gates protect fixed conditioning reuse."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_speech.components.cfm_prefix_cache import (
    CacheBinding,
    PrefixCallLedger,
    tensor_epoch,
)


def binding():
    return CacheBinding("voice", (1400, 1500), {"runtime": "tested"}, ("weights", 0))


def test_only_matching_voice_shape_runtime_and_epoch_is_eligible():
    value = binding()
    assert value.matches("voice", 1400, False, {"runtime": "tested"}, ("weights", 0))
    for voice, length, final, environment, epoch in [
        ("other", 1400, False, {"runtime": "tested"}, ("weights", 0)),
        ("voice", 1492, False, {"runtime": "tested"}, ("weights", 0)),
        ("voice", 1400, True, {"runtime": "tested"}, ("weights", 0)),
        ("voice", 1400, False, {"runtime": "other"}, ("weights", 0)),
        ("voice", 1400, False, {"runtime": "tested"}, ("weights", 1)),
    ]:
        assert not value.matches(voice, length, final, environment, epoch)
    value.invalidate()
    assert not value.matches(
        "voice", 1400, False, {"runtime": "tested"}, ("weights", 0)
    )


def test_unbounded_or_duplicate_shape_bank_is_rejected():
    for shapes in ((1400, 1500, 1600), (1492,), (1500, 1500), ()):
        with pytest.raises(ValueError):
            CacheBinding("voice", shapes, {}, ())


def test_layer_and_euler_step_order_and_count_are_exact():
    ledger = PrefixCallLedger(("a", "b"), steps=2)
    assert ledger.next("a") == (0, "a")
    with pytest.raises(ValueError):
        ledger.next("a")
    assert ledger.next("b") == (0, "b")
    with pytest.raises(ValueError):
        ledger.finish()
    assert ledger.next("a") == (1, "a")
    assert ledger.next("b") == (1, "b")
    ledger.finish()
    with pytest.raises(ValueError):
        ledger.next("a")


def test_zero_calls_allow_an_entire_native_fallback_but_partial_never_does():
    PrefixCallLedger(("a", "b")).finish(allow_empty=True)
    ledger = PrefixCallLedger(("a", "b"))
    ledger.next("a")
    with pytest.raises(ValueError):
        ledger.finish(allow_empty=True)


def test_tensor_epoch_tracks_parameters_buffers_and_unregistered_noise():
    module = torch.nn.Linear(2, 2)
    module.register_buffer("_inference_noise", torch.zeros(1, 2, 4))
    module.rand_noise = torch.ones(1, 2, 4)
    first = tensor_epoch(module)
    with torch.no_grad():
        module.weight.add_(1)
    second = tensor_epoch(module)
    assert second != first
    module.rand_noise = module.rand_noise.clone()
    third = tensor_epoch(module)
    assert third != second
    module._inference_noise = module._inference_noise.clone()
    assert tensor_epoch(module) != third


def test_tensor_epoch_detects_encoder_training_mode_without_weight_changes():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1)).eval()
    before = tensor_epoch(module)
    module[1].train()
    assert tensor_epoch(module) != before
