# SPDX-License-Identifier: Apache-2.0
"""Flow encoder precision boundaries, mask ownership and reversible parameter storage."""

import pytest
import torch

from sglang_omni.models.moss_speech.components.encoder_precision import (
    EncoderPrecisionMode,
    FlowEncoderBF16Scope,
)
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import scoped_global_rng
from tests.unit_test.fixtures.moss_speech_codec import tiny_condition_encoder


@pytest.mark.parametrize("mode", ["linear", "attention", "full"])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
@torch.no_grad()
def test_encoder_bf16_preserves_masks_rng_and_exact_fp32_restoration(
    device: str, mode: EncoderPrecisionMode
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    with scoped_global_rng():
        encoder = tiny_condition_encoder(torch.device(device))
        hidden_states = torch.randn(1, 17, 512, device=device)
        lengths = torch.tensor([17], device=device)
        reference, reference_mask = encoder(hidden_states, lengths, streaming=True)
        original = {
            name: parameter.detach().cpu().clone()
            for name, parameter in encoder.named_parameters()
        }
        identities = {
            name: id(parameter) for name, parameter in encoder.named_parameters()
        }
        cpu_rng = torch.get_rng_state().clone()
        scope = FlowEncoderBF16Scope(encoder, mode)
        for activation in range(2):
            with scope:
                actual, mask = encoder(hidden_states, lengths, streaming=True)
                repeated, repeated_mask = encoder(
                    hidden_states, lengths, streaming=True
                )
                assert actual.dtype is torch.float32 and actual.shape == reference.shape
                assert torch.isfinite(actual).all() and torch.equal(
                    mask, reference_mask
                )
                assert torch.equal(actual, repeated) and torch.equal(
                    mask, repeated_mask
                )
                assert all(
                    operator.forward.executions == operator.forward.bf16_executions > 0
                    for operator in scope.operators
                )
                assert all(
                    operator.forward.last_compute_dtype is torch.float32
                    for operator in scope.fp32_operators
                )
                assert all(
                    operator.module.weight.dtype is torch.bfloat16
                    for operator in scope.operators
                )
                assert all(
                    operator.module.weight.data_ptr()
                    == operator.forward.resident_weight.data_ptr()
                    for operator in scope.operators
                )
                assert all(
                    backup.device.type == "cpu" and backup.dtype is torch.float32
                    for operator in scope.operators
                    for backup in operator.forward.fp32_cpu_parameters.values()
                )
                assert scope.forward.executions == (activation + 1) * 2
                assert (
                    scope.forward.last_output_dtype is torch.float32
                    and scope.forward.last_mask_dtype is torch.bool
                )
                if mode == "full":
                    assert (
                        scope.convolution_executions
                        == scope.bf16_convolution_executions
                        > 0
                    )
                else:
                    assert scope.convolution_executions == 0
            assert not scope.active and not scope.handles
            assert torch.equal(
                encoder(hidden_states, lengths, streaming=True)[0], reference
            )
            for name, parameter in encoder.named_parameters():
                assert (
                    id(parameter) == identities[name]
                    and parameter.dtype is torch.float32
                )
                assert torch.equal(parameter.detach().cpu(), original[name])
        assert torch.equal(cpu_rng, torch.get_rng_state())


def test_encoder_bf16_partial_entry_restores_parameters_and_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scoped_global_rng():
        encoder = tiny_condition_encoder(torch.device("cpu"))
        originals = {
            name: parameter.clone() for name, parameter in encoder.named_parameters()
        }
        scope = FlowEncoderBF16Scope(encoder)

        def fail() -> None:
            raise MemoryError("encoder preparation failed")

        monkeypatch.setattr(scope.operators[1].forward, "prepare", fail)
        with pytest.raises(MemoryError, match="encoder preparation failed"):
            scope.__enter__()
        assert not scope.active and not scope.handles and "forward" not in vars(encoder)
        for name, parameter in encoder.named_parameters():
            assert parameter.dtype is torch.float32 and torch.equal(
                parameter, originals[name]
            )


def test_encoder_bf16_rejects_nested_scopes_training_and_gradients() -> None:
    with scoped_global_rng():
        encoder = tiny_condition_encoder(torch.device("cpu"))
        hidden_states = torch.randn(1, 17, 512)
        lengths = torch.tensor([17])
        with FlowEncoderBF16Scope(encoder):
            with pytest.raises(ValueError, match="unmodified operator"):
                FlowEncoderBF16Scope(encoder).__enter__()
            with pytest.raises(RuntimeError, match="no-grad evaluation"):
                encoder(hidden_states, lengths)
            encoder.train()
            with (
                torch.no_grad(),
                pytest.raises(RuntimeError, match="no-grad evaluation"),
            ):
                encoder(hidden_states, lengths)
        with pytest.raises(ValueError, match="supported native encoder"):
            FlowEncoderBF16Scope(encoder, "unknown")
