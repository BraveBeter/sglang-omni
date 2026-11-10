# SPDX-License-Identifier: Apache-2.0
"""Forward-local bias ownership, live masks, and conservative consumer fallback."""

from unittest.mock import patch

import pytest
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

import sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.decoder as decoder_module
from sglang_omni.vendor.moss_speech_codec.cosyvoice_flow.decoder import (
    CausalConditionalDecoder,
)
from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import scoped_global_rng
from sglang_omni.vendor.moss_speech_codec.matcha_components.bias_broadcast import (
    install_broadcast_bias_attention,
)


@pytest.fixture
def middle_bias_decoder() -> CausalConditionalDecoder:
    with scoped_global_rng():
        torch.manual_seed(42)
        return CausalConditionalDecoder(
            in_channels=32,
            out_channels=8,
            channels=(16,),
            dropout=0,
            attention_head_dim=8,
            n_blocks=1,
            num_mid_blocks=3,
            num_heads=2,
            act_fn="gelu",
            static_chunk_size=4,
        ).eval()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.accelerator),
    ],
    ids=["fp32-cpu", "bf16-cuda"],
)
def test_middle_bias_reuse_preserves_live_mask_and_call_ownership(
    middle_bias_decoder: CausalConditionalDecoder,
    streaming: bool,
    broadcast: bool,
    device: str,
    dtype: torch.dtype,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    else:
        pass
    decoder = middle_bias_decoder.to(device=device, dtype=dtype)
    if broadcast:
        install_broadcast_bias_attention(decoder)
    else:
        pass
    features = (torch.arange(128, dtype=torch.float32).reshape(2, 8, 8) / 128).to(
        device=device, dtype=dtype
    )
    timestep = torch.ones(2, device=device, dtype=dtype)
    mask = torch.ones(2, 1, 8, device=device, dtype=dtype)
    with torch.inference_mode():
        for visible_frames in (8, 5, 8):
            mask.zero_()
            mask[:, :, :visible_frames] = 1
            decoder.reuse_middle_attention_bias = False
            expected = decoder(
                features,
                mask,
                features,
                timestep,
                features[:, :, 0],
                features,
                streaming,
            )
            decoder.reuse_middle_attention_bias = True
            original_mask = mask.clone()
            with patch.object(
                decoder_module, "mask_to_bias", wraps=decoder_module.mask_to_bias
            ) as construction:
                actual = decoder(
                    features,
                    mask,
                    features,
                    timestep,
                    features[:, :, 0],
                    features,
                    streaming,
                )
            assert construction.call_count == 3
            assert torch.equal(expected, actual) and torch.equal(mask, original_mask)
    assert decoder.middle_bias_constructions == 3
    assert decoder.middle_bias_reuses == 6 and decoder.middle_bias_fallbacks == 0


class MutatingBiasProcessor(AttnProcessor2_0):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        attention_mask[:, :, -1] = 0
        return super().__call__(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )


@pytest.mark.parametrize("consumer", ["processor", "hook", "preparer"])
def test_middle_bias_reuse_falls_back_for_mutating_consumers(
    middle_bias_decoder: CausalConditionalDecoder, consumer: str
) -> None:
    decoder = middle_bias_decoder
    attention = decoder.mid_blocks[0][1][0].attn1
    hook = None
    if consumer == "processor":
        attention.set_processor(MutatingBiasProcessor())
    elif consumer == "hook":

        def mutate_bias(
            module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            kwargs: dict[str, torch.Tensor],
        ) -> None:
            kwargs["attention_mask"][:, :, -1] = 0

        hook = decoder.mid_blocks[0][1][0].register_forward_pre_hook(
            mutate_bias, with_kwargs=True
        )
    else:
        original_preparer = attention.prepare_attention_mask

        def prepare_mask(mask: torch.Tensor, *args: int, **kwargs: int) -> torch.Tensor:
            mask[:, :, -1] = 0
            return original_preparer(mask, *args, **kwargs)

        attention.prepare_attention_mask = prepare_mask
    features = torch.arange(128, dtype=torch.float32).reshape(2, 8, 8) / 128
    mask = torch.ones(2, 1, 8)
    mask[:, :, -2:] = 0
    try:
        with torch.inference_mode():
            decoder.reuse_middle_attention_bias = False
            expected = decoder(
                features,
                mask,
                features,
                torch.ones(2),
                features[:, :, 0],
                features,
                True,
            )
            decoder.reuse_middle_attention_bias = True
            with patch.object(
                decoder_module, "mask_to_bias", wraps=decoder_module.mask_to_bias
            ) as construction:
                actual = decoder(
                    features,
                    mask,
                    features,
                    torch.ones(2),
                    features[:, :, 0],
                    features,
                    True,
                )
        assert torch.equal(expected, actual)
        assert construction.call_count == 5
        assert decoder.middle_bias_reuses == 0 and decoder.middle_bias_fallbacks == 1
    finally:
        if hook is not None:
            hook.remove()
        else:
            pass


def test_middle_bias_reuse_does_not_retain_bias_after_exception(
    middle_bias_decoder: CausalConditionalDecoder,
) -> None:
    decoder = middle_bias_decoder
    decoder.reuse_middle_attention_bias = True
    features = torch.ones(2, 8, 8)
    mask = torch.ones(2, 1, 8)

    def fail(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        raise RuntimeError("Injected middle consumer failure")

    hook = decoder.mid_blocks[1][0].register_forward_pre_hook(fail)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="middle consumer"):
        decoder(
            features, mask, features, torch.ones(2), features[:, :, 0], features, True
        )
    hook.remove()
    mask[:, :, 4:] = 0
    with torch.inference_mode():
        actual = decoder(
            features, mask, features, torch.ones(2), features[:, :, 0], features, False
        )
        decoder.reuse_middle_attention_bias = False
        expected = decoder(
            features, mask, features, torch.ones(2), features[:, :, 0], features, False
        )
    assert torch.equal(expected, actual) and decoder.middle_bias_reuses == 2


@pytest.mark.parametrize("training", [False, True])
def test_middle_bias_reuse_keeps_training_and_grad_consumers_separate(
    middle_bias_decoder: CausalConditionalDecoder, training: bool
) -> None:
    decoder = middle_bias_decoder.train(training)
    decoder.reuse_middle_attention_bias = True
    features = torch.ones(2, 8, 8, requires_grad=True)
    output = decoder(
        features,
        torch.ones(2, 1, 8),
        features,
        torch.ones(2),
        features[:, :, 0],
        features,
        True,
    )
    output.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert decoder.middle_bias_reuses == 0 and decoder.middle_bias_fallbacks == 1
