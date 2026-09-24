"""Strict local CFM processor routing, provenance, projection and layout gates."""

from __future__ import annotations

import importlib
from unittest.mock import Mock, patch

import pytest
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


def module():
    return importlib.import_module(
        "sglang_omni.models.moss_speech.components.cosyvoice_flow.cfm_attention"
    )


def make_attention() -> Attention:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        return Attention(
            query_dim=32, heads=8, dim_head=64, dropout=0, bias=False
        ).eval()


def bias_and_plan(length: int = 1400):
    api = module()
    mask, proof = api.new_full_valid_mask(torch.empty(1, 80, length), length)
    cfg = mask.expand(2, -1, -1).clone()
    proof = api.expand_cfg_provenance(proof, mask, cfg)
    positions = torch.arange(length)
    allowed = positions.unsqueeze(0) < ((positions // 20 + 1) * 20).unsqueeze(1)
    bias = ((1.0 - allowed.float()) * -1e10).unsqueeze(0).repeat(16, 1, 1)
    return bias, api.make_attention_plan(
        cfg, bias, proof, streaming=True, chunk_size=20
    )


def test_full_valid_constructor_and_cfg_identity() -> None:
    api = module()
    mask, proof = api.new_full_valid_mask(torch.empty(1, 80, 24), 24)
    assert mask.shape == (1, 1, 24) and mask.stride() == (24, 24, 1)
    assert torch.equal(mask, torch.ones_like(mask))
    assert proof.matches(mask)
    assert not proof.matches(mask.clone())
    cfg = mask.expand(2, -1, -1).clone()
    derived = api.expand_cfg_provenance(proof, mask, cfg)
    assert derived.matches(cfg)
    assert api.expand_cfg_provenance(proof, mask.clone(), cfg) is None
    assert api.expand_cfg_provenance(None, mask, cfg) is None


def test_mask_metadata_adds_no_tensor_reads_or_synchronization() -> None:
    api = module()
    mask, proof = api.new_full_valid_mask(torch.empty(1, 80, 24), 24)
    cfg = mask.expand(2, -1, -1).clone()
    bias = torch.zeros(16, 24, 24)
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host read")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
        patch.object(torch.cuda, "synchronize", side_effect=AssertionError("sync")),
    ):
        derived = api.expand_cfg_provenance(proof, mask, cfg)
        plan = api.make_attention_plan(
            cfg, bias, derived, streaming=True, chunk_size=20
        )
        assert plan is not None and plan.bias is bias


@pytest.mark.parametrize(
    "invalid", ["no_proof", "different_tensor", "final", "no_chunk"]
)
def test_unproven_masks_do_not_create_plans(invalid: str) -> None:
    api = module()
    mask, proof = api.new_full_valid_mask(torch.empty(1, 80, 24), 24)
    cfg = mask.expand(2, -1, -1).clone()
    proof = api.expand_cfg_provenance(proof, mask, cfg)
    if invalid == "no_proof":
        proof = None
    if invalid == "different_tensor":
        cfg = cfg.clone()
    plan = api.make_attention_plan(
        cfg,
        torch.zeros(16, 24, 24),
        proof,
        streaming=invalid != "final",
        chunk_size=0 if invalid == "no_chunk" else 20,
    )
    assert plan is None


@pytest.mark.parametrize(
    "changes",
    [
        {"device_name": "NVIDIA GeForce RTX 4090", "capability": (8, 9)},
        {"device_name": "NVIDIA A100-SXM4-80GB"},
        {"torch_build": "2.10.0+cu128"},
        {"cuda_build": "12.6"},
        {"dtype": torch.bfloat16},
    ],
)
def test_untested_runtime_is_not_enabled(changes: dict) -> None:
    options = dict(
        device_name="NVIDIA A800-SXM4-80GB",
        capability=(8, 0),
        torch_build="2.9.1+cu128",
        cuda_build="12.8",
        dtype=torch.float32,
    )
    assert module().runtime_supported(**options)
    options.update(changes)
    assert not module().runtime_supported(**options)


@pytest.mark.parametrize(
    "condition",
    ["cpu", "training", "grad", "missing_plan", "foreign_bias", "cross_attention"],
)
def test_unsupported_context_calls_original_processor(condition: str) -> None:
    api = module()
    attn = make_attention()
    baseline = AttnProcessor2_0()
    original = Mock(wraps=baseline)
    processor = api.MossCfmAttnProcessor(original)
    hidden = torch.randn(2, 256, 32)
    bias, plan = bias_and_plan(256)
    if condition == "training":
        attn.train()
    if condition == "missing_plan":
        plan = None
    if condition == "foreign_bias":
        bias = bias.clone()
    encoder = hidden if condition == "cross_attention" else None
    context = torch.enable_grad() if condition == "grad" else torch.no_grad()
    with (
        context,
        patch.object(processor, "_runtime_supported", return_value=condition != "cpu"),
        patch.object(api, "shape_supported", return_value=True),
    ):
        expected = baseline(
            attn, hidden, encoder_hidden_states=encoder, attention_mask=bias
        )
        actual = processor(
            attn,
            hidden,
            encoder_hidden_states=encoder,
            attention_mask=bias,
            moss_cfm_mask_plan=plan,
        )
        original.assert_called_once()
    assert torch.equal(actual, expected)


def test_projection_shapes_and_attention_output_layout_are_unchanged() -> None:
    api = module()
    attn = make_attention()
    original = AttnProcessor2_0()
    processor = api.MossCfmAttnProcessor(original)
    hidden = torch.randn(2, 1400, 32)
    bias, plan = bias_and_plan()
    seen = []
    projections = []
    handles = []
    for name in ("to_q", "to_k", "to_v"):
        handles.append(
            getattr(attn, name).register_forward_hook(
                lambda mod, inputs, output, n=name: projections.append(
                    (n, tuple(inputs[0].shape), output.detach().clone())
                )
            )
        )
    output_inputs = []
    handles.append(
        attn.to_out[0].register_forward_pre_hook(
            lambda mod, inputs: output_inputs.append(
                (inputs[0].stride(), inputs[0].detach().clone())
            )
        )
    )

    def sdpa(q, k, v, attn_mask=None, **kwargs):
        seen.append(
            (
                tuple(q.shape),
                tuple(k.shape),
                attn_mask.stride(),
                attn_mask.contiguous().view(torch.int32).clone(),
            )
        )
        b, h, length, d = q.shape
        return torch.empty_strided(q.shape, (length * h * d, d, h * d, 1)).zero_()

    efficient = int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
    try:
        with (
            torch.no_grad(),
            patch.object(processor, "_runtime_supported", return_value=True),
            patch.object(torch, "_fused_sdp_choice", return_value=efficient),
            patch("torch.nn.functional.scaled_dot_product_attention", side_effect=sdpa),
        ):
            expected = original(attn, hidden, attention_mask=bias)
            actual = processor(
                attn, hidden, attention_mask=bias, moss_cfm_mask_plan=plan
            )
        assert [r[0][-2] for r in seen] == [1400, 640, 760]
        assert [r[1][-2] for r in seen] == [1400, 640, 1400]
        assert [r[0] for r in projections] == ["to_q", "to_k", "to_v"] * 2
        for first, second in zip(projections[:3], projections[3:]):
            assert first[1] == second[1] == (2, 1400, 32)
            assert torch.equal(first[2], second[2])
        assert output_inputs[0][0] == output_inputs[1][0] == (1400 * 512, 512, 1)
        assert torch.equal(output_inputs[0][1], output_inputs[1][1])
        assert torch.equal(actual, expected)
        full = seen[0][3]
        assert torch.equal(seen[1][3], full[..., :640, :640])
        assert torch.equal(seen[2][3], full[..., 640:, :])
    finally:
        for handle in handles:
            handle.remove()


def test_user_math_backend_falls_back_without_repeating_projections() -> None:
    api = module()
    attn = make_attention()
    original = AttnProcessor2_0()
    processor = api.MossCfmAttnProcessor(original)
    hidden = torch.randn(2, 1400, 32)
    bias, plan = bias_and_plan(1400)
    before = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )
    with torch.no_grad():
        expected = original(attn, hidden, attention_mask=bias)
    with (
        torch.no_grad(),
        patch.object(processor, "_runtime_supported", return_value=True),
        patch.object(
            torch,
            "_fused_sdp_choice",
            return_value=int(torch.nn.attention.SDPBackend.MATH),
        ) as choice,
        patch.object(attn.to_q, "forward", wraps=attn.to_q.forward) as q,
    ):
        actual = processor(attn, hidden, attention_mask=bias, moss_cfm_mask_plan=plan)
        assert q.call_count == 1
        assert choice.call_count >= 1
    after = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )
    assert before == after and torch.equal(actual, expected)


def test_installation_preserves_checkpoint_keys_and_custom_processors() -> None:
    api = module()
    normal = make_attention()
    custom = make_attention()

    class CustomProcessor(AttnProcessor2_0):
        pass

    custom_processor = CustomProcessor()
    custom.set_processor(custom_processor)
    container = torch.nn.ModuleList([normal, custom])
    names = list(container.state_dict())
    assert api.install_cfm_processors(container) == 1
    assert isinstance(normal.processor, api.MossCfmAttnProcessor)
    assert custom.processor is custom_processor
    assert names == list(container.state_dict())


@pytest.mark.parametrize(
    "length,chunk_size,expected",
    [
        (1400, 20, True),
        (1420, 20, True),
        (3400, 20, True),
        (1380, 20, False),
        (1401, 20, False),
        (3420, 20, False),
        (1500, 100, True),
        (3400, 100, True),
        (1400, 100, False),
        (1520, 100, False),
        (3500, 100, False),
        (1500, 50, False),
    ],
)
def test_only_measured_discrete_shapes_are_enabled(
    length: int, chunk_size: int, expected: bool
) -> None:
    assert module().shape_supported(length, chunk_size) is expected


def test_unknown_shape_falls_back_before_projection_or_backend_query() -> None:
    api = module()
    attn = make_attention()
    original = Mock(wraps=AttnProcessor2_0())
    processor = api.MossCfmAttnProcessor(original)
    hidden = torch.randn(2, 256, 32)
    bias, plan = bias_and_plan(256)
    with (
        torch.no_grad(),
        patch.object(processor, "_runtime_supported", return_value=True),
        patch.object(
            torch, "_fused_sdp_choice", side_effect=AssertionError("dispatch")
        ),
    ):
        processor(attn, hidden, attention_mask=bias, moss_cfm_mask_plan=plan)
    original.assert_called_once()


def test_segment_backend_change_falls_back_before_any_segment_runs() -> None:
    api = module()
    attn = make_attention()
    baseline = AttnProcessor2_0()
    processor = api.MossCfmAttnProcessor(baseline)
    hidden = torch.randn(2, 256, 32)
    bias, plan = bias_and_plan(256)
    efficient = int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
    math = int(torch.nn.attention.SDPBackend.MATH)
    with torch.no_grad():
        expected = baseline(attn, hidden, attention_mask=bias)
    with (
        torch.no_grad(),
        patch.object(processor, "_runtime_supported", return_value=True),
        patch.object(api, "shape_supported", return_value=True),
        patch.object(
            torch, "_fused_sdp_choice", side_effect=[efficient, efficient, math]
        ),
        patch(
            "torch.nn.functional.scaled_dot_product_attention",
            wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as sdpa,
    ):
        actual = processor(attn, hidden, attention_mask=bias, moss_cfm_mask_plan=plan)
    assert sdpa.call_count == 1 and torch.equal(actual, expected)


def test_cfg_provenance_rejects_changed_length() -> None:
    api = module()
    mask, proof = api.new_full_valid_mask(torch.empty(1, 80, 24), 24)
    assert api.expand_cfg_provenance(proof, mask, torch.ones(2, 1, 25)) is None


def test_extra_positional_fallback_preserves_original_call_semantics() -> None:
    attn = make_attention()
    original = AttnProcessor2_0()
    processor = module().MossCfmAttnProcessor(original)
    hidden = torch.randn(2, 24, 32)
    with torch.no_grad():
        expected = original(attn, hidden, None, None, None, 1.0)
        actual = processor(attn, hidden, None, None, None, 1.0)
    assert torch.equal(actual, expected)


def test_modified_mask_preparation_uses_original_processor() -> None:
    attn = make_attention()
    original = Mock(wraps=AttnProcessor2_0())
    processor = module().MossCfmAttnProcessor(original)
    hidden = torch.randn(2, 256, 32)
    bias, plan = bias_and_plan(256)
    native = attn.prepare_attention_mask
    attn.prepare_attention_mask = lambda *args, **kwargs: native(
        *args, **kwargs
    ).zero_()
    with (
        torch.no_grad(),
        patch.object(processor, "_runtime_supported", return_value=True),
        patch.object(module(), "shape_supported", return_value=True),
    ):
        processor(attn, hidden, attention_mask=bias, moss_cfm_mask_plan=plan)
    original.assert_called_once()


def test_custom_projection_without_out_features_falls_back() -> None:
    class WrappedProjection(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, hidden):
            return self.linear(hidden)

    attn = make_attention()
    attn.to_q = WrappedProjection(attn.to_q)
    original = Mock(wraps=AttnProcessor2_0())
    processor = module().MossCfmAttnProcessor(original)
    hidden = torch.randn(2, 256, 32)
    bias, plan = bias_and_plan(256)
    with (
        torch.no_grad(),
        patch.object(processor, "_runtime_supported", return_value=True),
        patch.object(module(), "shape_supported", return_value=True),
    ):
        processor(attn, hidden, attention_mask=bias, moss_cfm_mask_plan=plan)
    original.assert_called_once()


def test_custom_decoder_bias_constructor_never_receives_mask_plan() -> None:
    from sglang_omni.models.moss_speech.components.cosyvoice_flow.decoder import (
        CausalConditionalDecoder,
    )

    api = module()
    decoder = CausalConditionalDecoder.__new__(CausalConditionalDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.static_chunk_size = 20
    decoder.eval()
    mask, proof = api.new_full_valid_mask(torch.empty(1, 80, 24), 24)
    cfg = mask.expand(2, -1, -1).clone()
    proof = api.expand_cfg_provenance(proof, mask, cfg)
    bias = torch.zeros(16, 24, 24)
    with torch.no_grad():
        assert decoder._make_attention_plan(cfg, bias, proof, True) is not None
        decoder._attention_bias = lambda *args, **kwargs: bias
        assert decoder._make_attention_plan(cfg, bias, proof, True) is None


def test_custom_block_processor_never_receives_private_mask_keyword() -> None:
    from sglang_omni.models.moss_speech.components.cosyvoice_flow.decoder import (
        CausalConditionalDecoder,
    )

    received = []

    class CustomProcessor(AttnProcessor2_0):
        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            moss_cfm_mask_plan="absent",
        ):
            received.append(moss_cfm_mask_plan)
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )

    decoder = CausalConditionalDecoder(
        in_channels=320,
        out_channels=80,
        channels=(32,),
        dropout=0,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=0,
        num_heads=8,
        act_fn="gelu",
        static_chunk_size=20,
    ).eval()
    for block in decoder.modules():
        if isinstance(block, Attention):
            block.set_processor(CustomProcessor())
    api = module()
    mask, proof = api.new_full_valid_mask(torch.empty(1, 80, 24), 24)
    cfg = mask.expand(2, -1, -1).clone()
    proof = api.expand_cfg_provenance(proof, mask, cfg)
    with torch.no_grad():
        decoder(
            torch.randn(2, 80, 24),
            cfg,
            torch.randn(2, 80, 24),
            torch.zeros(2),
            spks=torch.randn(2, 80),
            cond=torch.randn(2, 80, 24),
            streaming=True,
            attention_bias_cache={},
            full_valid_mask=proof,
        )
    assert received and all(value == "absent" for value in received)
