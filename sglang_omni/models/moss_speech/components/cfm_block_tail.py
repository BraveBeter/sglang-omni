# SPDX-License-Identifier: Apache-2.0
"""Qualified native CFM block tails, retaining full-stride Q/K/V storage."""
from __future__ import annotations

from typing import Any, Callable

STAGES = (
    "norm1",
    "q",
    "k",
    "v",
    "sdpa",
    "to_out",
    "scaled",
    "residual1",
    "norm3",
    "ff_first",
    "gelu",
    "ff_last",
    "output",
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def raw_equal(left: Any, right: Any) -> bool:
    import torch

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def validate_native_block(block: Any) -> None:
    """Validate before installing any temporary instance forward wrappers."""
    import torch
    from diffusers.models.activations import GELU
    from diffusers.models.attention_processor import Attention
    from diffusers.models.lora import LoRACompatibleLinear

    from .matcha_components.transformer import BasicTransformerBlock, FeedForward

    def original(module: Any, cls: type) -> bool:
        return (
            type(module) is cls
            and "forward" not in vars(module)
            and getattr(module.forward, "__func__", None) is cls.forward
        )

    require(
        original(block, BasicTransformerBlock)
        and not block.training
        and not block.use_ada_layer_norm
        and not block.use_ada_layer_norm_zero
        and not block.only_cross_attention
        and block.attn2 is None
        and block._chunk_size is None
        and getattr(block, "pos_embed", None) is None,
        "Unsupported native block body",
    )
    attention = block.attn1
    require(
        original(attention, Attention)
        and attention.heads == 8
        and not attention.residual_connection
        and attention.rescale_output_factor == 1.0
        and attention.group_norm is None
        and attention.spatial_norm is None
        and attention.norm_q is None
        and attention.norm_k is None
        and "prepare_attention_mask" not in vars(attention),
        "Unsupported native attention body",
    )
    require(
        all(
            original(norm, torch.nn.LayerNorm) and norm.normalized_shape == (256,)
            for norm in (block.norm1, block.norm3)
        ),
        "Unsupported native normalization",
    )
    require(
        original(block.ff, FeedForward)
        and len(block.ff.net) == 3
        and original(block.ff.net[0], GELU)
        and block.ff.net[0].approximate in ("none", "tanh")
        and "gelu" not in vars(block.ff.net[0]),
        "Unsupported native FF/GELU body",
    )
    require(
        all(
            original(drop, torch.nn.Dropout) and drop.p == 0.0
            for drop in (attention.to_out[1], block.ff.net[1])
        ),
        "Unsupported native dropout",
    )
    modules = (
        attention.to_q,
        attention.to_k,
        attention.to_v,
        attention.to_out[0],
        block.ff.net[0].proj,
        block.ff.net[2],
    )
    require(
        all(
            type(module) in (torch.nn.Linear, LoRACompatibleLinear)
            and original(module, type(module))
            and getattr(module, "lora_layer", None) is None
            for module in modules
        ),
        "Unsupported native projection body",
    )


class IntermediateGate:
    """Startup-only signed-byte and declared-layout gate for every actual stage."""

    def __init__(self, reference: dict[str, Any], length: int):
        require(set(reference) == set(STAGES), "Incomplete native intermediates")
        self.reference, self.length, self.seen = reference, length, set()

    @property
    def count(self) -> int:
        return len(self.seen)

    def __call__(self, name: str, value: Any) -> None:
        require(
            name in STAGES and name not in self.seen,
            "Duplicate/unknown block intermediate",
        )
        expected = self.reference[name]
        if value.shape == expected.shape:
            layout = value.stride() == expected.stride()
        else:
            tail, width = self.length - 1280, expected.shape[-1]
            require(
                value.shape == (2, tail, width), "Unknown tail intermediate geometry"
            )
            expected = expected[:, 1280:, :]
            layout = value.stride() == (tail * width, width, 1)
        require(
            layout and raw_equal(value, expected),
            "Native block intermediate differs: " + name,
        )
        self.seen.add(name)

    def finish(self) -> int:
        require(self.seen == set(STAGES), "Incomplete native block intermediate gate")
        return self.count


def capture_native(
    block: Any, original: Callable, args: tuple, kwargs: dict
) -> tuple[Any, dict]:
    """Capture one original block only, restoring inner hooks even on failure."""
    captured, handles = {}, []

    def snapshot(name, value):
        require(name not in captured, "Duplicate original block operator")
        captured[name] = value.clone()
        require(
            captured[name].stride() == value.stride(), "Native capture layout changed"
        )

    def post(name):
        return lambda module, inputs, value: snapshot(name, value)

    def pre(name):
        return lambda module, inputs: snapshot(name, inputs[0])

    try:
        for name, module in (
            ("norm1", block.norm1),
            ("q", block.attn1.to_q),
            ("k", block.attn1.to_k),
            ("v", block.attn1.to_v),
            ("to_out", block.attn1.to_out[0]),
            ("scaled", block.attn1),
            ("norm3", block.norm3),
            ("ff_first", block.ff.net[0].proj),
            ("gelu", block.ff.net[0]),
            ("ff_last", block.ff.net[2]),
        ):
            handles.append(module.register_forward_hook(post(name)))
        handles.append(block.attn1.to_out[0].register_forward_pre_hook(pre("sdpa")))
        handles.append(block.norm3.register_forward_pre_hook(pre("residual1")))
        result = original(*args, **kwargs)
        snapshot("output", result)
        require(
            set(captured) == set(STAGES), "Incomplete original block operator capture"
        )
        return result, captured
    finally:
        for handle in handles:
            handle.remove()


class TailBlock:
    """Current-input tail; caller owns serialization, plans and immutable prefixes."""

    def __init__(
        self,
        block: Any,
        key: tuple,
        query: Any,
        key_slot: Any,
        value_slot: Any,
        output_prefix: Any,
        plans: dict,
    ):
        import torch

        self.block, self.key, self.length = block, key, key[0]
        require(
            self.length in (1392, 1400, 1424, 1500) and key[1] in range(10),
            "Unknown block identity",
        )
        self.query, self.key_slot, self.value_slot = query, key_slot, value_slot
        self.output_prefix, self.plans = output_prefix, plans
        require(
            set(plans) == {"q", "k", "v", "to_out", "ff_first", "ff"},
            "Incomplete tail projections",
        )
        require(
            all(
                value.shape == (2, self.length, 512)
                and value.is_contiguous()
                and value.dtype == torch.float32
                and value.device == query.device
                for value in (query, key_slot, value_slot)
            ),
            "Unknown full-stride Q/K/V storage",
        )
        require(
            output_prefix.shape == (2, 1280, 256)
            and output_prefix.is_contiguous()
            and output_prefix.dtype == query.dtype
            and output_prefix.device == query.device,
            "Unknown whole-block prefix storage",
        )
        self._blocks = self._sdpa = self._projections = 0

    def counts(self) -> tuple[int, int, int]:
        return self._blocks, self._sdpa, self._projections

    def run(
        self, hidden: Any, mask: Any, *, key: tuple, observe: Callable | None = None
    ) -> Any:
        import torch
        import torch.nn.functional as F

        require(key == self.key, "Changed block prefix identity")
        length = self.length
        require(
            hidden.shape == (2, length, 256)
            and hidden.is_contiguous()
            and hidden.dtype == self.query.dtype
            and hidden.device == self.query.device,
            "Unknown current block input layout",
        )

        def record(name, value):
            if observe is not None:
                observe(name, value)
            return value

        def project(kind, value):
            result = self.plans[kind](value)
            self._projections += 1
            return result

        current = hidden[:, 1280:, :]
        normalized = record("norm1", self.block.norm1(current))
        for kind, slot in (
            ("q", self.query),
            ("k", self.key_slot),
            ("v", self.value_slot),
        ):
            # Consume each borrowed result before any following projection can overwrite it.
            slot[:, 1280:, :].copy_(record(kind, project(kind, normalized)))
        q, k, v = (
            value.view(2, length, 8, 64).transpose(1, 2)
            for value in (self.query, self.key_slot, self.value_slot)
        )
        q = q[..., 1280:, :]
        require(
            q.stride() == (length * 512, 64, 512, 1)
            and q.storage_offset() == 1280 * 512
            and all(
                value.stride() == (length * 512, 64, 512, 1)
                and value.storage_offset() == 0
                for value in (k, v)
            ),
            "Changed original SDPA layout",
        )
        bias = self.block.attn1.prepare_attention_mask(mask, length, 2).view(
            2, 8, length, length
        )
        bias = bias[..., 1280:, :]
        require(
            int(torch._fused_sdp_choice(q, k, v, bias, 0.0, False))
            == int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
            "Unqualified tail SDPA backend",
        )
        attention = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        self._sdpa += 1
        attention = record(
            "sdpa", attention.transpose(1, 2).reshape(2, -1, 512).to(q.dtype)
        )
        output = record("to_out", project("to_out", attention))
        output = record(
            "scaled",
            self.block.attn1.to_out[1](output) / self.block.attn1.rescale_output_factor,
        )
        residual = record("residual1", output + current)
        normalized = record("norm3", self.block.norm3(residual))
        first = record("ff_first", project("ff_first", normalized))
        activated = record("gelu", self.block.ff.net[0].gelu(first))
        ff = record("ff_last", project("ff", self.block.ff.net[1](activated)))
        output = torch.cat((self.output_prefix, ff + residual), dim=1)
        self._blocks += 1
        return record("output", output)
