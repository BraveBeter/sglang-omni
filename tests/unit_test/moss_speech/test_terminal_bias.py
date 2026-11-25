# SPDX-License-Identifier: Apache-2.0
"""Terminal Flow must preserve per-query bias in both private slots."""

import pytest
import torch
from diffusers.models.attention_processor import Attention

from sglang_omni.models.moss_speech.codec import install_flow_attention_broadcast


class BiasEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [Attention(8, heads=2, dim_head=4) for _ in range(2)]
        )

    def forward(
        self, mask: torch.Tensor, *, fail: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = [
            block.prepare_attention_mask(mask, mask.shape[-1], mask.shape[0])
            for block in self.blocks
        ]
        if fail:
            raise RuntimeError("failed estimator")
        else:
            return first, second


@pytest.mark.parametrize(
    "width,aligned", [(31, False), (32, False), (33, True), (40, False)]
)
def test_slot_bias_alignment(width: int, aligned: bool) -> None:
    slots = [torch.nn.Module(), torch.nn.Module()]
    for slot in slots:
        slot.estimator = BiasEstimator()
        install_flow_attention_broadcast(slot, terminal_bias_min_width=32)
    mask = (
        torch.arange(2 * 5 * width)
        .reshape(2, 5, width)
        .remainder(23)
        .to(torch.bfloat16)
    )
    expected = mask.unsqueeze(1).expand(-1, 2, -1, -1)
    first, second = slots[0].estimator(mask)
    other, _ = slots[1].estimator(mask)
    for output in (first, second, other):
        assert torch.equal(output, expected)
        assert output.stride(1) == 0
    assert first.data_ptr() == second.data_ptr()
    if aligned:
        assert first.stride(-2) % 8 == 0
        assert first.data_ptr() != other.data_ptr()
        assert first.data_ptr() != mask.data_ptr()
    else:
        assert first.data_ptr() == other.data_ptr() == mask.data_ptr()
    with pytest.raises(RuntimeError, match="failed estimator"):
        slots[0].estimator(mask, fail=True)
    mask.neg_()
    after, _ = slots[0].estimator(mask)
    assert torch.equal(after, mask.unsqueeze(1).expand_as(after))
