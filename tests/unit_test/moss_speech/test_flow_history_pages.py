# SPDX-License-Identifier: Apache-2.0
"""Each committed Euler step retains its own exact closed K/V and convolution tails."""
from contextvars import ContextVar

import torch
from torch.nn import functional as F

from sglang_omni.models.moss_speech.flow_history_pages import (
    PagedHistoryPrefill,
    PageLayout,
    PageStep,
)
from sglang_omni.models.moss_speech.flow_prefix import ReferencePrefix


@torch.inference_mode()
def test_page_io_reads_prefix_and_writes_only_closed_frames():
    generator = torch.Generator().manual_seed(42)
    shared = ReferencePrefix(
        reference_key=b"x",
        mel_frames=100,
        key_values=[torch.randn(2, 100, 4, generator=generator)],
        convolution_tails=[torch.randn(2, 3, 2, generator=generator)],
    )
    layout = PageLayout(shared, 1, 100)
    pages = [layout.view(torch.zeros(layout.elements)) for _ in range(3)]
    pages[0].key_values[0].normal_(generator=generator)
    pages[0].convolution_tails[0].normal_(generator=generator)
    target = PagedHistoryPrefill.__new__(PagedHistoryPrefill)
    target.current = ContextVar("test_solve", default=None)
    target.active_pages = ContextVar("test_pages", default=None)
    target.chunk_frames = 100
    projection = torch.nn.Linear(3, 4, bias=False)
    conv = torch.nn.Conv1d(3, 5, kernel_size=3)
    conv.causal_padding = 2
    hidden = torch.randn(2, 300, 3, generator=generator)
    channels = hidden.transpose(1, 2).contiguous()
    step = PageStep(shared, (pages[0],), tuple(pages[1:]))
    expected = projection(hidden)
    with target.apply_pages(step):
        result = target.project(projection, hidden)
        assert torch.equal(
            result,
            torch.cat((shared.key_values[0], pages[0].key_values[0], expected), dim=1),
        )
        actual = target.convolve(conv, channels)
    assert torch.equal(
        actual,
        F.conv1d(
            torch.cat((pages[0].convolution_tails[0], channels), dim=-1),
            conv.weight,
            conv.bias,
        ),
    )
    for page_number, page in enumerate(pages[1:]):
        start = page_number * 100
        assert torch.equal(page.key_values[0], expected[:, start : start + 100])
        assert torch.equal(
            page.convolution_tails[0], channels[..., start + 98 : start + 100]
        )
