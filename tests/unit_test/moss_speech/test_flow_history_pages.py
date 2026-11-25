# SPDX-License-Identifier: Apache-2.0
"""Each committed Euler step retains its own exact closed K/V and convolution tails."""
from contextvars import ContextVar
from threading import get_ident
from unittest.mock import Mock, patch
from weakref import ref

import torch
from torch.nn import functional as F

from sglang_omni.models.moss_speech.flow_history_graph import HistoryTransaction
from sglang_omni.models.moss_speech.flow_history_pages import (
    PagedHistoryPrefill,
    PageLayout,
    PageStep,
    StreamingPagedHistoryGraphs,
)
from sglang_omni.models.moss_speech.flow_prefix import PrefixSolve, ReferencePrefix
from sglang_omni.models.moss_speech.flow_prefix_graph import StreamingReferenceGraphs


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


@torch.inference_mode()
def test_changed_voice_releases_old_graphs_and_retries_its_bank():
    native = StreamingReferenceGraphs.__new__(StreamingReferenceGraphs)
    native.steps = 1
    native.owner_thread = get_ident()
    native.owner_stream = Mock()
    native.reference = native.step_views = native.step_buffer = native.step_bank = None
    native.graphs = {}
    native.estimator = Mock(return_value=torch.zeros(1))
    prior, changed = (
        ReferencePrefix(
            reference_key=bytes([frames]),
            mel_frames=frames,
            key_values=[torch.ones(2, frames, 1)],
            convolution_tails=[torch.zeros(2, 1, 2)],
        )
        for frames in (100, 200)
    )
    prefill = Mock()
    prefill.layout = PageLayout(prior, 1, 100)
    prefill.current = ContextVar("voice_transition")
    prefill.transaction = ContextVar("request", default=None)
    native.prefill = prefill
    paged = StreamingPagedHistoryGraphs(native, prefill)
    with patch.object(torch.cuda, "mem_get_info", return_value=(2**40, 2**41)):
        native.bind(prior)
    native.pool = object()
    paged.reference_buffer = native.step_buffer
    old_buffer = ref(native.step_buffer)
    paged.runners[(4, 3)] = Mock()
    old_runner = ref(paged.runners[(4, 3)])
    prefill.current.set(PrefixSolve(prefix=changed, prefill=True))
    with patch.object(torch.cuda, "current_stream", return_value=native.owner_stream):
        paged(*(torch.zeros(1),) * 6, streaming=True)
    assert old_runner() is None
    prefill.current.set(PrefixSolve(prefix=changed, prefill=False))
    with (
        patch.object(torch.cuda, "current_stream", return_value=native.owner_stream),
        patch.object(torch.cuda, "graph_pool_handle", return_value=object()),
        patch.object(torch.cuda, "mem_get_info", return_value=(0, 2**41)),
    ):
        paged(*(torch.zeros(1),) * 6, streaming=True)
    assert old_buffer() is None and not paged.runners
    assert native.step_bank is None and native.reference is changed
    with (
        patch.object(torch.cuda, "mem_get_info", return_value=(2**40, 2**41)) as budget,
        patch("sglang_omni.models.moss_speech.flow_history_pages.PagedEstimatorGraphs"),
    ):
        assert paged.stage(changed, 0, 0, first_step=False) is None
        budget.assert_not_called()
        assert paged.stage(changed, 0, 0, first_step=True) is not None
        budget.assert_called_once()
    assert native.step_bank is not None and native.step_views.mel_frames == 200
    assert torch.equal(native.step_bank[0, :400], torch.ones(400))


def test_final_bucket_hides_extra_keys_and_returns_only_valid_frames():
    shared = ReferencePrefix(reference_key=b"v", mel_frames=100)
    target = PagedHistoryPrefill.__new__(PagedHistoryPrefill)
    target.current = ContextVar(
        "final-prefix", default=PrefixSolve(prefix=shared, prefill=False)
    )
    target.transaction = ContextVar(
        "final-request", default=HistoryTransaction(history=None, final=True)
    )
    target.layout = Mock()
    target.chunk_frames = 100
    target.graph_final_flow = True
    noise = torch.arange(245, dtype=torch.float32).reshape(1, 1, -1)
    times = torch.tensor([0.0, 1.0])
    mask = torch.ones_like(noise)
    widths = []

    def solve(noise, times, mu, mask, speaker, cond, streaming):
        widths.append(mask.size(-1))
        assert torch.equal(mask[..., :145], torch.ones_like(mask[..., :145]))
        assert torch.count_nonzero(mask[..., 145:]) == 0
        return mu * mask

    for enabled in (True, False):
        target.graph_final_flow = enabled
        result = target.solve_euler(solve, noise, times, noise, mask, None, noise, True)
        assert torch.equal(result[..., 100:], noise[..., 100:])
        assert result.shape == noise.shape
    assert widths == [200, 145]
