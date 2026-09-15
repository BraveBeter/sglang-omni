# SPDX-License-Identifier: Apache-2.0
"""Host metadata reuses the upstream attention kernels and KV writes."""
import tests.unit_test.moss_speech.sglang_cpu_env  # noqa: F401  # isort: skip

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

from sglang_omni.models.moss_speech.attention import MossSpeechTorchNativeBackend


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_host_metadata_is_per_forward_and_does_not_mutate_batch(monkeypatch, mode):
    pool = SimpleNamespace()
    runner = SimpleNamespace(
        device="cpu", req_to_token_pool=pool, token_to_kv_pool=pool
    )
    backend = MossSpeechTorchNativeBackend(runner)
    q = torch.randn(2, 4)
    layer = object()
    seen = []

    def forward(self, query, key, value, layer_arg, batch, save_kv_cache=True):
        assert self is backend and query is q and layer_arg is layer
        seen.append(batch)
        assert save_kv_cache is False
        return query

    monkeypatch.setattr(TorchNativeAttnBackend, "forward_" + mode, forward)
    for indices in [[3, 5], [7, 2]]:
        fb = SimpleNamespace(
            out_cache_loc=torch.tensor([6, 9]),
            seq_lens=torch.tensor([10, 20]),
            seq_lens_cpu=torch.tensor([10, 20]),
            req_pool_indices=torch.tensor(indices),
            extend_prefix_lens=torch.tensor([8, 17]),
            extend_prefix_lens_cpu=[8, 17],
            extend_seq_lens=torch.tensor([2, 3]),
            extend_seq_lens_cpu=[2, 3],
            encoder_lens=None,
        )
        original = vars(fb).copy()
        backend.init_forward_metadata(fb)
        for _ in range(2):
            result = getattr(backend, "forward_" + mode)(q, q, q, layer, fb, False)
            assert result is q
        assert seen[-1] is seen[-2] and seen[-1] is not fb
        assert seen[-1].req_pool_indices.tolist() == indices
        assert seen[-1].seq_lens.tolist() == fb.seq_lens_cpu.tolist()
        assert seen[-1].out_cache_loc is fb.out_cache_loc
        assert seen[-1].extend_prefix_lens.tolist() == [8, 17]
        assert all(getattr(fb, name) is value for name, value in original.items())
    assert seen[0] is not seen[-1]


def test_indexing_metadata_uses_python_integers():
    pool = SimpleNamespace()
    backend = MossSpeechTorchNativeBackend(
        SimpleNamespace(device="cpu", req_to_token_pool=pool, token_to_kv_pool=pool)
    )
    fb = SimpleNamespace(
        out_cache_loc=None,
        seq_lens=torch.tensor([12]),
        seq_lens_cpu=torch.tensor([12]),
        req_pool_indices=torch.tensor([3]),
    )
    backend.init_forward_metadata(fb)
    assert type(backend._host_batch.seq_lens[0]) is int
    assert backend._host_batch.seq_lens.shape == (1,)
    assert type(backend._host_batch.req_pool_indices[0]) is int
