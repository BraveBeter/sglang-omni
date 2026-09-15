# SPDX-License-Identifier: Apache-2.0
"""Eager torch-native attention with host-side indexing metadata."""
from __future__ import annotations

from copy import copy
from typing import Any

import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend


class _HostIndexVector(list):
    """The indexing-only shape interface expected by the upstream SDPA loop."""

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    def tolist(self) -> list[int]:
        return list(self)


class MossSpeechTorchNativeBackend(TorchNativeAttnBackend):
    """Prepare indexing scalars once, retaining upstream attention and KV math.

    The eager backend reads sequence lengths and request slots from Python in
    every layer. Keep only those indexing tensors on the host; activations,
    token mappings and KV tensors remain on their original device.
    """

    def init_forward_metadata(self, forward_batch: Any) -> None:
        super().init_forward_metadata(forward_batch)
        host = copy(forward_batch)
        for name in ("seq_lens", "extend_prefix_lens", "extend_seq_lens"):
            value = getattr(forward_batch, name, None)
            if value is None:
                continue
            cpu = getattr(forward_batch, name + "_cpu", None)
            setattr(
                host,
                name,
                _HostIndexVector(
                    value.cpu().tolist()
                    if cpu is None
                    else torch.as_tensor(cpu).tolist()
                ),
            )
        host.req_pool_indices = _HostIndexVector(
            forward_batch.req_pool_indices.cpu().tolist()
        )
        if getattr(forward_batch, "encoder_lens", None) is not None:
            host.encoder_lens = _HostIndexVector(
                forward_batch.encoder_lens.cpu().tolist()
            )
        self._host_batch = host

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        return super().forward_extend(q, k, v, layer, self._host_batch, save_kv_cache)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        return super().forward_decode(q, k, v, layer, self._host_batch, save_kv_cache)
