# SPDX-License-Identifier: Apache-2.0
"""Per-forward CPU indices for eager native attention."""

import logging
from dataclasses import dataclass, replace

import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

logger = logging.getLogger(__name__)


def cpu_indices(
    device_indices: torch.Tensor, mirror: torch.Tensor | list[int] | None
) -> torch.Tensor:
    """Snapshot a current mirror, or read the whole vector once."""
    if mirror is None:
        return device_indices.to(device="cpu", copy=True)
    elif isinstance(mirror, torch.Tensor):
        if mirror.device.type != "cpu":
            raise ValueError("Attention metadata mirrors must reside on CPU")
        else:
            indices = mirror.to(dtype=device_indices.dtype, copy=True)
    else:
        indices = torch.tensor(mirror, dtype=device_indices.dtype, device="cpu")
    if indices.shape != device_indices.shape:
        raise ValueError("Attention metadata mirror shape differs from device indices")
    else:
        return indices


@dataclass(frozen=True, kw_only=True)
class HostAttentionMetadata:
    batch_identity: int
    sequence_lengths: torch.Tensor
    request_slots: torch.Tensor
    extend_prefix_lengths: torch.Tensor | None
    extend_lengths: torch.Tensor | None


class MossSpeechTorchNativeBackend(TorchNativeAttnBackend):
    """Delegate native SDPA and KV writes using small CPU index snapshots."""

    host_metadata: HostAttentionMetadata | None = None
    executions: int = 0

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        self.host_metadata = None
        if forward_batch.forward_mode == ForwardMode.IDLE:
            super().init_forward_metadata(forward_batch)
            return
        elif forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise ValueError(
                "MOSS host attention metadata requires ordinary extend or decode"
            )
        elif forward_batch.encoder_lens is not None:
            raise ValueError(
                "MOSS host attention metadata supports decoder-only attention"
            )
        else:
            pass
        super().init_forward_metadata(forward_batch)
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            assert forward_batch.extend_prefix_lens is not None
            assert forward_batch.extend_seq_lens is not None
            prefix_lengths = cpu_indices(
                forward_batch.extend_prefix_lens, forward_batch.extend_prefix_lens_cpu
            )
            extend_lengths = cpu_indices(
                forward_batch.extend_seq_lens, forward_batch.extend_seq_lens_cpu
            )
            request_slots_mirror = forward_batch.req_pool_indices_cpu
        else:
            prefix_lengths = None
            extend_lengths = None
            # note (Codex): SGLang only guarantees fresh request-slot mirrors for ordinary extend.
            request_slots_mirror = None
        self.host_metadata = HostAttentionMetadata(
            batch_identity=id(forward_batch),
            sequence_lengths=cpu_indices(
                forward_batch.seq_lens, forward_batch.seq_lens_cpu
            ),
            request_slots=cpu_indices(
                forward_batch.req_pool_indices, request_slots_mirror
            ),
            extend_prefix_lengths=prefix_lengths,
            extend_lengths=extend_lengths,
        )

    def host_batch_view(self, forward_batch: ForwardBatch) -> ForwardBatch:
        metadata = self.host_metadata
        if metadata is None or metadata.batch_identity != id(forward_batch):
            raise RuntimeError(
                "Initialize host attention metadata for this forward batch first"
            )
        else:
            # note (Codex): The backend retains no batch, activation, or GPU metadata references.
            return replace(
                forward_batch,
                seq_lens=metadata.sequence_lengths,
                req_pool_indices=metadata.request_slots,
                extend_prefix_lens=metadata.extend_prefix_lengths,
                extend_seq_lens=metadata.extend_lengths,
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        output = super().forward_extend(
            q, k, v, layer, self.host_batch_view(forward_batch), save_kv_cache
        )
        self.executions += 1
        if self.executions == 1:
            logger.info("MOSS host attention metadata execution active")
        else:
            pass
        return output

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        output = super().forward_decode(
            q, k, v, layer, self.host_batch_view(forward_batch), save_kv_cache
        )
        self.executions += 1
        if self.executions == 1:
            logger.info("MOSS host attention metadata execution active")
        else:
            pass
        return output
