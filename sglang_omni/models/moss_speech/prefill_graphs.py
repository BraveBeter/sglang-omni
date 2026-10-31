# SPDX-License-Identifier: Apache-2.0
"""Startup-only exact-length B1 prefill capture with private KV and owned output."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor import forward_context as context_api
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from sglang_omni.models.moss_speech.model_layers import reference_rope_frequencies

logger = logging.getLogger(__name__)


def validate_lengths(lengths):
    if lengths is None:
        return ()
    elif (
        not isinstance(lengths, (list, tuple))
        or not 1 <= len(lengths) <= 8
        or any(type(n) is not int or not 1 <= n <= 512 for n in lengths)
        or len(set(lengths)) != len(lengths)
    ):
        raise ValueError(
            "prefill_graph_lengths requires 1..8 unique exact lengths in [1, 512]"
        )
    else:
        return tuple(sorted(lengths))


def execution_policy():
    cuda = torch.backends.cuda
    return (
        cuda.flash_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.cudnn_sdp_enabled(),
        tuple(torch._C._get_sdp_priority_order()),  # noqa: leading-underscore
        cuda.matmul.allow_tf32,
        cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(),
    )


class PrefillEntry:
    """Native attention operates exclusively on this entry's private fixed KV."""

    def __init__(self, model, length):
        self.length, self.model = length, model
        self.device = model.embed_tokens.weight.device
        self.graph = self.output = None
        self.pool = self.backend = None
        try:
            with torch.inference_mode(), torch.cuda.device(self.device):
                req = ReqToTokenPool(
                    size=1,
                    max_context_len=length,
                    device=str(self.device),
                    enable_memory_saver=False,
                )
                self.pool = MHATokenToKVPool(
                    size=length,
                    page_size=1,
                    dtype=torch.bfloat16,
                    head_num=model.config.num_key_value_heads,
                    head_dim=model.head_dim,
                    layer_num=model.total_attention_layers,
                    device=str(self.device),
                    enable_memory_saver=False,
                    enable_alt_stream=False,
                )
                self.backend = TorchNativeAttnBackend(
                    SimpleNamespace(
                        device=str(self.device),
                        req_to_token_pool=req,
                        token_to_kv_pool=self.pool,
                    )
                )
                req.req_to_token[1, :length] = torch.arange(
                    1, length + 1, device=self.device
                )
                self.ids = torch.zeros(length, device=self.device, dtype=torch.long)
                self.positions = torch.arange(length, device=self.device)
                self.embeds = torch.zeros(
                    (length, model.hidden_size),
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                # These exact constants replace host scalar reads in native SDPA.
                # Request indexing and KV scatter are private; no live metadata is frozen.
                self.batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=self.ids,
                    positions=self.positions,
                    req_pool_indices=torch.tensor([1], device="cpu"),
                    seq_lens=torch.tensor([length], device="cpu"),
                    out_cache_loc=torch.arange(1, length + 1, device=self.device),
                    seq_lens_sum=length,
                    extend_num_tokens=length,
                    extend_seq_lens=torch.tensor([length], device="cpu"),
                    extend_prefix_lens=torch.tensor([0], device="cpu"),
                    extend_seq_lens_cpu=[length],
                    extend_prefix_lens_cpu=[0],
                    rids=["capture"],
                )
                self.backend.init_forward_metadata(self.batch)
                self.rope = reference_rope_frequencies(
                    model.head_dim, model.config.rope_theta
                ).to(self.device)
                self.last_indices = torch.arange(length - 1, length, device=self.device)
                warmup = torch.cuda.Stream(device=self.device)
                warmup.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(warmup):
                    for _ in range(2):
                        self.body()
                torch.cuda.current_stream(self.device).wait_stream(warmup)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.output = self.body()
        except BaseException:
            self.close()
            raise

    def body(self):
        context = (
            replace(context_api.get_forward_context(), attn_backend=self.backend)
            if context_api.has_forward_context()
            else context_api.ForwardContext(attn_backend=self.backend)
        )
        with context_api.forward_context(context):
            return self.model.forward_eager(
                self.ids,
                self.positions,
                self.batch,
                self.embeds,
                rope_frequencies=self.rope,
                last_token_indices=self.last_indices,
            )

    def replay(self, positions, embeds, target, locations):
        with torch.inference_mode():
            self.positions.copy_(positions)
            self.embeds.copy_(embeds)
            self.graph.replay()
            for source, destination in zip(
                (*self.pool.k_buffer, *self.pool.v_buffer),
                (*target.k_buffer, *target.v_buffer),
            ):
                destination.index_copy_(0, locations, source[1:])
        output = self.output
        return type(output)(
            next_token_logits=output.text_logits.clone(),
            text_logits=output.text_logits.clone(),
            audio_logits=output.audio_logits.clone(),
            hidden_states=output.hidden_states.clone(),
        )

    def close(self):
        if self.graph is not None:
            torch.cuda.synchronize(self.device)
            self.graph.reset()
            self.graph = self.output = None
        else:
            pass
        self.pool = self.backend = None


class PrefillGraphs:
    """A finite explicit bank; no padding, request-time capture or implicit dependency."""

    def __init__(self, model, backend, lengths):
        self.model, self.backend = model, backend
        self.lengths = validate_lengths(lengths)
        self.device = model.embed_tokens.weight.device
        self.lock = threading.RLock()
        self.entries = {}
        self.closed = False
        self.hits = 0
        self.signature = self.stream = None
        self.reason = None

    def identity(self):
        pool = self.backend.token_to_kv_pool
        return (
            id(pool),
            id(self.backend.req_to_token_pool),
            tuple(
                (
                    name,
                    id(p),
                    p.data_ptr(),
                    p.shape,
                    p.dtype,
                    p.device,
                    (
                        None
                        if p.is_inference() or name == "decode_input_embedding.weight"
                        else p._version
                    ),
                )  # noqa: leading-underscore
                for name, p in self.model.named_parameters()
            ),
            tuple(
                (id(t), t.data_ptr(), t.shape, t.dtype, t.device)
                for t in (*pool.k_buffer, *pool.v_buffer)
            ),
            execution_policy(),
        )

    def initialize(self):
        with self.lock:
            if self.closed or self.entries:
                return
            elif (
                self.device.type != "cuda"
                or type(self.backend) is not TorchNativeAttnBackend
                or self.backend.use_sliding_window_kv_pool
                or self.model.training
                or torch.__version__ != "2.14.1+cu130"
                or torch.version.cuda != "13.0"
                or torch.cuda.get_device_capability(self.device) != (8, 0)
            ):
                self.reason = "unqualified runtime, model or backend"
                logger.info(f"MOSS prefill graph fallback: {self.reason}")
                return
            else:
                pass
            try:
                for length in self.lengths:
                    self.entries[length] = PrefillEntry(self.model, length)
                torch.cuda.synchronize(self.device)
                self.stream = torch.cuda.current_stream(self.device).cuda_stream
                self.signature = self.identity()
            except BaseException:
                self.close()
                raise
            logger.info(f"MOSS prefill graph captured lengths={self.lengths}")

    def try_replay(self, ids, positions, batch, input_embeds=None):
        with self.lock:
            if (
                self.closed
                or not self.entries
                or input_embeds is None
                or torch.is_grad_enabled()
                or self.model.training
                or not batch.forward_mode.is_extend()
                or batch.batch_size != 1
                or batch.encoder_lens is not None
                or batch.spec_info is not None
                or batch.extend_prefix_lens_cpu != [0]
                or not context_api.has_forward_context()
                or context_api.get_attn_backend() is not self.backend
                or torch.cuda.current_stream(self.device).cuda_stream != self.stream
                or self.identity() != self.signature
                or any(
                    m._forward_hooks or m._forward_pre_hooks
                    for m in self.model.modules()
                )  # noqa: leading-underscore
            ):
                return None
            else:
                pass
            lengths = batch.extend_seq_lens_cpu
            if lengths is None or len(lengths) != 1 or lengths[0] not in self.entries:
                return None
            else:
                pass
            entry = self.entries[lengths[0]]
            locations = batch.out_cache_loc
            if (
                positions.shape != (entry.length,)
                or positions.dtype != torch.int64
                or positions.device != self.device
                or input_embeds.shape != entry.embeds.shape
                or input_embeds.dtype != entry.embeds.dtype
                or input_embeds.device != self.device
                or locations.shape != (entry.length,)
                or locations.dtype != torch.int64
                or locations.device != self.device
            ):
                return None
            else:
                pass
            output = entry.replay(
                positions, input_embeds, self.backend.token_to_kv_pool, locations
            )
            self.hits += 1
            return output

    def close(self):
        with self.lock:
            if self.closed:
                return
            else:
                pass
            for entry in self.entries.values():
                entry.close()
            self.entries.clear()
            self.closed = True
            logger.info(f"MOSS prefill graph closed replays={self.hits}")
