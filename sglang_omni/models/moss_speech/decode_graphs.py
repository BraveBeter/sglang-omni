# SPDX-License-Identifier: Apache-2.0
"""Finite exact-length B1 decode graphs retaining the native SDPA kernel shapes."""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor import forward_context as context_api
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from sglang_omni.models.moss_speech.model_layers import reference_rope_frequencies

logger = logging.getLogger(__name__)
# Cover the supported 512-row prompt range; longer decode falls back to eager.
# Each exact SDPA shape keeps BF16 rounding; no capacity padding or varlen Flash.
LENGTHS = tuple(range(1, 513))


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


class DecodeBackend(TorchNativeAttnBackend):
    """Native KV writes and identical B1 SDPA layouts, with a fixed host length."""

    def _run_sdpa_forward_decode(  # noqa: leading-underscore
        self,
        query,
        output,
        k_cache,
        v_cache,
        req_to_token,
        req_pool_indices,
        seq_lens,
        encoder_lens=None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size=None,
    ):
        assert encoder_lens is None and not causal and not is_cross_attn
        assert sliding_window_size is None
        tokens = req_to_token[req_pool_indices[:1], : self.length].reshape(self.length)
        q = query.movedim(0, 1).unsqueeze(0)
        k = k_cache[tokens].movedim(0, 1).unsqueeze(0)
        v = v_cache[tokens].movedim(0, 1).unsqueeze(0)
        result = (
            torch.nn.functional.scaled_dot_product_attention(
                q, k, v, enable_gqa=enable_gqa, scale=scaling, is_causal=False
            )
            .squeeze(0)
            .movedim(1, 0)
        )
        output.copy_(result)
        return output


class DecodeEntry:
    def __init__(self, model, backend, length, rope):
        self.model, self.length = model, length
        self.device = model.decode_input_embedding.weight.device
        self.graph = self.output = None
        self.rope = rope
        try:
            with torch.inference_mode(), torch.cuda.device(self.device):
                self.ids = torch.zeros(1, dtype=torch.int64, device=self.device)
                self.positions = torch.zeros_like(self.ids)
                self.req = torch.zeros_like(self.ids)
                self.locations = torch.zeros_like(self.ids)
                self.last_indices = torch.zeros_like(self.ids)
                self.batch = ForwardBatch(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=1,
                    input_ids=self.ids,
                    positions=self.positions,
                    req_pool_indices=self.req,
                    seq_lens=torch.tensor([length], device="cpu"),
                    out_cache_loc=self.locations,
                    seq_lens_sum=length,
                    rids=["capture"],
                )
                self.backend = DecodeBackend(backend)
                self.backend.length = length
                self.backend.init_forward_metadata(self.batch)
                warmup = torch.cuda.Stream(device=self.device)
                warmup.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(warmup):
                    for _ in range(2):
                        self.body()
                torch.cuda.current_stream(self.device).wait_stream(warmup)
                self.graph = torch.cuda.CUDAGraph()
                # Separate pools permit arbitrary replay order across exact lengths.
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
                rope_frequencies=self.rope,
                last_token_indices=self.last_indices,
            )

    def replay(self, inputs):
        with torch.inference_mode():
            for target, actual in inputs:
                target.copy_(actual)
            self.graph.replay()
        output = self.output
        return type(output)(
            next_token_logits=output.text_logits.clone(),
            text_logits=output.text_logits.clone(),
            audio_logits=output.audio_logits.clone(),
            hidden_states=output.hidden_states.clone(),
        )

    def close(self):
        # note (Codex): Warmup may fail before a graph exists while work is in flight.
        torch.cuda.synchronize(self.device)
        if self.graph is not None:
            self.graph.reset()
            self.graph = self.output = None
        else:
            pass


class DecodeGraph:
    """Startup-only exact bank with independent buffers and serialized teardown."""

    def __init__(self, model, backend, *, lengths=LENGTHS):
        self.model, self.backend = model, backend
        self.lengths = tuple(lengths)
        if (
            not self.lengths
            or len(set(self.lengths)) != len(self.lengths)
            or any(type(n) is not int or n not in LENGTHS for n in self.lengths)
        ):
            raise ValueError("Decode graph lengths must be unique integers in [1, 512]")
        else:
            pass
        self.device = model.decode_input_embedding.weight.device
        self.lock = threading.RLock()
        self.entries = {}
        self.hits = 0
        self.closed = False
        self.reason = self.stream = self.signature = None

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
            self.backend.req_to_token_pool.req_to_token.data_ptr(),
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
                or any(
                    m._forward_hooks or m._forward_pre_hooks
                    for m in self.model.modules()
                )  # noqa: leading-underscore
                or torch.__version__ != "2.14.1+cu130"
                or torch.version.cuda != "13.0"
                or torch.cuda.get_device_capability(self.device) != (8, 0)
                or self.backend.req_to_token_pool.req_to_token.shape[1]
                < max(self.lengths)
            ):
                self.reason = "unqualified runtime, model, backend, pool or hooks"
                logger.info(f"MOSS decode graph fallback: {self.reason}")
                return
            else:
                pass
            pool = self.backend.token_to_kv_pool
            tensors = [
                self.backend.req_to_token_pool.req_to_token,
                *pool.k_buffer,
                *pool.v_buffer,
            ]
            with torch.inference_mode(), torch.cuda.device(self.device):
                saved = [t[0].clone() for t in tensors]
                rope = reference_rope_frequencies(
                    self.model.head_dim, self.model.config.rope_theta
                ).to(self.device)
                try:
                    for t in tensors:
                        t[0].zero_()
                    for length in self.lengths:
                        self.entries[length] = DecodeEntry(
                            self.model, self.backend, length, rope
                        )
                    torch.cuda.synchronize(self.device)
                except BaseException:
                    self.close()
                    raise
                finally:
                    for t, value in zip(tensors, saved):
                        t[0].copy_(value)
            self.signature = self.identity()
            self.stream = torch.cuda.current_stream(self.device).cuda_stream
            logger.info(f"MOSS exact decode graphs captured count={len(self.entries)}")

    def try_replay(self, ids, positions, batch, input_embeds=None):
        with self.lock:
            if (
                self.closed
                or not self.entries
                or input_embeds is not None
                or torch.is_grad_enabled()
                or self.model.training
                or not batch.forward_mode.is_decode()
                or batch.batch_size != 1
                or batch.encoder_lens is not None
                or batch.spec_info is not None
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
            cpu = batch.seq_lens_cpu
            if (
                cpu is None
                or (isinstance(cpu, torch.Tensor) and cpu.device.type != "cpu")
                or len(cpu) != 1
                or int(cpu[0]) not in self.entries
            ):
                return None
            else:
                pass
            entry = self.entries[int(cpu[0])]
            inputs = (
                (entry.ids, ids),
                (entry.positions, positions),
                (entry.req, batch.req_pool_indices),
                (entry.locations, batch.out_cache_loc),
            )
            if any(
                not isinstance(value, torch.Tensor)
                or value.shape != target.shape
                or value.dtype != target.dtype
                or value.device != target.device
                or value.stride() != target.stride()
                for target, value in inputs
            ):
                return None
            else:
                pass
            output = entry.replay(inputs)
            self.hits += 1
            if self.hits == 1:
                logger.info("MOSS full decode graph replay active")
            else:
                pass
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
            logger.info(f"MOSS decode graph closed replays={self.hits}")
