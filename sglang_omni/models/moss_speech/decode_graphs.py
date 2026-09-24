# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded B1 decode graphs for the qualified native MOSS runtime.

Sampling, request ownership and prefill stay outside these graphs. The private
Flash operation is intentionally restricted to the runtime tested for exact
BF16 logits; unsupported runtimes continue through the ordinary model forward.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Callable, Iterator

import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor import forward_context as context_api
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from .attention import MossSpeechTorchNativeBackend, _HostIndexVector
from .decode_norm_pointwise import close_norm_owner, norm_capture

logger = logging.getLogger(__name__)
CAPACITIES = (128, 512, 1024)


def decode_graph_capacity(
    *, is_decode: bool, batch_size: int, seq_len: int
) -> int | None:
    if not is_decode or batch_size != 1 or seq_len < 1:
        return None
    return next((size for size in CAPACITIES if seq_len <= size), None)


def sdpa_policy() -> tuple:
    """Snapshot dispatch controls; a changed policy must retain native behavior."""
    cuda = torch.backends.cuda
    return (
        cuda.flash_sdp_enabled(),
        cuda.mem_efficient_sdp_enabled(),
        cuda.math_sdp_enabled(),
        cuda.cudnn_sdp_enabled(),
        tuple(getattr(torch._C, "_get_sdp_priority_order", lambda: ())()),
    )


class DecodeGraphCache:
    """Own a fixed startup allocation and serialize replay with destruction."""

    def __init__(self, *, synchronize: Callable[[], None]) -> None:
        self.lock = threading.RLock()
        self._synchronize = synchronize
        self._entries: dict[int, Any] = {}
        self._ready = False
        self._closed = False

    @property
    def size(self) -> int:
        with self.lock:
            return len(self._entries)

    def initialize(self, factory: Callable[[int], Any]) -> None:
        with self.lock:
            if self._ready or self._closed:
                return
            try:
                for capacity in CAPACITIES:
                    self._entries[capacity] = factory(capacity)
                self._ready = True
            except BaseException:
                self.close()
                raise

    def replay(self, capacity: int, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            if self._closed or not self._ready:
                return None
            entry = self._entries.get(capacity)
            return None if entry is None else entry.replay(*args, **kwargs)

    def close(self) -> None:
        with self.lock:
            if self._closed:
                return
            self._closed = True
            self._ready = False
            if self._entries:
                # stop() may call us from another thread while GPU work is queued.
                self._synchronize()
                for entry in self._entries.values():
                    entry.close()
                self._entries.clear()


@contextmanager
def scoped_backend(backend: Any) -> Iterator[None]:
    context = (
        replace(context_api.get_forward_context(), attn_backend=backend)
        if context_api.has_forward_context()
        else context_api.ForwardContext(attn_backend=backend)
    )
    with context_api.forward_context(context):
        yield


@contextmanager
def projection_mode(model: Any, enabled: bool) -> Iterator[None]:
    layers = list(model.layers) + list(model.text_block) + list(model.audio_block)
    previous = [getattr(layer, "enable_projection_graphs", False) for layer in layers]
    try:
        for layer in layers:
            layer.enable_projection_graphs = enabled
        yield
    finally:
        for layer, value in zip(layers, previous):
            layer.enable_projection_graphs = value


@contextmanager
def preserve_warmup_state(model: Any, runtime: Any) -> Iterator[None]:
    """Borrow only SGLang's reserved request/KV slot zero before stage ready."""
    pool = runtime.token_to_kv_pool
    tensors = [
        model._decode_input_embedding.weight,
        runtime.req_to_token_pool.req_to_token,
    ]
    tensors += list(pool.k_buffer) + list(pool.v_buffer)
    last = {
        name: value for name, value in vars(model).items() if name.startswith("_last_")
    }
    stash = model._dual_logits_by_rid.copy()
    device = model._decode_input_embedding.weight.device
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
        saved = [tensor[0].clone() for tensor in tensors]
        try:
            for tensor in tensors:
                tensor[0].zero_()
            yield
        finally:
            for tensor, value in zip(tensors, saved):
                tensor[0].copy_(value)
            for name in tuple(vars(model)):
                if name.startswith("_last_") and name not in last:
                    delattr(model, name)
            for name, value in last.items():
                setattr(model, name, value)
            model._dual_logits_by_rid.clear()
            model._dual_logits_by_rid.update(stash)


def owned_decode_output(model: Any, raw: Any, batch: Any) -> Any:
    from .sglang_model import MossSpeechModelOutput, publish_dual_logits

    text, audio, hidden = (
        raw.text_logits.clone(),
        raw.audio_logits.clone(),
        raw.hidden_states.clone(),
    )
    output = MossSpeechModelOutput(
        next_token_logits=text,
        text_logits=text,
        audio_logits=audio,
        hidden_states=hidden,
    )
    # Keep the existing compute_dual_logits contract without retaining graph outputs.
    model._last_text_hidden, model._last_audio_hidden = hidden[0], hidden[1]
    model._last_head_indices = torch.arange(text.shape[0], device=text.device)
    model._last_request_lengths = [1] * text.shape[0]
    model._last_prefix_lengths = [0] * text.shape[0]
    batch._moss_request_lengths = model._last_request_lengths
    batch._moss_prefix_lengths = model._last_prefix_lengths
    publish_dual_logits(model, output, batch)
    batch._moss_dual_logits = (text, audio)
    return output


class _VarlenNativeBackend(TorchNativeAttnBackend):
    """Retain native KV writes, with capacity gathers and GPU actual KV length."""

    def prepare(self, batch: Any, lengths: torch.Tensor) -> None:
        selected = self.req_to_token_pool.req_to_token[
            batch.req_pool_indices[:1], : self.capacity
        ].reshape(self.capacity)
        self.tokens = torch.where(self.indices < lengths[0], selected, 0)
        self.cu_k[1:].copy_(lengths)

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: Any,
        req_pool_indices: Any,
        seq_lens: Any,
        encoder_lens: Any = None,
        scaling: float | None = None,
        enable_gqa: bool = False,
        causal: bool = False,
        is_cross_attn: bool = False,
        sliding_window_size: int | None = None,
    ) -> torch.Tensor:
        if (
            query.shape != (1, 32, 128)
            or encoder_lens is not None
            or causal
            or is_cross_attn
            or sliding_window_size is not None
        ):
            raise ValueError("Full decode graph requires native B1 MOSS attention")
        key, value = k_cache[self.tokens], v_cache[self.tokens]
        if not (query.dtype == key.dtype == value.dtype == torch.bfloat16):
            raise ValueError("Full decode graph requires BF16 Q/K/V")
        result = torch.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            self.cu_q,
            self.cu_k,
            1,
            self.capacity,
            0.0,
            False,
            False,
            scale=scaling,
        )[0]
        output[:1, :, :] = result
        return output


class _DecodeGraph:
    def __init__(self, model: Any, runtime: Any, capacity: int) -> None:
        self.model, self.capacity = model, capacity
        self.norm_owner = getattr(model, "_moss_decode_norm_owner", None)
        device = model._decode_input_embedding.weight.device
        self.input_ids = torch.zeros(1, device=device, dtype=torch.int64)
        self.positions = torch.zeros_like(self.input_ids)
        self.lengths = torch.ones_like(self.input_ids)
        self.batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=self.input_ids,
            positions=self.positions,
            req_pool_indices=torch.zeros_like(self.input_ids),
            seq_lens=_HostIndexVector([capacity]),
            out_cache_loc=torch.zeros_like(self.input_ids),
            seq_lens_sum=capacity,
            seq_lens_cpu=[capacity],
            rids=None,
        )
        self.backend = _VarlenNativeBackend(runtime)
        self.backend.capacity = capacity
        self.backend.indices = torch.arange(capacity, device=device)
        self.backend.cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        self.backend.cu_k = torch.tensor([0, 1], device=device, dtype=torch.int32)
        self.backend.init_forward_metadata(self.batch)
        self.graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.device(device),
            projection_mode(model, False),
        ):
            warmup = torch.cuda.Stream(device=device)
            warmup.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup):
                for _ in range(2):
                    self._forward()
            torch.cuda.current_stream(device).wait_stream(warmup)
            with torch.cuda.graph(self.graph):
                self.outputs = self._forward()
        self.replays = 0

    def _forward(self) -> Any:
        self.backend.prepare(self.batch, self.lengths)
        with (
            scoped_backend(self.backend),
            norm_capture(self.norm_owner, self.model, self.capacity),
        ):
            return self.model._forward_eager(
                input_ids=self.input_ids,
                positions=self.positions,
                forward_batch=self.batch,
            )

    def accepts(self, batch: Any, input_ids: Any, positions: Any) -> bool:
        return all(
            isinstance(actual, torch.Tensor)
            and actual.shape == target.shape
            and actual.dtype == target.dtype
            and actual.device == target.device
            and actual.stride() == target.stride()
            for target, actual in self._inputs(batch, input_ids, positions)
        )

    def _inputs(self, batch: Any, input_ids: Any, positions: Any) -> tuple:
        return (
            (self.input_ids, input_ids),
            (self.positions, positions),
            (self.lengths, batch.seq_lens),
            (self.batch.req_pool_indices, batch.req_pool_indices),
            (self.batch.out_cache_loc, batch.out_cache_loc),
        )

    def replay(
        self, batch: Any, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> Any:
        if not self.accepts(batch, input_ids, positions):
            return None
        # Startup captures create inference tensors. The scheduler may call
        # decode under no_grad, so writes to these private buffers need the
        # same mode. Clone outputs after leaving it: the sampler mutates its
        # owned logits in the caller's ordinary inference context.
        with torch.inference_mode():
            for target, actual in self._inputs(batch, input_ids, positions):
                target.copy_(actual)
            self.graph.replay()
        self.replays += 1
        return owned_decode_output(self.model, self.outputs, batch)

    def close(self) -> None:
        self.outputs = None
        self.graph.reset()


class MossSpeechDecodeGraphManager:
    """No request-time capture: unsupported calls immediately use eager decode."""

    def __init__(self, model: Any, runtime: Any) -> None:
        self.model, self.runtime = model, runtime
        self.device = model._decode_input_embedding.weight.device
        self.backend = runtime.attn_backend
        self.pool = runtime.token_to_kv_pool
        self.request_pool = runtime.req_to_token_pool
        self.cache = DecodeGraphCache(
            synchronize=lambda: torch.cuda.synchronize(self.device)
        )
        self._storage: tuple = ()
        self._policy = sdpa_policy()
        self.reason: str | None = None
        self.startup_seconds = 0.0
        self.hits = 0

    def _storage_key(self) -> tuple:
        return (
            self.model._decode_input_embedding.weight.data_ptr(),
            self.request_pool.req_to_token.data_ptr(),
            *(
                tensor.data_ptr()
                for tensor in list(self.pool.k_buffer) + list(self.pool.v_buffer)
            ),
        )

    def _qualified(self) -> bool:
        from .sglang_model import MossSpeechSGLangModel

        if type(self.model) is not MossSpeechSGLangModel or type(self.backend) not in (
            TorchNativeAttnBackend,
            MossSpeechTorchNativeBackend,
        ):
            return False
        if (
            self.device.type != "cuda"
            or torch.__version__ != "2.9.1+cu128"
            or torch.version.cuda != "12.8"
        ):
            return False
        if torch.cuda.get_device_name(
            self.device
        ) != "NVIDIA A800-SXM4-80GB" or torch.cuda.get_device_capability(
            self.device
        ) != (
            8,
            0,
        ):
            return False
        if (
            not torch.backends.cuda.flash_sdp_enabled()
            or self.backend.use_sliding_window_kv_pool
        ):
            return False
        layers = (
            list(self.model.layers)
            + list(self.model.text_block)
            + list(self.model.audio_block)
        )
        if (
            len(self.model.layers),
            len(self.model.text_block),
            len(self.model.audio_block),
        ) != (32, 4, 4):
            return False
        if self.request_pool.req_to_token.shape[1] < CAPACITIES[-1]:
            return False
        if not hasattr(self.pool, "k_buffer") or not hasattr(self.pool, "v_buffer"):
            return False
        buffers = list(self.pool.k_buffer) + list(self.pool.v_buffer)
        if len(buffers) != 80 or any(
            t.dtype != torch.bfloat16
            or t.device != self.device
            or t.shape[1:] != (8, 128)
            for t in buffers
        ):
            return False
        for layer in layers:
            attention = layer.self_attn.attn
            if (
                attention.tp_q_head_num != 32
                or attention.tp_k_head_num != 8
                or attention.qk_head_dim != 128
                or attention.v_head_dim != 128
                or attention.is_cross_attention
                or getattr(attention, "sliding_window_size", -1) not in (-1, None)
            ):
                return False
        return (
            not os.environ.get("MOSS_DUMP_DIR")
            and self.model._decode_input_embedding.weight.dtype == torch.bfloat16
        )

    def initialize(self) -> None:
        if not self._qualified():
            self.reason = "unqualified runtime, model, or Flash policy"
            logger.info("MOSS full decode graphs disabled: %s", self.reason)
            return
        # The direct Flash primitive is valid only when ordinary SDPA selects
        # Flash for these same native BF16 shapes under the current policy.
        query = torch.empty((1, 32, 1, 128), device=self.device, dtype=torch.bfloat16)
        for length in (1, *CAPACITIES):
            key = torch.empty(
                (1, 8, length, 128), device=self.device, dtype=torch.bfloat16
            )
            if (
                int(
                    torch.ops.aten._fused_sdp_choice(
                        query,
                        key,
                        key,
                        None,
                        0.0,
                        False,
                        scale=128**-0.5,
                        enable_gqa=True,
                    )
                )
                != 1
            ):
                self.reason = "native SDPA does not select Flash"
                logger.info("MOSS full decode graphs disabled: %s", self.reason)
                return
        self._policy = sdpa_policy()
        start = time.perf_counter()
        try:
            with preserve_warmup_state(self.model, self.runtime):
                self.cache.initialize(
                    lambda capacity: _DecodeGraph(self.model, self.runtime, capacity)
                )
            torch.cuda.synchronize(self.device)
            self._storage = self._storage_key()
        except torch.cuda.OutOfMemoryError:
            self.close()
            self.reason = "insufficient memory during startup capture"
            logger.warning("MOSS full decode graphs disabled: %s", self.reason)
            return
        except BaseException:
            if getattr(self.model, "_moss_decode_norm_owner", None) is not None:
                self.close()
            raise
        self.startup_seconds = time.perf_counter() - start
        logger.info(
            "MOSS full decode graphs ready: graphs=%d startup_seconds=%.3f",
            self.cache.size,
            self.startup_seconds,
        )

    def try_replay(
        self, input_ids: Any, positions: Any, batch: Any, input_embeds: Any = None
    ) -> Any:
        with self.cache.lock:
            if (
                not self.cache.size
                or input_embeds is not None
                or batch is None
                or os.environ.get("MOSS_DUMP_DIR")
            ):
                return None
            mode = getattr(batch, "forward_mode", None)
            if (
                mode is None
                or not mode.is_decode()
                or batch.batch_size != 1
                or batch.encoder_lens is not None
                or getattr(batch, "spec_info", None) is not None
            ):
                return None
            cpu = getattr(batch, "seq_lens_cpu", None)
            if (
                cpu is None
                or (isinstance(cpu, torch.Tensor) and cpu.device.type != "cpu")
                or len(cpu) != 1
            ):
                return None
            capacity = decode_graph_capacity(
                is_decode=True, batch_size=1, seq_len=int(cpu[0])
            )
            if (
                capacity is None
                or not context_api.has_forward_context()
                or context_api.get_attn_backend() is not self.backend
                or not torch.backends.cuda.flash_sdp_enabled()
                or sdpa_policy() != self._policy
            ):
                return None
            if (
                self.backend.token_to_kv_pool is not self.pool
                or self.backend.req_to_token_pool is not self.request_pool
                or self.runtime.attn_backend is not self.backend
                or self.runtime.token_to_kv_pool is not self.pool
                or self.runtime.req_to_token_pool is not self.request_pool
                or self._storage_key() != self._storage
            ):
                return None
            rids = getattr(batch, "rids", None)
            if rids is not None and len(rids) != 1:
                return None
            output = self.cache.replay(capacity, batch, input_ids, positions)
            if output is not None:
                self.hits += 1
            return output

    def close(self) -> None:
        if getattr(self.model, "_moss_decode_norm_owner", None) is not None:
            # The first startup body can fail before insertion into the cache.
            # Keep launch code alive until even that queued warmup has drained.
            torch.cuda.synchronize(self.device)
        self.cache.close()
        # Release native launch code only after all owned graphs were reset.
        close_norm_owner(self.model)
