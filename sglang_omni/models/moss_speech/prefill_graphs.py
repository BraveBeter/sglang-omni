# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact B1 native prefill graphs, prepared for every length 1..128.

Private KV storage keeps startup independent of request allocation. Replay writes
all forty layers to the actual allocated locations and owns every published output.
Sampling and original request/feedback handling remain outside this module.
"""
from __future__ import annotations

import logging
import operator
import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import torch
from sglang.srt.model_executor import forward_context as context_api
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from .attention import MossSpeechTorchNativeBackend
from .decode_graphs import projection_mode, scoped_backend, sdpa_policy

logger = logging.getLogger(__name__)
LENGTHS = tuple(range(1, 129))


def prefill_length(
    *, is_extend: bool, batch_size: int, length: int, prefix: int
) -> int | None:
    if (
        is_extend
        and batch_size == 1
        and type(length) is int
        and length in LENGTHS
        and prefix == 0
    ):
        return length
    return None


def native_norm_parameter_profile(model: Any) -> bool:
    """Admit only original norm-weight grad flags, without changing tensors.

    Native prefill detaches its full hidden states before the non-grad heads.
    RMSNorm weights retain their constructor flags even in eval mode. Unknown
    trainable registrations still require eager execution, including an allowed
    norm weight aliased as a head or another module's parameter.
    """
    from .sglang_model import MossSpeechRMSNorm

    try:
        groups = (model.layers, model.text_block, model.audio_block)
        if tuple(map(len, groups)) != (32, 4, 4) or model.training:
            return False
        norms = [model.text_norm, model.audio_norm]
        for group in groups:
            for layer in group:
                norms.extend(
                    (
                        layer.input_layernorm,
                        layer.post_attention_layernorm,
                        layer.self_attn.q_norm,
                        layer.self_attn.k_norm,
                    )
                )
        if len({id(norm) for norm in norms}) != 162 or any(
            type(norm) is not MossSpeechRMSNorm
            or norm.training
            or getattr(norm.forward, "__func__", None) is not MossSpeechRMSNorm.forward
            or set(norm._parameters) != {"weight"}
            or not isinstance(norm.weight, torch.nn.Parameter)
            for norm in norms
        ):
            return False
        allowed = {id(norm): norm.weight for norm in norms}
        if len({id(weight) for weight in allowed.values()}) != 162:
            return False
        for module in model.modules():
            for name, parameter in module._parameters.items():
                if parameter is None or not parameter.requires_grad:
                    continue
                if name != "weight" or allowed.get(id(module)) is not parameter:
                    return False
        return True
    except (AttributeError, TypeError):
        return False


def execution_policy() -> tuple:
    matmul = torch.backends.cuda.matmul
    return (
        sdpa_policy(),
        matmul.allow_tf32,
        matmul.allow_fp16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(),
        torch.get_float32_matmul_precision(),
    )


class CapturedLengths:
    """Only replace the known host read, retaining native GPU index arithmetic."""

    def __init__(self, tensor: torch.Tensor, length: int) -> None:
        if length not in LENGTHS or tensor.shape != (1,):
            raise ValueError("Unsupported exact prefill metadata")
        self.tensor, self.length = tensor, length

    def tolist(self) -> list[int]:
        return [self.length]

    def numel(self) -> int:
        return self.tensor.numel()

    def flatten(self) -> torch.Tensor:
        return self.tensor.flatten()


def tensor_signature(value: torch.Tensor) -> tuple:
    return (
        id(value),
        value.data_ptr(),
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        value.dtype,
        value.device,
        value.requires_grad,
        None if value.is_inference() else value._version,
    )


class PrefillGraphCache:
    """Fixed startup-only allocation; replay and destruction share one lock."""

    def __init__(self, *, synchronize: Callable[[], None]) -> None:
        self.lock = threading.RLock()
        self._synchronize = synchronize
        self._entries: dict[int, Any] = {}
        self._ready = self._closed = False

    @property
    def size(self) -> int:
        with self.lock:
            return len(self._entries)

    def initialize(self, factory: Callable[[int], Any]) -> None:
        with self.lock:
            if self._ready or self._closed:
                return
            try:
                for length in LENGTHS:
                    self._entries[length] = factory(length)
                self._ready = True
            except BaseException:
                self.close()
                raise

    def replay(self, length: int, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            if self._closed or not self._ready:
                return None
            entry = self._entries.get(length)
            return None if entry is None else entry.replay(*args, **kwargs)

    def close(self) -> None:
        with self.lock:
            self._ready = False
            if self._closed:
                return
            if self._entries:
                # A synchronization failure retains owners and permits retry.
                self._synchronize()
                for entry in self._entries.values():
                    entry.close()
                self._entries.clear()
            self._closed = True


def scatter_private_kv(
    private: Any, target: Any, locations: torch.Tensor, length: int
) -> None:
    if (
        not isinstance(locations, torch.Tensor)
        or locations.shape != (length,)
        or locations.dtype != torch.int64
        or locations.stride() != (1,)
    ):
        raise ValueError("Expected original one-dimensional allocated KV locations")
    if (
        len(private.k_buffer) != 40
        or len(private.v_buffer) != 40
        or len(target.k_buffer) != 40
        or len(target.v_buffer) != 40
    ):
        raise ValueError("Complete forty-layer KV required")
    for source, destination in zip(
        (*private.k_buffer, *private.v_buffer), (*target.k_buffer, *target.v_buffer)
    ):
        if (
            source.shape[0] != length + 1
            or source.shape[1:] != destination.shape[1:]
            or source.dtype != destination.dtype
            or source.device != destination.device
            or locations.device != destination.device
        ):
            raise ValueError("Changed private/production KV geometry")
        destination.index_copy_(0, locations, source[1:])


def owned_prefill_output(
    model: Any,
    raw: Any,
    text_hidden: torch.Tensor,
    audio_hidden: torch.Tensor,
    indices: torch.Tensor,
    batch: Any,
    length: int,
) -> Any:
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
    model._last_text_hidden = text_hidden.clone()
    model._last_audio_hidden = audio_hidden.clone()
    model._last_head_indices = indices.clone()
    model._last_request_lengths = [length]
    model._last_prefix_lengths = [0]
    batch._moss_request_lengths = [length]
    batch._moss_prefix_lengths = [0]
    publish_dual_logits(model, output, batch)
    batch._moss_dual_logits = (text, audio)
    return output


@contextmanager
def preserve_publication(model: Any, runner: Any) -> Iterator[None]:
    previous = [
        (obj, {k: v for k, v in vars(obj).items() if k.startswith("_last_")})
        for obj in (model, runner)
    ]
    stash = model._dual_logits_by_rid
    saved_stash = stash.copy()
    try:
        yield
    finally:
        for obj, saved in previous:
            for key in tuple(vars(obj)):
                if key.startswith("_last_") and key not in saved:
                    delattr(obj, key)
            for key, value in saved.items():
                setattr(obj, key, value)
        stash.clear()
        stash.update(saved_stash)
        model._dual_logits_by_rid = stash


class _PrefillGraph:
    def __init__(self, model: Any, runtime: Any, runner: Any, length: int) -> None:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        self.model, self.length, self.device = (
            model,
            length,
            model.embed_tokens.weight.device,
        )
        self.req = self.pool = self.backend = self.batch = self.graph = self.embeds = (
            self.outputs
        ) = None
        self.text_hidden = self.audio_hidden = self.indices = None
        self.captures = self.body_calls = self.replays = 0
        self.closed = False
        self.private_bytes = 0
        start = time.perf_counter()
        try:
            with (
                torch.cuda.device(self.device),
                torch.inference_mode(),
                projection_mode(model, False),
            ):
                self.req = ReqToTokenPool(
                    size=1,
                    max_context_len=length,
                    device=runtime.device,
                    enable_memory_saver=False,
                )
                self.pool = MHATokenToKVPool(
                    size=length,
                    page_size=1,
                    dtype=torch.bfloat16,
                    head_num=8,
                    head_dim=128,
                    layer_num=40,
                    device=runtime.device,
                    enable_memory_saver=False,
                )
                locations = torch.arange(
                    1, length + 1, device=self.device, dtype=torch.int64
                )
                self.req.req_to_token[1].copy_(locations)
                self.backend = MossSpeechTorchNativeBackend(
                    SimpleNamespace(
                        device=runtime.device,
                        req_to_token_pool=self.req,
                        token_to_kv_pool=self.pool,
                    )
                )
                self.batch = ForwardBatch(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=1,
                    input_ids=torch.zeros(
                        length, device=self.device, dtype=torch.int64
                    ),
                    positions=torch.arange(
                        length, device=self.device, dtype=torch.int64
                    ),
                    req_pool_indices=torch.ones(
                        1, device=self.device, dtype=torch.int64
                    ),
                    seq_lens=torch.tensor(
                        [length], device=self.device, dtype=torch.int64
                    ),
                    out_cache_loc=locations,
                    seq_lens_sum=length,
                    seq_lens_cpu=[length],
                    extend_num_tokens=length,
                    extend_seq_lens=torch.tensor(
                        [length], device=self.device, dtype=torch.int64
                    ),
                    extend_prefix_lens=torch.zeros(
                        1, device=self.device, dtype=torch.int64
                    ),
                    extend_start_loc=torch.zeros(
                        1, device=self.device, dtype=torch.int64
                    ),
                    extend_seq_lens_cpu=[length],
                    extend_prefix_lens_cpu=[0],
                    rids=None,
                )
                self.backend.init_forward_metadata(self.batch)
                self.batch.extend_seq_lens = CapturedLengths(
                    self.batch.extend_seq_lens, length
                )
                rows = torch.tensor(
                    [
                        [151667 if i % 3 == 1 else 100 + i * 17, 1 + (i * 37) % 511]
                        for i in range(length)
                    ],
                    dtype=torch.long,
                )
                # Same native dual-channel embeddings; actual prompt data is absent.
                self.embeds = torch.cat([runner._rows_to_embeds(rows)], dim=0).to(
                    device=self.device, dtype=model.dtype
                )
                self.graph = torch.cuda.CUDAGraph()
                warm = torch.cuda.Stream(device=self.device)
                warm.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(warm):
                    for _ in range(2):
                        self._forward()
                torch.cuda.current_stream(self.device).wait_stream(warm)
                with torch.cuda.graph(self.graph):
                    self.outputs = self._forward()
                    self.text_hidden = model._last_text_hidden
                    self.audio_hidden = model._last_audio_hidden
                    self.indices = model._last_head_indices
                self.captures = 1
                torch.cuda.synchronize(self.device)
            if (
                self.body_calls != 3
                or self.text_hidden.shape[0] != length
                or self.audio_hidden.shape[0] != length
            ):
                raise RuntimeError("Incomplete native prefill graph")
            self._owners = (
                self.req.req_to_token,
                *self.pool.k_buffer,
                *self.pool.v_buffer,
                self.embeds,
                self.batch.positions,
                self.batch.input_ids,
                locations,
                self.outputs.text_logits,
                self.outputs.audio_logits,
                self.outputs.hidden_states,
                self.text_hidden,
                self.audio_hidden,
                self.indices,
            )
            self._signature = tuple(tensor_signature(t) for t in self._owners)
            self.private_bytes = sum(
                t.numel() * t.element_size()
                for t in (
                    self.req.req_to_token,
                    *self.pool.k_buffer,
                    *self.pool.v_buffer,
                )
            )
            self.prepare_seconds = time.perf_counter() - start
        except BaseException:
            torch.cuda.synchronize(self.device)
            self.close()
            raise

    def _forward(self) -> Any:
        self.body_calls += 1
        with scoped_backend(self.backend):
            return self.model._forward_eager(
                input_ids=self.batch.input_ids,
                positions=self.batch.positions,
                forward_batch=self.batch,
                input_embeds=self.embeds,
            )

    def accepts(self, embeddings: torch.Tensor, positions: torch.Tensor) -> bool:
        return (
            not self.closed
            and isinstance(embeddings, torch.Tensor)
            and not embeddings.requires_grad
            and all(
                isinstance(actual, torch.Tensor)
                and (actual.shape, actual.dtype, actual.device, actual.stride())
                == (target.shape, target.dtype, target.device, target.stride())
                for actual, target in (
                    (embeddings, self.embeds),
                    (positions, self.batch.positions),
                )
            )
            and tuple(tensor_signature(t) for t in self._owners) == self._signature
        )

    def replay(
        self, batch: Any, embeddings: torch.Tensor, positions: torch.Tensor, pool: Any
    ) -> Any:
        with torch.inference_mode():
            self.embeds.copy_(embeddings)
            self.batch.positions.copy_(positions)
            self.graph.replay()
            scatter_private_kv(self.pool, pool, batch.out_cache_loc, self.length)
        output = owned_prefill_output(
            self.model,
            self.outputs,
            self.text_hidden,
            self.audio_hidden,
            self.indices,
            batch,
            self.length,
        )
        self.replays += 1
        return output

    def close(self) -> None:
        if self.closed:
            return
        if self.graph is not None:
            self.graph.reset()
        self._owners = ()
        self.req = self.pool = self.backend = self.batch = self.graph = self.embeds = (
            self.outputs
        ) = None
        self.text_hidden = self.audio_hidden = self.indices = None
        self.closed = True


class MossSpeechPrefillGraphManager:
    def __init__(self, model: Any, runtime: Any) -> None:
        self.model, self.runtime = model, runtime
        self.device = model.embed_tokens.weight.device
        self.backend, self.pool, self.request_pool = (
            runtime.attn_backend,
            runtime.token_to_kv_pool,
            runtime.req_to_token_pool,
        )
        self.cache = PrefillGraphCache(
            synchronize=lambda: torch.cuda.synchronize(self.device)
        )
        self.hits = self.fallbacks = self.scatter_calls = 0
        self.startup_seconds = 0.0
        self.reason = "not initialized"
        self.memory: dict[str, int] = {}
        self._epoch: tuple = ()
        self._parameter_owners: tuple = ()
        self._rope_owners: tuple = ()
        self._rope_layers: tuple = ()
        self._rope_storage: tuple = ()

    def _qualified(self) -> bool:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        from .sglang_model import MossSpeechSGLangModel

        pool = self.pool
        if (
            getattr(self.model._forward_eager, "__func__", None)
            is not MossSpeechSGLangModel._forward_eager
            or getattr(self.model.compute_dual_logits, "__func__", None)
            is not MossSpeechSGLangModel.compute_dual_logits
        ):
            return False
        if (
            type(self.model) is not MossSpeechSGLangModel
            or type(self.backend) is not MossSpeechTorchNativeBackend
            or type(pool) is not MHATokenToKVPool
            or self.model.training
            or self.device.type != "cuda"
            or torch.__version__ != "2.9.1+cu128"
            or torch.version.cuda != "12.8"
            or os.environ.get("MOSS_DUMP_DIR")
        ):
            return False
        if (
            torch.cuda.get_device_name(self.device) != "NVIDIA A800-SXM4-80GB"
            or torch.cuda.get_device_capability(self.device) != (8, 0)
            or self.backend.use_sliding_window_kv_pool
        ):
            return False
        if (
            (
                pool.page_size,
                pool.dtype,
                pool.head_num,
                pool.head_dim,
                pool.layer_num,
                pool.kv_cache_layout,
            )
            != (1, torch.bfloat16, 8, 128, 40, "nhd")
            or len(pool.k_buffer) != 40
            or len(pool.v_buffer) != 40
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
        for layer in layers:
            attn = layer.self_attn.attn
            if (
                (
                    attn.tp_q_head_num,
                    attn.tp_k_head_num,
                    attn.qk_head_dim,
                    attn.v_head_dim,
                )
                != (32, 8, 128, 128)
                or attn.is_cross_attention
                or getattr(attn, "sliding_window_size", -1) not in (-1, None)
            ):
                return False
        return (
            self.model.embed_tokens.weight.dtype == torch.bfloat16
            and native_norm_parameter_profile(self.model)
        )

    def _parameter_epoch(self) -> tuple:
        # Registration identities detect replacement, versions detect in-place updates.
        # Mutable decode feedback storage is not consumed by native prefill embeds.
        return tuple(
            (
                id(module),
                module.training,
                tuple(
                    (key, tensor_signature(t) if t is not None else None)
                    for key, t in (
                        *module._parameters.items(),
                        *module._buffers.items(),
                    )
                ),
            )
            for module in self._modules
            if module is not self.model._decode_input_embedding
        )

    def _rope_epoch(self) -> tuple:
        return tuple(
            tensor_signature(layer._projection_inv_freq) for layer in self._rope_layers
        )

    def _pool_signature(self) -> tuple:
        return tuple(
            (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
            for t in (
                self.request_pool.req_to_token,
                *self.pool.k_buffer,
                *self.pool.v_buffer,
            )
        )

    def initialize(self, runner: Any) -> None:
        with self.cache.lock:
            if self.cache._ready or self.cache._closed:
                return
            if not self._qualified():
                self.reason = "unsupported native prefill runtime"
                return
            self._modules = tuple(self.model.modules())
            self._registrations = tuple(
                (m, tuple(m._modules.items())) for m in self._modules
            )
            self._epoch = self._parameter_epoch()
            self._parameter_owners = tuple(
                t
                for module in self._modules
                if module is not self.model._decode_input_embedding
                for t in (*module._parameters.values(), *module._buffers.values())
                if t is not None
            )
            self._rope_layers = (
                tuple(self.model.layers)
                + tuple(self.model.text_block)
                + tuple(self.model.audio_block)
            )
            self._policy = execution_policy()
            self._pool_storage = self._pool_signature()
            self._eager = getattr(self.model._forward_eager, "__func__", None)
            self._heads = getattr(self.model.compute_dual_logits, "__func__", None)
            self._stream = torch.cuda.current_stream(self.device).cuda_stream
            start = time.perf_counter()
            self.memory = dict(
                allocated_before=torch.cuda.memory_allocated(self.device),
                reserved_before=torch.cuda.memory_reserved(self.device),
            )
            torch.cuda.reset_peak_memory_stats(self.device)
            try:
                with (
                    torch.cuda.device(self.device),
                    torch.inference_mode(),
                    torch.random.fork_rng(devices=[self.device.index]),
                    preserve_publication(self.model, runner),
                ):
                    self.cache.initialize(
                        lambda n: _PrefillGraph(self.model, self.runtime, runner, n)
                    )
                torch.cuda.synchronize(self.device)
                self._rope_owners = tuple(
                    layer._projection_inv_freq for layer in self._rope_layers
                )
                self._rope_storage = self._rope_epoch()
                if self._parameter_epoch() != self._epoch:
                    self.cache.close()
                    raise RuntimeError(
                        "Native prefill preparation changed model parameters or buffers"
                    )
                self.reason = "ready"
            except torch.cuda.OutOfMemoryError:
                self.cache.close()
                self.reason = "insufficient memory during all-length startup capture"
                logger.warning("MOSS prefill graphs disabled: %s", self.reason)
            finally:
                self.startup_seconds = time.perf_counter() - start
                self.memory.update(
                    allocated_after=torch.cuda.memory_allocated(self.device),
                    reserved_after=torch.cuda.memory_reserved(self.device),
                    peak_allocated=torch.cuda.max_memory_allocated(self.device),
                    private_bytes=sum(
                        g.private_bytes for g in self.cache._entries.values()
                    ),
                )
            logger.info("MOSS prefill graphs: %s", self.status())

    def _unchanged(self) -> bool:
        return (
            self.runtime.attn_backend is self.backend
            and self.runtime.token_to_kv_pool is self.pool
            and self.runtime.req_to_token_pool is self.request_pool
            and self.backend.token_to_kv_pool is self.pool
            and self.backend.req_to_token_pool is self.request_pool
            and self._pool_signature() == self._pool_storage
            and all(
                tuple(m._modules.items()) == saved for m, saved in self._registrations
            )
            and self._parameter_epoch() == self._epoch
            and getattr(self.model._forward_eager, "__func__", None) is self._eager
            and getattr(self.model.compute_dual_logits, "__func__", None) is self._heads
            and self._rope_epoch() == self._rope_storage
            and execution_policy() == self._policy
            and not os.environ.get("MOSS_DUMP_DIR")
        )

    def try_replay(
        self,
        input_ids: Any,
        positions: Any,
        batch: Any,
        input_embeds: Any = None,
        **kwargs: Any,
    ) -> Any:
        del input_ids
        with self.cache.lock:

            def fallback() -> None:
                self.fallbacks += 1
                return None

            if (
                not self.cache._ready
                or self.cache._closed
                or batch is None
                or input_embeds is None
                or set(kwargs) - {"omni_prefill_rids"}
            ):
                return fallback()
            mode = getattr(batch, "forward_mode", None)
            if (
                mode is None
                or mode is not ForwardMode.EXTEND
                or batch.batch_size != 1
                or getattr(batch, "encoder_lens", None) is not None
                or getattr(batch, "spec_info", None) is not None
            ):
                return fallback()

            def cpu_list(name: str) -> list | None:
                value = getattr(batch, name, None)
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    return None
                return (
                    list(value)
                    if isinstance(value, (list, tuple, torch.Tensor))
                    else None
                )

            lengths, prefixes, totals = (
                cpu_list("extend_seq_lens_cpu"),
                cpu_list("extend_prefix_lens_cpu"),
                cpu_list("seq_lens_cpu"),
            )
            if (
                lengths is None
                or prefixes is None
                or totals is None
                or len(lengths) != 1
                or prefixes != [0]
                or totals != lengths
            ):
                return fallback()
            try:
                if isinstance(lengths[0], bool):
                    return fallback()
                n = operator.index(lengths[0])
            except TypeError:
                return fallback()
            if (
                prefill_length(is_extend=True, batch_size=1, length=n, prefix=0) is None
                or getattr(batch, "extend_num_tokens", None) != n
            ):
                return fallback()
            rids = getattr(batch, "rids", None)
            if rids is not None and len(rids) != 1:
                return fallback()
            loc = getattr(batch, "out_cache_loc", None)
            if (
                not isinstance(loc, torch.Tensor)
                or loc.shape != (n,)
                or loc.dtype != torch.int64
                or loc.device != self.device
                or loc.stride() != (1,)
            ):
                return fallback()
            if (
                torch.cuda.current_device() != self.device.index
                or torch.cuda.current_stream(self.device).cuda_stream != self._stream
                or not context_api.has_forward_context()
                or context_api.get_attn_backend() is not self.backend
                or not self._unchanged()
            ):
                return fallback()
            entry = self.cache._entries[n]
            if not entry.accepts(input_embeds, positions):
                return fallback()
            output = self.cache.replay(n, batch, input_embeds, positions, self.pool)
            if output is not None:
                self.hits += 1
                self.scatter_calls += 80
            return output

    def status(self) -> dict[str, Any]:
        with self.cache.lock:
            entries = self.cache._entries
            return dict(
                enabled=self.cache._ready and not self.cache._closed,
                closed=self.cache._closed,
                graphs=len(entries),
                lengths=list(entries),
                captures=sum(g.captures for g in entries.values()),
                body_calls=sum(g.body_calls for g in entries.values()),
                replays=self.hits,
                fallbacks=self.fallbacks,
                scatter_calls=self.scatter_calls,
                startup_seconds=self.startup_seconds,
                memory=dict(self.memory),
                reason=self.reason,
            )

    def close(self) -> None:
        with self.cache.lock, torch.cuda.device(self.device):
            if getattr(self, "_close_logged", False):
                return
            receipt = self.status()
            self.cache.close()
            self._parameter_owners = self._rope_owners = ()
            # Retain actual startup counts before cache.close releases entries.
            # A failed synchronization leaves owners intact and emits no receipt.
            receipt.update(
                enabled=False,
                closed=self.cache._closed,
                remaining_graphs=self.cache.size,
            )
            logger.info("MOSS prefill graphs closed: %s", receipt)
            self._close_logged = True
