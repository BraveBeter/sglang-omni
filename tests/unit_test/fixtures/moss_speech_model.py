# SPDX-License-Identifier: Apache-2.0
"""Small native MOSS models and scoped attention pools for regression tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import NotRequired, TypedDict
from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_parallel

from sglang_omni.models.moss_speech.decode_graphs import DecodeEntry, DecodeGraph
from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig
from sglang_omni.models.moss_speech.model_layers import MossSpeechRMSNorm
from sglang_omni.models.moss_speech.model_runner import MossSpeechModelRunner
from sglang_omni.models.moss_speech.native_norm import NativeNorm
from sglang_omni.models.moss_speech.sglang_model import (
    MossSpeechModelOutput,
    MossSpeechSGLangModel,
)


class ARPathProof(TypedDict):
    executions: int
    complete_heads_exact: bool
    all_kv_exact: bool
    captures: NotRequired[int]
    replays: NotRequired[int]
    native_capture_calls: NotRequired[int]


class SerialDecodeProof(TypedDict):
    batch_sizes: list[int]
    executions: dict[str, int]
    captures: int
    row_replays: int
    kv_layers: int
    complete_heads_exact: bool
    complete_hidden_exact: bool
    all_kv_exact: bool
    ownership_exact: bool
    all_row_rejection_exact: bool
    all_row_rejection_count: int
    runtime_guards_exact: bool
    runtime_guard_count: int
    boundary_fallback_exact: bool
    boundary_lengths: list[int]
    no_partial_retry: bool
    exactness_cases: int
    native_capture_calls: int


class CumulativeARProof(TypedDict):
    optimizations: dict[str, ARPathProof]
    complete_heads_exact: bool
    all_kv_exact: bool
    joint_optimizations: list[str]
    joint_complete_heads_exact: bool
    joint_all_kv_exact: bool
    serial_decode_graph: NotRequired[SerialDecodeProof]


@dataclass(kw_only=True)
class AttentionPools:
    device: str
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MHATokenToKVPool


class RowCommitRunner(MossSpeechModelRunner):
    """Replace engine allocation while retaining the real row-commit methods."""

    def __init__(self, model: MossSpeechSGLangModel) -> None:
        self.model = model


@contextmanager
def native_runtime(device: torch.device) -> Iterator[None]:
    with (
        get_context().override_server_args(
            device=device.type,
            dtype="bfloat16",
            attention_backend="torch_native",
            disable_cuda_graph=True,
            enable_torch_compile=False,
            disable_radix_cache=True,
            tp_size=1,
        ),
        get_parallel().override(
            tp_rank=0,
            pp_rank=0,
            attn_tp_rank=0,
            attn_dp_rank=0,
            attn_cp_rank=0,
            moe_tp_rank=0,
            moe_dp_rank=0,
            moe_ep_rank=0,
            tp_group=None,
            attn_tp_group=None,
        ),
        torch.device(device),
        torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []
        ),
    ):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            yield
        finally:
            torch.set_default_dtype(previous_dtype)


def tiny_model(
    device: torch.device, *, audio_vocab_size: int = 64
) -> MossSpeechSGLangModel:
    configuration = MossSpeechConfig(
        vocab_size=256,
        audio_vocab_size=audio_vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_shared_layers=2,
        num_modality_layers=1,
        num_hidden_layers=3,
        max_position_embeddings=16,
        moss_decode_buffer_bs=2,
    )
    model = MossSpeechSGLangModel(configuration, init_device=device)
    generator = torch.Generator(device=device).manual_seed(183)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.2, generator=generator)
    return model


def checkpoint_weights(
    configuration: MossSpeechConfig,
) -> list[tuple[str, torch.Tensor]]:
    """Construct the external checkpoint schema without the native name mapper."""
    assert configuration.audio_vocab_size is not None
    hidden_size = configuration.hidden_size
    query_rows = configuration.num_attention_heads * configuration.head_dim
    key_value_rows = configuration.num_key_value_heads * configuration.head_dim
    shapes: list[tuple[str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", (configuration.vocab_size, hidden_size)),
        ("model.audio_embed.weight", (configuration.audio_vocab_size, hidden_size)),
        ("model.text_norm.weight", (hidden_size,)),
        ("model.audio_norm.weight", (hidden_size,)),
        ("text_lm_head.weight", (configuration.vocab_size, hidden_size)),
        ("audio_lm_head.weight", (configuration.audio_vocab_size, hidden_size)),
    ]
    layer_shapes = [
        ("self_attn.q_proj.weight", (query_rows, hidden_size)),
        ("self_attn.k_proj.weight", (key_value_rows, hidden_size)),
        ("self_attn.v_proj.weight", (key_value_rows, hidden_size)),
        ("self_attn.o_proj.weight", (hidden_size, query_rows)),
        ("self_attn.q_norm.weight", (configuration.head_dim,)),
        ("self_attn.k_norm.weight", (configuration.head_dim,)),
        ("mlp.gate_proj.weight", (configuration.intermediate_size, hidden_size)),
        ("mlp.up_proj.weight", (configuration.intermediate_size, hidden_size)),
        ("mlp.down_proj.weight", (hidden_size, configuration.intermediate_size)),
        ("input_layernorm.weight", (hidden_size,)),
        ("post_attention_layernorm.weight", (hidden_size,)),
    ]
    for block, depth in (
        ("shared_block", configuration.num_shared_layers),
        ("text_block", configuration.num_modality_layers),
        ("audio_block", configuration.num_modality_layers),
    ):
        for layer_index in range(depth):
            shapes.extend(
                (f"model.{block}.layers.{layer_index}.{leaf}", shape)
                for leaf, shape in layer_shapes
            )
    return [
        (name, torch.full(shape, index + 1, dtype=torch.bfloat16))
        for index, (name, shape) in enumerate(shapes)
    ]


def prefill(
    model: MossSpeechSGLangModel,
    prompt_embeddings: list[torch.Tensor],
    request_ids: list[str],
    *,
    max_context_len: int = 16,
) -> tuple[MossSpeechModelOutput, TorchNativeAttnBackend]:
    device = model.embed_tokens.weight.device
    token_lengths = [embeddings.shape[0] for embeddings in prompt_embeddings]
    token_count = sum(token_lengths)
    request_pool = ReqToTokenPool(
        size=len(token_lengths),
        max_context_len=max_context_len,
        device=str(device),
        enable_memory_saver=False,
    )
    kv_pool = MHATokenToKVPool(
        size=max(32, 2 * max_context_len),
        page_size=1,
        dtype=torch.bfloat16,
        head_num=model.config.num_key_value_heads,
        head_dim=model.head_dim,
        layer_num=model.total_attention_layers,
        device=str(device),
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    next_slot = 1
    for request_index, token_length in enumerate(token_lengths, 1):
        request_pool.req_to_token[request_index, :token_length] = torch.arange(
            next_slot, next_slot + token_length, device=device
        )
        next_slot += token_length
    backend = TorchNativeAttnBackend(
        AttentionPools(
            device=str(device),
            req_to_token_pool=request_pool,
            token_to_kv_pool=kv_pool,
        )
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(token_lengths),
        input_ids=torch.zeros(token_count, dtype=torch.long, device=device),
        req_pool_indices=torch.arange(1, len(token_lengths) + 1, device=device),
        seq_lens=torch.tensor(token_lengths, dtype=torch.int32, device=device),
        out_cache_loc=torch.arange(1, token_count + 1, device=device),
        seq_lens_sum=token_count,
        positions=torch.cat(
            [torch.arange(length, device=device) for length in token_lengths]
        ),
        extend_seq_lens=torch.tensor(token_lengths, dtype=torch.int32, device=device),
        extend_prefix_lens=torch.zeros(
            len(token_lengths), dtype=torch.int32, device=device
        ),
        extend_seq_lens_cpu=token_lengths,
        extend_prefix_lens_cpu=[0] * len(token_lengths),
        extend_num_tokens=token_count,
        rids=request_ids,
    )
    backend.init_forward_metadata(batch)
    with forward_context(ForwardContext(attn_backend=backend)):
        output = model(
            batch.input_ids,
            batch.positions,
            batch,
            input_embeds=torch.cat(prompt_embeddings),
        )
    return output, backend


def decode(
    model: MossSpeechSGLangModel,
    backend: TorchNativeAttnBackend,
    prompt_lengths: list[int],
    feedback_embeddings: torch.Tensor,
    request_ids: list[str],
) -> MossSpeechModelOutput:
    device = model.embed_tokens.weight.device
    batch_size = len(prompt_lengths)
    token_slots = torch.arange(
        sum(prompt_lengths) + 1,
        sum(prompt_lengths) + 1 + batch_size,
        device=device,
    )
    for request_index, token_length in enumerate(prompt_lengths, 1):
        backend.req_to_token_pool.req_to_token[request_index, token_length] = (
            token_slots[request_index - 1]
        )
    model.decode_input_embedding.weight[:batch_size].copy_(feedback_embeddings)
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=torch.arange(batch_size, device=device),
        req_pool_indices=torch.arange(1, batch_size + 1, device=device),
        seq_lens=torch.tensor(
            [length + 1 for length in prompt_lengths], dtype=torch.int32, device=device
        ),
        out_cache_loc=token_slots,
        seq_lens_cpu=torch.tensor(
            [length + 1 for length in prompt_lengths], device="cpu"
        ),
        seq_lens_sum=sum(prompt_lengths) + batch_size,
        positions=torch.tensor(prompt_lengths, device=device),
        rids=request_ids,
    )
    backend.init_forward_metadata(batch)
    with forward_context(ForwardContext(attn_backend=backend)):
        return model(batch.input_ids, batch.positions, batch)


def assert_full_decode_graph(model: MossSpeechSGLangModel, *, capacity=16) -> dict:
    """Replay full dual heads and all KV layers, including output reuse and close."""
    device = model.embed_tokens.weight.device
    prompt = model.embed_tokens(torch.tensor([1, 2, 3], device=device))
    _, backend = prefill(model, [prompt], ["prompt"], max_context_len=capacity)
    graph_lengths = tuple(range(1, min(capacity, 512) + 1))
    manager = DecodeGraph(model, backend, lengths=graph_lengths)
    model.decode_graph = manager
    try:
        memory_before = {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
        }
        started = perf_counter()
        manager.initialize()
        startup_seconds = perf_counter() - started
        memory_after = {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
        }
        assert len(manager.entries) == len(graph_lengths), manager.reason
        snapshots = []
        generator = torch.Generator(device=device).manual_seed(9901)
        lengths = ((512, 1023) if capacity > 512 else ()) + tuple(
            n - 1 for n in graph_lengths
        )
        for length in lengths:
            # Fill all earlier KV rows deterministically, including unused-gap rows.
            backend.req_to_token_pool.req_to_token[1, :length] = torch.arange(
                1, length + 1, device=device
            )
            for cache in (
                *backend.token_to_kv_pool.k_buffer,
                *backend.token_to_kv_pool.v_buffer,
            ):
                cache[1 : length + 1].normal_(mean=0.0, std=0.125, generator=generator)
            feedback = model.embed_tokens(torch.tensor([length], device=device))
            model.decode_graph = None
            expected = decode(model, backend, [length], feedback, ["eager"])
            expected_kv = [
                t.clone()
                for t in (
                    *backend.token_to_kv_pool.k_buffer,
                    *backend.token_to_kv_pool.v_buffer,
                )
            ]
            model.decode_graph = manager
            actual = decode(model, backend, [length], feedback, ["graph"])
            for name in ("text_logits", "audio_logits", "hidden_states"):
                torch.testing.assert_close(
                    getattr(actual, name), getattr(expected, name), rtol=0, atol=0
                )
            for actual_kv, reference in zip(
                (
                    *backend.token_to_kv_pool.k_buffer,
                    *backend.token_to_kv_pool.v_buffer,
                ),
                expected_kv,
            ):
                torch.testing.assert_close(
                    actual_kv[1 : length + 2], reference[1 : length + 2], rtol=0, atol=0
                )
            snapshots.append((actual, actual.text_logits.clone()))
        assert manager.hits == len(graph_lengths)
        for actual, saved in snapshots:
            torch.testing.assert_close(actual.text_logits, saved, rtol=0, atol=0)
        # Sampling may mutate an owned output without poisoning the next replay.
        snapshots[-1][0].text_logits.fill_(float("-inf"))
        again = decode(model, backend, [lengths[-1]], feedback, ["reuse"])
        torch.testing.assert_close(
            again.text_logits, expected.text_logits, rtol=0, atol=0
        )
        observed = []
        handle = model.layers[0].register_forward_pre_hook(
            lambda module, args: observed.append(len(args))
        )
        previous_hits = manager.hits
        try:
            hooked = decode(model, backend, [lengths[-1]], feedback, ["hooked"])
            torch.testing.assert_close(
                hooked.text_logits, expected.text_logits, rtol=0, atol=0
            )
            assert observed == [5] and manager.hits == previous_hits
        finally:
            handle.remove()
        manager.close()
        manager.close()
        after = decode(model, backend, [lengths[-1]], feedback, ["closed"])
        torch.testing.assert_close(
            after.audio_logits, expected.audio_logits, rtol=0, atol=0
        )
        original_body = DecodeEntry.body

        def fail_warmup(entry: DecodeEntry) -> MossSpeechModelOutput:
            original_body(entry)
            raise RuntimeError("injected warmup failure")

        failed = DecodeGraph(model, backend, lengths=(1,))
        with (
            patch.object(DecodeEntry, "body", fail_warmup),
            patch(
                "torch.cuda.synchronize", wraps=torch.cuda.synchronize
            ) as synchronize,
            pytest.raises(RuntimeError, match="injected warmup failure"),
        ):
            failed.initialize()
        assert synchronize.call_count > 0
        assert failed.closed and not failed.entries
        return {
            "captures": len(graph_lengths),
            "startup_seconds": startup_seconds,
            "memory_before_bytes": memory_before,
            "memory_after_bytes": memory_after,
            "replays": manager.hits,
            "complete_heads_exact": True,
            "kv_layers": model.total_attention_layers,
            "owned_output_and_close": True,
            "partial_warmup_failure_synchronized": True,
            "failed_warmup_synchronize_calls": synchronize.call_count,
        }
    finally:
        manager.close()
        model.decode_graph = None
        model.dual_logits_by_request_id.clear()


def assert_serial_decode_graph(
    model: MossSpeechSGLangModel,
    *,
    capacity: int = 1024,
    native_norm_library: str | None = None,
    require_native_execution: bool = True,
) -> SerialDecodeProof:
    """Qualify actual serial dispatch, complete outputs, KV ownership and fallbacks."""
    original_embedding = model.decode_input_embedding
    original_embedding_contents = original_embedding.weight.clone()
    had_decode_graph = "decode_graph" in model.__dict__
    original_decode_graph = model.__dict__.get("decode_graph")
    original_logits = dict(model.dual_logits_by_request_id)
    original_training = [(module, module.training) for module in model.modules()]
    norms = [
        module for module in model.modules() if isinstance(module, MossSpeechRMSNorm)
    ]
    previous_native = [module.native_norm for module in norms]
    native = NativeNorm(native_norm_library) if native_norm_library else None
    try:
        for module in norms:
            module.native_norm = native
        device = model.embed_tokens.weight.device
        if model.decode_input_embedding.num_embeddings < 4:
            model.decode_input_embedding = torch.nn.Embedding(
                4, model.hidden_size, device=device, dtype=torch.bfloat16
            )
        else:
            pass
        request_pool = ReqToTokenPool(
            size=8,
            max_context_len=capacity,
            device=str(device),
            enable_memory_saver=False,
        )
        kv_pool = MHATokenToKVPool(
            size=4 * capacity + 8,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=model.config.num_key_value_heads,
            head_dim=model.head_dim,
            layer_num=model.total_attention_layers,
            device=str(device),
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        backend = TorchNativeAttnBackend(
            AttentionPools(
                device=str(device),
                req_to_token_pool=request_pool,
                token_to_kv_pool=kv_pool,
            )
        )
        graph_lengths = tuple(
            length for length in (1, 2, 8, 32, 128, 511, 512) if length <= capacity
        )
        for i in range(4):
            request_pool.req_to_token[2 * i + 1, :capacity] = torch.arange(
                1 + i * capacity, 1 + (i + 1) * capacity, device=device
            )
        generator = torch.Generator(device=device).manual_seed(31508)
        caches = (*kv_pool.k_buffer, *kv_pool.v_buffer)
        for cache in caches:
            cache.normal_(mean=0, std=0.125, generator=generator)
        model.decode_input_embedding.weight[:4].copy_(
            model.embed_tokens(torch.tensor([1, 2, 3, 4], device=device))
        )
        embedding_snapshot = model.decode_input_embedding.weight.clone()
        decode_graph = DecodeGraph(
            model, backend, lengths=graph_lengths, serial_batch_replay=True
        )
        model.decode_graph = decode_graph

        def make_batch(order: list[int], token_lengths: list[int]) -> ForwardBatch:
            batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=len(order),
                input_ids=torch.tensor(order, device=device),
                positions=torch.tensor(
                    [length - 1 for length in token_lengths], device=device
                ),
                req_pool_indices=torch.tensor(
                    [2 * i + 1 for i in order], device=device
                ),
                out_cache_loc=torch.tensor(
                    [i * capacity + length for i, length in zip(order, token_lengths)],
                    device=device,
                ),
                seq_lens=torch.tensor(token_lengths, dtype=torch.int32, device=device),
                seq_lens_cpu=torch.tensor(token_lengths, device="cpu"),
                seq_lens_sum=sum(token_lengths),
                rids=[f"serial-request-{i}" for i in order],
            )
            backend.init_forward_metadata(batch)
            return batch

        try:
            decode_graph.initialize()
            assert len(decode_graph.entries) == len(graph_lengths), decode_graph.reason
            if native is not None and require_native_execution:
                assert decode_graph.native_capture_calls > 0
            else:
                pass
            snapshots = [cache.clone() for cache in caches]
            cases = [
                (list(range(count)), [length] * count)
                for count in (1, 2, 4)
                for length in graph_lengths
            ]
            cases.extend(
                (
                    list(reversed(range(count))),
                    [graph_lengths[i % len(graph_lengths)] for i in range(count)],
                )
                for count in (2, 4)
            )
            retained_outputs = []
            with forward_context(ForwardContext(attn_backend=backend)):
                for order, token_lengths in cases:
                    batch = make_batch(order, token_lengths)
                    for cache, saved in zip(caches, snapshots):
                        cache.copy_(saved)
                    for module in norms:
                        module.native_norm = None
                    try:
                        expected = model.forward_eager(
                            batch.input_ids, batch.positions, batch
                        )
                    finally:
                        for module in norms:
                            module.native_norm = native
                    expected_caches = [cache.clone() for cache in caches]
                    for cache, saved in zip(caches, snapshots):
                        cache.copy_(saved)
                    actual = model(batch.input_ids, batch.positions, batch)
                    for name in ("text_logits", "audio_logits", "hidden_states"):
                        torch.testing.assert_close(
                            getattr(actual, name),
                            getattr(expected, name),
                            rtol=0,
                            atol=0,
                        )
                    assert all(
                        torch.equal(cache, saved)
                        for cache, saved in zip(caches, expected_caches)
                    )
                    assert torch.equal(
                        model.decode_input_embedding.weight, embedding_snapshot
                    )
                    retained_outputs.append((actual, actual.audio_logits.clone()))
                    actual.text_logits.fill_(float("-inf"))
                    assert torch.isfinite(actual.next_token_logits).all()
                    for i, request_id in enumerate(batch.rids):
                        cached_text, cached_audio = model.dual_logits_by_request_id[
                            (request_id, "decode")
                        ]
                        torch.testing.assert_close(
                            cached_text, expected.text_logits[i], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            cached_audio, expected.audio_logits[i], rtol=0, atol=0
                        )
                assert all(
                    torch.equal(output.audio_logits, saved)
                    for output, saved in retained_outputs
                )
                boundary_lengths = (513, 40960)
                rejected_before_mutation = 0
                for length in boundary_lengths:
                    batch = make_batch(
                        [0, 1, 2, 3], [8 if 8 in graph_lengths else 2] * 4
                    )
                    batch.seq_lens_cpu[-1] = length
                    before = [cache.clone() for cache in caches]
                    assert (
                        decode_graph.try_replay(batch.input_ids, batch.positions, batch)
                        is None
                    )
                    assert all(
                        torch.equal(cache, saved)
                        for cache, saved in zip(caches, before)
                    )
                    rejected_before_mutation += 1
                for invalid in ("alias", "dtype", "missing_entry"):
                    batch = make_batch([0, 1, 2, 3], [graph_lengths[0]] * 4)
                    if invalid == "alias":
                        batch.out_cache_loc[-1] = batch.out_cache_loc[0]
                    elif invalid == "dtype":
                        batch.input_ids = batch.input_ids.to(torch.int32)
                    else:
                        batch.seq_lens_cpu[-1] = 7
                    before = [cache.clone() for cache in caches]
                    assert (
                        decode_graph.try_replay(batch.input_ids, batch.positions, batch)
                        is None
                    )
                    assert all(
                        torch.equal(cache, saved)
                        for cache, saved in zip(caches, before)
                    )
                    rejected_before_mutation += 1
                batch = make_batch([0, 1, 2, 3], [graph_lengths[0]] * 4)
                before = [cache.clone() for cache in caches]
                original_epsilon = norms[0].variance_epsilon
                norms[0].variance_epsilon = original_epsilon * 2
                try:
                    assert (
                        decode_graph.try_replay(batch.input_ids, batch.positions, batch)
                        is None
                    )
                finally:
                    norms[0].variance_epsilon = original_epsilon
                if native is not None:
                    with patch.object(norms[0], "native_norm", None):
                        assert (
                            decode_graph.try_replay(
                                batch.input_ids, batch.positions, batch
                            )
                            is None
                        )
                    with patch.object(native, "library", None):
                        assert (
                            decode_graph.try_replay(
                                batch.input_ids, batch.positions, batch
                            )
                            is None
                        )
                else:
                    pass
                handle = model.layers[0].register_forward_pre_hook(lambda *_: None)
                try:
                    assert (
                        decode_graph.try_replay(batch.input_ids, batch.positions, batch)
                        is None
                    )
                finally:
                    handle.remove()
                signature = decode_graph.signature
                decode_graph.signature = None
                try:
                    assert (
                        decode_graph.try_replay(batch.input_ids, batch.positions, batch)
                        is None
                    )
                finally:
                    decode_graph.signature = signature
                model.train()
                try:
                    assert (
                        decode_graph.try_replay(batch.input_ids, batch.positions, batch)
                        is None
                    )
                finally:
                    model.eval()
                with torch.cuda.stream(torch.cuda.Stream(device=device)):
                    assert (
                        decode_graph.try_replay(batch.input_ids, batch.positions, batch)
                        is None
                    )
                with forward_context(ForwardContext(attn_backend=backend)):
                    decode_graph.closed = True
                    try:
                        assert (
                            decode_graph.try_replay(
                                batch.input_ids, batch.positions, batch
                            )
                            is None
                        )
                    finally:
                        decode_graph.closed = False
                with forward_context(ForwardContext(attn_backend=SimpleNamespace())):
                    assert (
                        decode_graph.try_replay(batch.input_ids, batch.positions, batch)
                        is None
                    )
                assert all(
                    torch.equal(cache, saved) for cache, saved in zip(caches, before)
                )
                entry = decode_graph.entries[graph_lengths[0]]
                replay_entry = entry.replay
                attempts = 0

                def fail_second(
                    inputs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
                ) -> MossSpeechModelOutput:
                    nonlocal attempts
                    attempts += 1
                    if attempts == 2:
                        raise RuntimeError("injected partial serial replay failure")
                    else:
                        return replay_entry(inputs)

                with (
                    patch.object(entry, "replay", side_effect=fail_second),
                    patch.object(
                        model, "forward_eager", wraps=model.forward_eager
                    ) as eager,
                    pytest.raises(RuntimeError, match="partial serial replay failure"),
                ):
                    model(batch.input_ids, batch.positions, batch)
                assert attempts == 2 and eager.call_count == 0
            assert (
                decode_graph.try_replay(batch.input_ids, batch.positions, batch) is None
            )
            executions = {
                str(batch_size): count
                for batch_size, count in decode_graph.serial_batches.items()
            }
            assert executions["2"] > 0 and executions["4"] > 0
            return dict(
                batch_sizes=[2, 4],
                executions=executions,
                captures=len(graph_lengths),
                row_replays=decode_graph.hits,
                kv_layers=model.total_attention_layers,
                complete_heads_exact=True,
                complete_hidden_exact=True,
                all_kv_exact=True,
                ownership_exact=True,
                all_row_rejection_exact=True,
                all_row_rejection_count=rejected_before_mutation,
                runtime_guards_exact=True,
                runtime_guard_count=10 if native is not None else 8,
                boundary_fallback_exact=True,
                boundary_lengths=list(boundary_lengths),
                no_partial_retry=True,
                exactness_cases=len(cases),
                native_capture_calls=decode_graph.native_capture_calls,
            )
        finally:
            decode_graph.close()
            decode_graph.close()
    finally:
        for module, previous in zip(norms, previous_native, strict=True):
            module.native_norm = previous
        model.decode_input_embedding = original_embedding
        original_embedding.weight.copy_(original_embedding_contents)
        if had_decode_graph:
            model.decode_graph = original_decode_graph
        else:
            model.__dict__.pop("decode_graph", None)
        model.dual_logits_by_request_id.clear()
        model.dual_logits_by_request_id.update(original_logits)
        for module, training in original_training:
            module.training = training


def assert_cumulative_ar(
    model: MossSpeechSGLangModel,
    optimizations: list[str],
    projection_batch_size: int = 0,
    graph_lengths: tuple[int, ...] = (),
    native_norm_library: str | None = None,
) -> CumulativeARProof:
    """Qualify jointly captured native norms and serial row replay against Torch."""
    selected = set(optimizations) & {
        "ar-decode-graph",
        "ar-serial-decode-graph",
        "native-norm",
    }
    assert selected == {"ar-decode-graph", "ar-serial-decode-graph", "native-norm"}
    assert projection_batch_size == 0 and not graph_lengths
    assert native_norm_library
    serial = assert_serial_decode_graph(
        model, capacity=1024, native_norm_library=native_norm_library
    )
    serial_calls = sum(serial["executions"].values())
    b1_calls = serial["row_replays"] - sum(
        int(batch) * count for batch, count in serial["executions"].items()
    )
    assert b1_calls > 0 and serial_calls > 0 and serial["native_capture_calls"] > 0
    executions = {
        "ar-decode-graph": b1_calls,
        "ar-serial-decode-graph": serial_calls,
        "native-norm": serial["native_capture_calls"],
    }
    return {
        "optimizations": {
            feature: {
                "executions": executions[feature],
                "complete_heads_exact": True,
                "all_kv_exact": True,
            }
            for feature in sorted(selected)
        },
        "complete_heads_exact": True,
        "all_kv_exact": True,
        "joint_optimizations": sorted(selected),
        "joint_complete_heads_exact": True,
        "joint_all_kv_exact": True,
        "serial_decode_graph": serial,
    }
